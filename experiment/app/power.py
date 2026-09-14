"""電池切れの予兆: ロボットの稼働時間と Raspberry Pi の低電圧アラームを ssh で読む。

残量そのものはロボットのどこからも読めない(BMS は LED を光らせるだけ)。代わりに
- 稼働時間(`/proc/uptime`): 充電のタイミングを考える材料。切れた時刻を CSV に残せば持ち時間が分かる
- 低電圧(`vcgencmd get_throttled` の bit0 = いま低電圧、bit16 = 起動後に一度あった): 電池が減って
  重い動作をしたときの典型的な症状
を一定間隔で取り、変化を画面(power イベント)とセッションの CSV に出す。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from .config import Settings
from .events import EventBus

log = logging.getLogger(__name__)

UNDERVOLTAGE_NOW = 0x1
THROTTLED_NOW = 0x4
UNDERVOLTAGE_OCCURRED = 0x10000
REMOTE_COMMAND = "cat /proc/uptime; vcgencmd get_throttled"


class PowerError(Exception):
    pass


@dataclass
class PowerState:
    available: bool = False
    uptime_s: float | None = None
    undervoltage_now: bool = False
    undervoltage_occurred: bool = False
    throttled_now: bool = False
    checked_at: float | None = None  # time.time()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fmt_uptime(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(max(0.0, seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}"


def parse_output(text: str) -> tuple[float, int]:
    """`cat /proc/uptime; vcgencmd get_throttled` の出力 → (稼働秒, フラグ)。"""
    uptime: float | None = None
    flags: int | None = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("throttled="):
            flags = int(line.split("=", 1)[1], 16)
        elif uptime is None and line:
            head = line.split()[0]
            try:
                uptime = float(head)
            except ValueError:
                continue
    if uptime is None or flags is None:
        raise ValueError(f"想定外の出力: {text[:80]!r}")
    return uptime, flags


def _explain_failure(code: int, stderr: str) -> str:
    if "Permission denied" in stderr:
        return "ssh 鍵がロボットに登録されていません"
    if "No route to host" in stderr or "timed out" in stderr or "Connection refused" in stderr or "Could not resolve" in stderr:
        return "ロボットに ssh で届きません"
    return f"ssh に失敗しました({code}): {stderr.strip().splitlines()[-1] if stderr.strip() else '出力なし'}"


async def run_ssh(host: str, user: str, key: str, timeout: float = 8.0) -> str:
    cmd = [
        "ssh", "-4",
        "-i", str(Path(key).expanduser()),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "ConnectTimeout=3",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "LogLevel=ERROR",
        f"{user}@{host}",
        REMOTE_COMMAND,
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise PowerError("ssh がタイムアウトしました") from None
    if proc.returncode != 0:
        raise PowerError(_explain_failure(proc.returncode or -1, err.decode(errors="replace")))
    return out.decode(errors="replace")


Runner = Callable[[str, str, str], Awaitable[str]]


class PowerMonitor:
    def __init__(
        self,
        settings_ref: Callable[[], Settings],
        bus: EventBus,
        *,
        row: Callable[..., None] | None = None,
        runner: Runner = run_ssh,
    ) -> None:
        self.settings_ref = settings_ref
        self.bus = bus
        self.row = row or (lambda kind, **fields: None)
        self.runner = runner
        self.state = PowerState()
        self._task: asyncio.Task | None = None
        self._last_error: str | None = None

    def host(self) -> str:
        return urlsplit(self.settings_ref().robot.base_url).hostname or ""

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="power-monitor")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def poll_once(self) -> PowerState:
        s = self.settings_ref().power
        if not s.enabled:
            if self.state.available or self.state.error != "無効":
                self.state = PowerState(available=False, error="無効", checked_at=time.time())
                self._publish()
            return self.state
        try:
            out = await self.runner(self.host(), s.ssh_user, s.ssh_key)
            uptime, flags = parse_output(out)
        except (PowerError, ValueError, OSError) as e:
            self._on_failure(str(e))
            return self.state
        except Exception as e:  # 想定外でもループは死なない
            log.exception("power poll failed")
            self._on_failure(f"{type(e).__name__}: {e}")
            return self.state
        self._on_reading(
            PowerState(
                available=True,
                uptime_s=uptime,
                undervoltage_now=bool(flags & UNDERVOLTAGE_NOW),
                undervoltage_occurred=bool(flags & UNDERVOLTAGE_OCCURRED),
                throttled_now=bool(flags & THROTTLED_NOW),
                checked_at=time.time(),
            )
        )
        return self.state

    # ------------------------------------------------------------ transitions
    def _on_reading(self, new: PowerState) -> None:
        old = self.state
        up = fmt_uptime(new.uptime_s)
        if not old.available:
            self.row("system", detail=f"robot uptime {up}")
        elif old.uptime_s is not None and new.uptime_s is not None and new.uptime_s < old.uptime_s - 5:
            self.row("system", detail=f"robot rebooted (uptime {up})")
        if new.undervoltage_now and not old.undervoltage_now:
            self.bus.toast("error", "ロボットの電源電圧が下がっています。充電してください")
            self.row("power", result="undervoltage", detail=f"uptime {up}")
        elif new.undervoltage_occurred and not old.undervoltage_occurred and not new.undervoltage_now:
            self.row("power", result="undervoltage_occurred", detail=f"uptime {up}")
        self.state = new
        self._last_error = None
        self._publish()

    def _on_failure(self, msg: str) -> None:
        old = self.state
        if old.available:
            self.row("power", result="lost", detail=f"last uptime {fmt_uptime(old.uptime_s)}: {msg}")
        self.state = PowerState(available=False, error=msg, checked_at=time.time(), undervoltage_occurred=old.undervoltage_occurred)
        if msg != self._last_error:  # 同じ障害で毎回書かない
            log.warning("power monitor: %s", msg)
            self._last_error = msg
        self._publish()

    def _publish(self) -> None:
        self.bus.publish("power", **self.state.to_dict())

    async def _loop(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("power monitor tick failed")
            await asyncio.sleep(max(5.0, float(self.settings_ref().power.interval_s)))
