"""接続監視。1 秒周期でデーモンの状態を取り、緑/赤ランプの状態と自動復旧を担う。

状態遷移:
  connected --失敗--> degraded --(連続 3 回失敗 or 最終成功から 3 秒)--> disconnected
  degraded --成功--> connected
  disconnected --成功--> recovering --(復旧手順 成功)--> connected
                                    --(復旧手順 失敗)--> disconnected(次の成功ポーリングで再試行)
起動直後は disconnected から始まるので、最初の成功で復旧手順(=起動時の準備)が走る。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Awaitable, Callable

from .events import EventBus
from .robot import DaemonStatus, RobotClient, RobotError

log = logging.getLogger(__name__)


class ConnState(str, Enum):
    connected = "connected"
    degraded = "degraded"
    disconnected = "disconnected"
    recovering = "recovering"


@dataclass
class ConnSnapshot:
    state: str
    reason: str
    version: str | None
    face_detected: bool
    motor_mode: str | None
    since: float
    robot_url: str
    last_ok: float | None

    def to_dict(self) -> dict:
        return asdict(self)


class ConnectionMonitor:
    def __init__(
        self,
        robot: RobotClient,
        bus: EventBus,
        recovery: Callable[[], Awaitable[None]],
        *,
        interval: float = 1.0,
        poll_timeout: float = 1.5,
        fail_threshold: int = 3,
        stale_s: float = 3.0,
        recovery_backoff_s: float = 2.0,
    ) -> None:
        self.robot = robot
        self.bus = bus
        self.recovery = recovery
        self.interval = interval
        self.poll_timeout = poll_timeout
        self.fail_threshold = fail_threshold
        self.stale_s = stale_s
        self.recovery_backoff_s = recovery_backoff_s

        self.snapshot = ConnSnapshot(
            state=ConnState.disconnected.value,
            reason="未接続",
            version=None,
            face_detected=False,
            motor_mode=None,
            since=time.time(),
            robot_url=robot.base_url,
            last_ok=None,
        )
        self.status: DaemonStatus | None = None
        self._task: asyncio.Task | None = None
        self._recovery_task: asyncio.Task | None = None
        self._consecutive_failures = 0
        self._last_ok_mono: float | None = None
        self._prev_last_alive: float | None = None
        self._last_recovery_attempt: float = -1e9
        self.wakeup = asyncio.Event()  # 即時ポーリングを促す

    # ------------------------------------------------------------ public
    @property
    def state(self) -> ConnState:
        return ConnState(self.snapshot.state)

    def is_connected(self) -> bool:
        return self.snapshot.state == ConnState.connected.value

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="connection-monitor")

    async def stop(self) -> None:
        for t in (self._task, self._recovery_task):
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._task = None
        self._recovery_task = None

    def poll_now(self) -> None:
        self.wakeup.set()

    # ------------------------------------------------------------ loop
    async def _run(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # 監視ループ自体は決して死なない
                log.exception("monitor tick failed")
            try:
                await asyncio.wait_for(self.wakeup.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass
            self.wakeup.clear()

    async def _tick(self) -> None:
        ok, reason, st = await self._poll_once()
        now = time.monotonic()
        prev_state = self.state
        if ok:
            self._consecutive_failures = 0
            self._last_ok_mono = now
            self.snapshot.last_ok = time.time()
            if st is not None:
                self.snapshot.version = st.version
                self.snapshot.motor_mode = st.motor_mode
                if st.face_detected != self.snapshot.face_detected:
                    self.snapshot.face_detected = st.face_detected
                    self.bus.publish("face", detected=st.face_detected)
            if prev_state in (ConnState.connected, ConnState.degraded):
                self._set_state(ConnState.connected, "")
            elif prev_state == ConnState.disconnected:
                self._start_recovery()
            # recovering 中は復旧タスクの完了を待つ
        else:
            self._consecutive_failures += 1
            stale = self._last_ok_mono is None or (now - self._last_ok_mono) >= self.stale_s
            if prev_state == ConnState.connected:
                self._set_state(ConnState.degraded, reason)
            if prev_state in (ConnState.connected, ConnState.degraded) and (self._consecutive_failures >= self.fail_threshold or stale):
                self._set_state(ConnState.disconnected, reason)
            elif prev_state == ConnState.recovering:
                # 復旧中に落ちた: 復旧タスクは自身のエラーで終わる。状態だけ更新
                self._set_state(ConnState.disconnected, reason)
            elif prev_state == ConnState.disconnected:
                self.snapshot.reason = reason

    async def _poll_once(self) -> tuple[bool, str, DaemonStatus | None]:
        try:
            st = await asyncio.wait_for(self.robot.daemon_status(timeout=self.poll_timeout), timeout=self.poll_timeout + 0.5)
        except asyncio.TimeoutError:
            return False, "応答なし(タイムアウト)", None
        except RobotError as e:
            return False, str(e), None
        except Exception as e:  # 想定外は理由付きで失敗扱い
            return False, f"状態取得エラー: {e}", None
        self.status = st
        healthy, reason = self._healthy(st)
        return healthy, reason, st

    def _healthy(self, st: DaemonStatus) -> tuple[bool, str]:
        if st.error:
            return False, f"デーモンエラー: {st.error}"
        if st.state != "running":
            return False, f"デーモン状態: {st.state}"
        if not st.ready:
            return False, "バックエンド未準備"
        prev, self._prev_last_alive = self._prev_last_alive, st.last_alive
        if st.last_alive is not None and prev is not None and st.last_alive <= prev:
            return False, "制御ループが止まっています"
        return True, ""

    def _set_state(self, state: ConnState, reason: str) -> None:
        if self.snapshot.state == state.value and self.snapshot.reason == reason:
            return
        self.snapshot.state = state.value
        self.snapshot.reason = reason
        self.snapshot.since = time.time()
        self.snapshot.robot_url = self.robot.base_url
        log.info("connection: %s %s", state.value, reason)
        self.bus.publish("connection", **self.snapshot.to_dict())

    # ------------------------------------------------------------ recovery
    def _start_recovery(self) -> None:
        now = time.monotonic()
        if self._recovery_task is not None and not self._recovery_task.done():
            return
        if now - self._last_recovery_attempt < self.recovery_backoff_s:
            return
        self._last_recovery_attempt = now
        self._set_state(ConnState.recovering, "復旧中…")
        self._recovery_task = asyncio.create_task(self._run_recovery(), name="connection-recovery")

    async def _run_recovery(self) -> None:
        try:
            await self.recovery()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("recovery failed: %s", e)
            self.bus.toast("error", f"復旧に失敗しました: {e}")
            if self.state == ConnState.recovering:
                self._set_state(ConnState.disconnected, f"復旧失敗: {e}")
            return
        if self.state == ConnState.recovering:
            self._set_state(ConnState.connected, "")
            self.bus.publish("recovery", ok=True)
