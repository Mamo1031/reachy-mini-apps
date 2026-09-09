"""接続監視。1 秒周期でデーモンの状態を取り、緑/赤ランプの状態と自動復旧を担う。

状態遷移:
  connected --失敗--> degraded --(連続 3 回失敗 or 最終成功から 3 秒)--> disconnected
  degraded --成功--> connected
  disconnected --成功--> recovering --(復旧手順 成功)--> connected
                                    --(復旧手順 失敗)--> disconnected(指数バックオフで再試行)
  recovering --(連続失敗が閾値到達)--> disconnected(復旧タスクは取り消す)
起動直後は disconnected から始まるので、最初の成功で復旧手順(=起動時の準備)が走る。

「操作してよいか」は can_control()(connected または degraded)、「完全に正常か」は is_connected()。
一瞬のポーリング失敗(degraded)でボタンを拒否しない。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
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
        recovery_backoff_max_s: float = 30.0,
        resting_ref: Callable[[], bool] = lambda: False,
        on_change: Callable[[str, str], None] | None = None,
    ) -> None:
        self.robot = robot
        self.bus = bus
        self.recovery = recovery
        self.interval = interval
        self.poll_timeout = poll_timeout
        self.fail_threshold = fail_threshold
        self.stale_s = stale_s
        self.recovery_backoff_s = recovery_backoff_s
        self.recovery_backoff_max_s = recovery_backoff_max_s
        self.resting_ref = resting_ref
        self.on_change = on_change  # (state, reason) → セッションログなどに記録する

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
        self._prev_alive_mono: float | None = None
        self._stalled = False
        self._last_recovery_attempt: float = -1e9
        self._recovery_failures = 0
        self._last_recovery_error: str | None = None
        self.wakeup = asyncio.Event()  # 即時ポーリングを促す

    # ------------------------------------------------------------ public
    @property
    def state(self) -> ConnState:
        return ConnState(self.snapshot.state)

    def is_connected(self) -> bool:
        return self.snapshot.state == ConnState.connected.value

    def can_control(self) -> bool:
        """ボタン操作を受け付けてよいか(一瞬の失敗 = degraded では拒否しない)。"""
        return self.snapshot.state in (ConnState.connected.value, ConnState.degraded.value)

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
                # 外部要因でモーターが切れた(ダッシュボード操作・保護機能など)→ 復旧手順で入れ直す
                if st is not None and not self.resting_ref() and st.motor_mode not in (None, "enabled"):
                    self._set_state(ConnState.disconnected, f"モーターが無効です({st.motor_mode})")
                    self._start_recovery()
            elif prev_state == ConnState.disconnected:
                self._start_recovery()
            # recovering 中は復旧タスクの完了を待つ
        else:
            self._consecutive_failures += 1
            stale = self._last_ok_mono is None or (now - self._last_ok_mono) >= self.stale_s
            threshold = self._consecutive_failures >= self.fail_threshold or stale
            if prev_state == ConnState.connected:
                self._set_state(ConnState.degraded, reason)
            if prev_state in (ConnState.connected, ConnState.degraded) and threshold:
                self._set_state(ConnState.disconnected, reason)
            elif prev_state == ConnState.recovering:
                if threshold:  # 復旧中に本当に切れた: 復旧タスクを取り消す
                    self._cancel_recovery()
                    self._set_state(ConnState.disconnected, reason)
                # 1 回だけの失敗なら復旧を続けさせる
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
        # 制御ループ停止の検出: 前回のサンプルから 0.5 秒以上経っているときだけ比較する
        # (poll_now で連続して読むと同じ 20 ms ティックに当たり、誤検出するため)
        now = time.monotonic()
        prev, prev_mono = self._prev_last_alive, self._prev_alive_mono
        if st.last_alive is not None:
            if prev is not None and prev_mono is not None and now - prev_mono >= 0.5:
                self._stalled = st.last_alive <= prev
                self._prev_last_alive, self._prev_alive_mono = st.last_alive, now
            elif prev is None:
                self._prev_last_alive, self._prev_alive_mono = st.last_alive, now
            if self._stalled:  # 次の比較まで判定を保持する(短い間隔のポーリングでも揺れない)
                return False, "制御ループが止まっています"
        # モーターの無効化は「不健康」ではなく復旧のきっかけとして _tick で扱う
        # (起動直後はモーターが無効なので、ここで弾くと復旧が永遠に始まらない)
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
        if self.on_change is not None:
            try:
                self.on_change(state.value, reason)
            except Exception:
                log.exception("on_change failed")

    # ------------------------------------------------------------ recovery
    def _backoff(self) -> float:
        return min(self.recovery_backoff_max_s, self.recovery_backoff_s * (2 ** max(0, self._recovery_failures - 1)))

    def _start_recovery(self) -> None:
        now = time.monotonic()
        if self._recovery_task is not None and not self._recovery_task.done():
            return
        if self._recovery_failures and now - self._last_recovery_attempt < self._backoff():
            return
        self._last_recovery_attempt = now
        self._set_state(ConnState.recovering, "復旧中…")
        self._recovery_task = asyncio.create_task(self._run_recovery(), name="connection-recovery")

    def _cancel_recovery(self) -> None:
        if self._recovery_task is not None and not self._recovery_task.done():
            self._recovery_task.cancel()

    async def _run_recovery(self) -> None:
        try:
            await self.recovery()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._recovery_failures += 1
            msg = str(e)
            log.warning("recovery failed (%d): %s", self._recovery_failures, msg)
            if msg != self._last_recovery_error:  # 同じ失敗を毎回トーストしない
                self._last_recovery_error = msg
                self.bus.toast("error", f"復旧に失敗しました: {msg}(自動で再試行します)")
            if self.state == ConnState.recovering:
                self._set_state(ConnState.disconnected, f"復旧失敗: {msg}")
            return
        self._recovery_failures = 0
        self._last_recovery_error = None
        if self.state == ConnState.recovering:
            self._set_state(ConnState.connected, "")
            self.bus.publish("recovery", ok=True)
