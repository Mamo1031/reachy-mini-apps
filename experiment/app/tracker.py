"""アプリ側の顔追跡: デーモンは顔の検出器としてだけ使い、頭(首)と腰の動きはこちらで作る。

- 顔位置(-1〜1)を角度に換算し、デッドゾーンと EMA で「見るべき向き」(goal)を決める。
- 送信ループは goal へ向けて毎フレーム少しずつ動かす(首は速く、腰はゆっくり)。首の相対角が
  body_follow_deg を超えたら腰が動き出し、首は自然に正面へ戻る。
- 顔を見失ったら lost_hold_s は保持し、そのあとゆっくりニュートラルへ戻る。
- ジェスチャー再生中は suspend(何も送らない)。再開は最後に送った姿勢から続けるのでスナップしない。
- ロボットとの通信エラーではループを止めない(次のフレームで再試行)。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

from .config import Settings
from .events import EventBus
from .pose import DEG, Pose
from .robot import FaceObs, RobotClient, RobotError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Gaze:
    """ロボットが向いている方向。head_yaw はワールド(ベース)座標系で腰の回転を含む。rad。"""

    body_yaw: float = 0.0
    head_yaw: float = 0.0
    pitch: float = 0.0

    @property
    def neck(self) -> float:
        """首の相対角(頭 − 腰)。"""
        return self.head_yaw - self.body_yaw

    def is_near(self, other: "Gaze", eps: float = 1e-6) -> bool:
        return abs(self.body_yaw - other.body_yaw) <= eps and abs(self.head_yaw - other.head_yaw) <= eps and abs(self.pitch - other.pitch) <= eps


NEUTRAL_GAZE = Gaze()


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def _slew(cur: float, goal: float, step: float) -> float:
    d = goal - cur
    if abs(d) <= step:
        return goal
    return cur + step if d > 0 else cur - step


class FaceTracker:
    def __init__(
        self,
        robot: RobotClient,
        settings_ref: Callable[[], Settings],
        bus: EventBus,
        *,
        allowed: Callable[[], bool] = lambda: True,
    ) -> None:
        self.robot = robot
        self.settings_ref = settings_ref
        self.bus = bus
        self.allowed = allowed
        self.state = "off"  # off | idle | tracking | lost | parked
        self.suspended = False
        self.parked = False
        self.gaze = NEUTRAL_GAZE  # 最後にロボットへ送った向き
        self._goal = NEUTRAL_GAZE  # 平滑化した目標(head_yaw / pitch)
        self._body_following = False
        self._task: asyncio.Task | None = None
        self._last_face_at: float | None = None
        self._last_ts: float | None = None
        self.face_detected = False
        self.frames = 0
        self.errors = 0
        self._error_logged_at = 0.0

    # ------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if not self.running:
            self._task = asyncio.create_task(self._run(), name="face-tracker")
            self.state = "idle"

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self.state = "off"
        self._body_following = False
        self._set_detected(False)

    def suspend(self) -> None:
        """ジェスチャーなど他の送り手が頭を使う間、何も送らない。"""
        self.suspended = True

    def resume(self, sync_to: Pose | None = None) -> None:
        """再開。sync_to を渡すと「いま実際に送られている姿勢」から続ける(スナップしない)。"""
        if sync_to is not None:
            self.sync_from(sync_to)
        self.suspended = False

    def park(self, parked: bool) -> None:
        """一時停止: 顔を追わず、ゆっくりニュートラルへ戻って止まる(検出は続く)。"""
        self.parked = parked

    def reset_gaze(self) -> None:
        """ロボットが外部(goto / スリープ)でニュートラルへ戻されたときに呼ぶ。"""
        self.gaze = NEUTRAL_GAZE
        self._goal = NEUTRAL_GAZE
        self._body_following = False

    def sync_from(self, pose: Pose) -> None:
        g = Gaze(body_yaw=pose.body_yaw, head_yaw=pose.yaw, pitch=pose.pitch)
        self.gaze = g
        self._goal = g

    def snapshot(self) -> Gaze:
        return self.gaze if self.running else NEUTRAL_GAZE

    def status(self) -> dict[str, Any]:
        g = self.gaze
        return {
            "state": self.state,
            "suspended": self.suspended,
            "face_detected": self.face_detected,
            "body_yaw_deg": round(g.body_yaw / DEG, 1),
            "head_yaw_deg": round(g.head_yaw / DEG, 1),
            "pitch_deg": round(g.pitch / DEG, 1),
            "frames": self.frames,
            "errors": self.errors,
        }

    # ------------------------------------------------------------ control law
    def ingest(self, obs: FaceObs, dt: float, now: float | None = None) -> None:
        """観測 1 回分を取り込んで goal を更新する。"""
        t = self.settings_ref().motion.tracking
        now = time.monotonic() if now is None else now
        fresh = bool(obs.detected) and obs.x is not None and obs.y is not None and obs.ts != self._last_ts
        if obs.ts is not None:
            self._last_ts = obs.ts
        if not fresh:
            if self._last_face_at is not None and now - self._last_face_at > t.lost_hold_s:
                self._set_detected(False)
            return
        self._last_face_at = now
        self._set_detected(True)
        # 画像の右(+x)にいる顔を見るには右(負の yaw)へ、下(+y)なら下(正の pitch)へ
        err_yaw = -float(obs.x) * t.gain_h_deg * DEG
        err_pitch = float(obs.y) * t.gain_v_deg * DEG
        dz = t.dead_zone_deg * DEG
        g = self.gaze
        goal = self._goal
        alpha = min(1.0, dt / max(float(t.smoothing_s), dt))
        if abs(err_yaw) < dz:
            head_yaw = g.head_yaw  # 十分向いている: いまの向きで止まる
        else:
            target = _clamp(g.head_yaw + err_yaw, (t.body_max_deg + t.head_max_deg) * DEG)
            head_yaw = goal.head_yaw + (target - goal.head_yaw) * alpha
        if abs(err_pitch) < dz:
            pitch = g.pitch
        else:
            target = max(-t.pitch_up_deg * DEG, min(t.pitch_down_deg * DEG, g.pitch + err_pitch))
            pitch = goal.pitch + (target - goal.pitch) * alpha
        self._goal = Gaze(body_yaw=goal.body_yaw, head_yaw=head_yaw, pitch=pitch)
        self.state = "tracking"

    def step(self, dt: float, now: float | None = None) -> Gaze | None:
        """送信 1 フレーム分だけ goal へ近づけた向きを返す。動く必要がなければ None。"""
        t = self.settings_ref().motion.tracking
        now = time.monotonic() if now is None else now
        lost = self._last_face_at is None or now - self._last_face_at > t.lost_hold_s
        returning = self.parked or lost
        if returning:
            goal = NEUTRAL_GAZE
            rate_head = rate_body = t.return_rate_deg_s * DEG
            if self.parked:
                self.state = "parked"
            elif self.state == "tracking":
                self.state = "lost"
        else:
            goal = self._goal
            rate_head, rate_body = t.head_rate_deg_s * DEG, t.body_rate_deg_s * DEG
        g = self.gaze
        head_yaw = _slew(g.head_yaw, goal.head_yaw, rate_head * dt)
        pitch = _slew(g.pitch, goal.pitch, rate_head * dt)
        if returning:
            body_goal = 0.0
            self._body_following = False
        else:
            neck_goal = goal.head_yaw - g.body_yaw
            if abs(neck_goal) > t.body_follow_deg * DEG:
                self._body_following = True
            elif abs(neck_goal) < 2.0 * DEG:
                self._body_following = False
            body_goal = _clamp(goal.head_yaw, t.body_max_deg * DEG) if self._body_following else g.body_yaw
        body_yaw = _slew(g.body_yaw, body_goal, rate_body * dt)
        head_yaw = body_yaw + _clamp(head_yaw - body_yaw, t.head_max_deg * DEG)
        new = Gaze(body_yaw=body_yaw, head_yaw=head_yaw, pitch=pitch)
        if new.is_near(g):
            if returning and new.is_near(NEUTRAL_GAZE):
                self.state = "parked" if self.parked else "idle"
            return None
        return new

    # ------------------------------------------------------------ loops
    async def _run(self) -> None:
        poll = asyncio.create_task(self._poll_loop(), name="face-tracker-poll")
        send = asyncio.create_task(self._send_loop(), name="face-tracker-send")
        try:
            await asyncio.gather(poll, send)
        finally:
            for task in (poll, send):
                task.cancel()
            for task in (poll, send):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _poll_loop(self) -> None:
        while True:
            t = self.settings_ref().motion.tracking
            period = 1.0 / max(2.0, float(t.poll_hz))
            await asyncio.sleep(period)
            if self.suspended or not self.allowed():
                continue
            try:
                obs = await self.robot.get_face(timeout=0.3)
            except RobotError as e:
                self._note_error(e)
                continue
            except Exception:  # ループは決して死なない
                log.exception("face poll failed")
                continue
            self.ingest(obs, period)

    async def _send_loop(self) -> None:
        while True:
            m = self.settings_ref().motion
            period = 1.0 / max(10.0, float(m.stream_hz))
            await asyncio.sleep(period)
            if self.suspended or not self.allowed():
                continue
            new = self.step(period)
            if new is None:
                continue
            try:
                await self.robot.stream_target(Pose(yaw=new.head_yaw, pitch=new.pitch, body_yaw=new.body_yaw), head=True, antennas=False)
            except RobotError as e:
                self._note_error(e)
                continue
            except Exception:
                log.exception("face tracker send failed")
                continue
            self.gaze = new
            self.frames += 1

    def _set_detected(self, value: bool) -> None:
        if value != self.face_detected:
            self.face_detected = value
            self.bus.publish("face", detected=value)

    def _note_error(self, e: Exception) -> None:
        self.errors += 1
        now = time.monotonic()
        if now - self._error_logged_at > 5.0:  # 同じ障害で毎フレーム書かない
            self._error_logged_at = now
            log.warning("face tracker: %s", e)
