"""軌道プレイヤー: 軌道を一定レートで `set_target` にストリーミングする。

- 開始時は現在姿勢から ramp_in 秒で軌道の先頭へ、終了時は ramp_out 秒でニュートラルへ。
- 時刻基準でフレームを選ぶので、通信が一瞬遅れてもフレームを飛ばして追いつく(間延びしない)。
- 頭が動く軌道の間は顔追跡を weight 0 にし(検出は一時停止)、終わったら元の重みへ戻す。
- キャンセルは協調的(毎ティックで確認)。`to_neutral=True` なら短いランプでニュートラルへ戻してから終わる。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from .config import Settings
from .events import EventBus
from .gestures import Trajectory
from .pose import NEUTRAL, Pose, lerp_pose
from .robot import RobotBusy, RobotClient, RobotError

log = logging.getLogger(__name__)


class TrajectoryPlayer:
    def __init__(self, robot: RobotClient, settings_ref: Callable[[], Settings], bus: EventBus) -> None:
        self.robot = robot
        self.settings_ref = settings_ref
        self.bus = bus
        self._lock = asyncio.Lock()
        self._cancel = asyncio.Event()
        self._cancel_to_neutral = False
        self._abort = False
        self.is_playing = False
        self.current: str | None = None
        self.last_sent: Pose = NEUTRAL
        self.tracking_paused = False
        # 直近の再生の実測(設定画面の「動作の送信状況」と実機検証用)
        self.stats: dict[str, float | int | str] = {"name": "", "frames": 0, "late": 0, "duration_s": 0.0, "hz": 0.0}

    # ------------------------------------------------------------ public
    def cancel(self, *, to_neutral: bool) -> None:
        """再生中なら中断を要求する(即座に返る)。"""
        if not self.is_playing:
            return
        self._cancel_to_neutral = to_neutral
        self._abort = not to_neutral
        self._cancel.set()

    async def wait_idle(self, timeout: float = 1.5) -> bool:
        deadline = time.monotonic() + timeout
        while self.is_playing and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        return not self.is_playing

    async def play(self, traj: Trajectory, *, pause_tracking: bool, tracking_weight: float = 1.0) -> None:
        """軌道を再生する。同時に 1 本だけ(ロックで直列化)。"""
        async with self._lock:
            self._cancel.clear()
            self._cancel_to_neutral = False
            self._abort = False
            self.is_playing = True
            self.current = traj.name
            paused = False
            try:
                start = await self._present_pose()
                # 現在姿勢をまず送っておく(追跡を止めた瞬間にスナップしないため)。アンテナだけの軌道なら頭は送らない
                await self._send(start, head=traj.moves_head)
                if pause_tracking and traj.moves_head:
                    await self.robot.set_tracking(True, 0.0)
                    paused = True
                    self.tracking_paused = True
                await self._stream(start, traj)
            except RobotError as e:
                log.warning("gesture aborted: %s", e)
                self.bus.toast("error", f"動作を中断しました: {e}")
            finally:
                if paused:
                    self.tracking_paused = False
                    try:
                        await self.robot.set_tracking(True, tracking_weight)
                    except RobotError as e:
                        log.warning("tracking restore failed: %s", e)
                self.is_playing = False
                self.current = None

    # ------------------------------------------------------------ internals
    async def _present_pose(self) -> Pose:
        try:
            return await self.robot.present_pose()
        except RobotError as e:
            log.warning("present_pose failed, using last sent: %s", e)
            return self.last_sent

    async def _send(self, pose: Pose, *, head: bool) -> None:
        try:
            await self.robot.set_target(pose, head=head)
        except RobotBusy:
            # 実行中のムーブ(起動時の goto 等)が残っていると無視されるので止めて 1 回だけ再送
            n = await self.robot.clear_moves()
            log.info("cleared %d running move(s) before streaming", n)
            await self.robot.set_target(pose, head=head)
        self.last_sent = pose if head else self.last_sent.with_(ant_r=pose.ant_r, ant_l=pose.ant_l)

    async def _stream(self, start: Pose, traj: Trajectory) -> None:
        m = self.settings_ref().motion
        period = 1.0 / max(10.0, float(m.stream_hz))
        ramp_in = max(0.0, m.ramp_in_s)
        ramp_out = max(0.05, m.ramp_out_s)
        head = traj.moves_head
        t1 = ramp_in
        t2 = t1 + traj.duration
        t3 = t2 + ramp_out
        t0 = time.monotonic()
        next_tick = t0
        frames = late = 0
        try:
            while True:
                if self._cancel.is_set():
                    if self._cancel_to_neutral:
                        await self._ramp(self.last_sent, NEUTRAL, ramp_out, period, head=head)
                    return
                t = time.monotonic() - t0
                if t < t1 and ramp_in > 0:
                    pose = lerp_pose(start, traj.first, t / t1)
                elif t < t2:
                    pose = traj.pose_at(t - t1)
                elif t < t3:
                    pose = lerp_pose(traj.last, NEUTRAL, (t - t2) / ramp_out)
                else:
                    await self._send(NEUTRAL, head=head)
                    return
                await self._send(pose, head=head)
                frames += 1
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                else:  # 遅れたら追いつく(次フレームは時刻基準で選ばれる)
                    late += 1
                    next_tick = time.monotonic()
        finally:
            elapsed = time.monotonic() - t0
            self.stats = {"name": traj.name, "frames": frames, "late": late, "duration_s": round(elapsed, 2), "hz": round(frames / elapsed, 1) if elapsed > 0 else 0.0}

    async def _ramp(self, a: Pose, b: Pose, duration: float, period: float, *, head: bool) -> None:
        t0 = time.monotonic()
        while True:
            if self._abort:
                return
            s = (time.monotonic() - t0) / duration
            if s >= 1.0:
                await self._send(b, head=head)
                return
            await self._send(lerp_pose(a, b, s), head=head)
            await asyncio.sleep(period)
