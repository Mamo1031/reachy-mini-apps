import asyncio
import time

import pytest

from app.config import Settings
from app.events import EventBus
from app.gestures import Trajectory
from app.player import TrajectoryPlayer
from app.pose import DEG, NEUTRAL, Pose
from app.robot import RobotBusy, RobotUnreachable


class FakeRobot:
    """set_target の呼び出しを記録する軽量フェイク(遅延・RobotBusy を注入できる)。"""

    def __init__(self):
        self.targets: list[tuple[float, Pose, bool]] = []
        self.tracking: list[tuple[bool, float]] = []
        self.latency = 0.0
        self.busy_count = 0  # 最初の N 回の set_target で RobotBusy
        self.cleared = 0
        self.present = NEUTRAL.with_(pitch=0.2, ant_r=-1.0, ant_l=1.0)
        self.fail_present = False
        self.unreachable_after = None  # 呼び出し回数 N 以降は RobotUnreachable

    async def present_pose(self, timeout=None):
        if self.fail_present:
            raise RobotUnreachable("x")
        return self.present

    async def set_target(self, pose, *, head=True, antennas=True):
        if self.unreachable_after is not None and len(self.targets) >= self.unreachable_after:
            raise RobotUnreachable("link down")
        if self.busy_count > 0:
            self.busy_count -= 1
            raise RobotBusy("busy")
        if self.latency:
            await asyncio.sleep(self.latency)
        self.targets.append((time.monotonic(), pose, head))

    async def stream_target(self, pose, *, head=True, antennas=True):
        await self.set_target(pose, head=head, antennas=antennas)
        return "http"

    async def clear_moves(self):
        self.cleared += 1
        return 1

    async def set_tracking(self, enabled, weight=1.0):
        self.tracking.append((enabled, weight))


def nod_traj(duration=0.5, hz=50) -> Trajectory:
    n = int(duration * hz)
    frames = [(i / hz, NEUTRAL.with_(pitch=0.3 * (i / n))) for i in range(n + 1)]
    return Trajectory("nod", frames, True)


def antenna_traj(duration=0.3, hz=50) -> Trajectory:
    n = int(duration * hz)
    frames = [(i / hz, NEUTRAL.with_(ant_l=0.5 * (i / n))) for i in range(n + 1)]
    return Trajectory("ant", frames, False)


@pytest.fixture
def setup():
    settings = Settings()
    settings.motion.ramp_in_s = 0.2
    settings.motion.ramp_out_s = 0.2
    settings.motion.stream_hz = 50
    robot = FakeRobot()
    player = TrajectoryPlayer(robot, lambda: settings, EventBus())  # type: ignore[arg-type]
    return settings, robot, player


async def restore_08(robot):
    await robot.set_tracking(True, 0.8)


async def test_play_ramps_in_streams_and_ramps_out_to_neutral(setup):
    settings, robot, player = setup
    t0 = time.monotonic()
    await player.play(nod_traj(0.5), pause_tracking=True, restore_tracking=lambda: restore_08(robot))
    elapsed = time.monotonic() - t0
    assert 0.85 < elapsed < 1.4
    poses = [p for _, p, _ in robot.targets]
    assert poses[0] == robot.present  # 最初に現在姿勢を送る(スナップ防止)
    assert poses[-1] == NEUTRAL
    assert max(p.pitch for p in poses) == pytest.approx(0.3, abs=0.02)
    # 50 Hz で約 0.9 秒 → 40 フレーム以上
    assert len(poses) >= 40
    # 追跡: weight 0 → 復元コールバック(0.8)の順
    assert robot.tracking == [(True, 0.0), (True, 0.8)]
    assert not player.is_playing and player.tracking_paused is False
    assert player.stats["frames"] >= 40 and player.stats["name"] == "nod"


async def test_antenna_only_does_not_pause_tracking_or_send_head(setup):
    _, robot, player = setup
    await player.play(antenna_traj(), pause_tracking=True)
    assert robot.tracking == []
    assert all(head is False for _, _, head in robot.targets[1:])


async def test_late_frames_are_skipped_not_delayed(setup):
    _, robot, player = setup
    robot.latency = 0.1  # 1 回の送信に 100 ms かかる
    t0 = time.monotonic()
    await player.play(nod_traj(0.6), pause_tracking=False)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.6  # 0.2 + 0.6 + 0.2 = 1.0 秒の軌道が大きく間延びしない
    assert len(robot.targets) < 20
    assert robot.targets[-1][1] == NEUTRAL


async def test_cancel_to_neutral_ends_at_neutral_quickly(setup):
    _, robot, player = setup

    async def restore():
        await robot.set_tracking(True, 1.0)

    task = asyncio.create_task(player.play(nod_traj(3.0), pause_tracking=True, restore_tracking=restore))
    await asyncio.sleep(0.4)
    t0 = time.monotonic()
    player.cancel(to_neutral=True)
    await task
    assert time.monotonic() - t0 < 0.5
    assert robot.targets[-1][1] == NEUTRAL
    assert robot.tracking[-1] == (True, 1.0)


async def test_cancel_immediate_returns_without_neutral(setup):
    _, robot, player = setup
    task = asyncio.create_task(player.play(nod_traj(3.0), pause_tracking=False))
    await asyncio.sleep(0.4)
    n = len(robot.targets)
    player.cancel(to_neutral=False)
    await task
    assert len(robot.targets) <= n + 1
    assert robot.targets[-1][1] != NEUTRAL


async def test_busy_triggers_clear_moves_and_retry(setup):
    _, robot, player = setup
    robot.busy_count = 1
    await player.play(nod_traj(0.2), pause_tracking=False)
    assert robot.cleared == 1
    assert robot.targets[-1][1] == NEUTRAL


async def test_present_pose_failure_falls_back_to_last_sent(setup):
    _, robot, player = setup
    robot.fail_present = True
    await player.play(nod_traj(0.2), pause_tracking=False)
    assert robot.targets[0][1] == NEUTRAL  # last_sent の初期値


async def test_unreachable_mid_gesture_restores_tracking_and_ends(setup):
    _, robot, player = setup
    robot.unreachable_after = 5

    async def restore():
        await robot.set_tracking(True, 1.0)

    await player.play(nod_traj(1.0), pause_tracking=True, restore_tracking=restore)
    assert not player.is_playing
    assert robot.tracking[-1] == (True, 1.0)


async def test_task_cancel_during_play_still_restores_tracking(setup):
    """音声先行の待ちの後、再生途中で Task.cancel されても finally で追跡を戻す。"""
    _, robot, player = setup

    async def restore():
        await robot.set_tracking(True, 1.0)

    task = asyncio.create_task(player.play(nod_traj(3.0), pause_tracking=True, restore_tracking=restore))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not player.is_playing and player.tracking_paused is False
    assert robot.tracking[-1] == (True, 1.0)


async def test_wait_idle(setup):
    _, robot, player = setup
    task = asyncio.create_task(player.play(nod_traj(0.3), pause_tracking=False))
    await asyncio.sleep(0.05)
    assert player.is_playing
    assert await player.wait_idle(timeout=2.0)
    await task
