import asyncio

import pytest

from app.config import Settings
from app.events import EventBus
from app.pose import DEG, Pose
from app.robot import FaceObs, RobotUnreachable
from app.tracker import NEUTRAL_GAZE, FaceTracker, Gaze


class StubRobot:
    """顔位置を返し、送られた姿勢を記録するだけの軽量フェイク。"""

    def __init__(self):
        self.face = FaceObs(False, None, None, None)
        self.sent: list[Pose] = []
        self.fail = False
        self.ts = 0.0

    async def get_face(self, timeout=None):
        if self.fail:
            raise RobotUnreachable("x")
        if self.face.detected:
            self.ts += 0.1
            return FaceObs(True, self.face.x, self.face.y, self.ts)
        return FaceObs(False, None, None, None)

    async def stream_target(self, pose, *, head=True, antennas=True):
        if self.fail:
            raise RobotUnreachable("x")
        assert head and not antennas  # トラッカーは頭と腰だけ送る
        self.sent.append(pose)
        return "http"


def make(allowed=lambda: True):
    settings = Settings()
    settings.motion.tracking.smoothing_s = 0.05  # 同期テストでは平滑化なし(dt 0.1 → alpha 1)
    robot = StubRobot()
    tracker = FaceTracker(robot, lambda: settings, EventBus(), allowed=allowed)  # type: ignore[arg-type]
    return tracker, settings, robot


def test_geometry_dead_zone_and_limits():
    tr, s, _ = make()
    t = s.motion.tracking
    tr.ingest(FaceObs(True, 0.5, 0.0, 1.0), 0.1, now=0.0)
    assert tr._goal.head_yaw == pytest.approx(-0.5 * t.gain_h_deg * DEG)  # 画像の右 → 右(負の yaw)
    assert tr._goal.pitch == 0.0
    assert tr.face_detected and tr.state == "tracking"

    tr, s, _ = make()
    tr.ingest(FaceObs(True, 0.0, 0.5, 1.0), 0.1, now=0.0)
    assert tr._goal.pitch == pytest.approx(0.5 * s.motion.tracking.gain_v_deg * DEG)  # 画像の下 → 下(正の pitch)

    tr, _, _ = make()
    tr.ingest(FaceObs(True, 0.05, -0.05, 1.0), 0.1, now=0.0)  # 2.2° は不感帯(3°)の内側
    assert tr._goal == NEUTRAL_GAZE

    tr, s, _ = make()
    tr.ingest(FaceObs(True, -1.0, -1.0, 1.0), 0.1, now=0.0)
    assert tr._goal.head_yaw == pytest.approx(44 * DEG)  # 60°(腰 35 + 首 25)の内側
    assert tr._goal.pitch == pytest.approx(-s.motion.tracking.pitch_up_deg * DEG)  # 上向きは上限で止まる
    tr.ingest(FaceObs(True, 1.0, 1.0, 1.0), 0.1, now=0.05)  # 同じ ts の観測は無視
    assert tr._goal.head_yaw == pytest.approx(44 * DEG)


def test_rate_limits_and_body_follow():
    tr, s, _ = make()
    t = s.motion.tracking
    tr.ingest(FaceObs(True, -1.0, 0.0, 1.0), 0.1, now=0.0)  # 44° 左
    dt, now = 0.02, 0.0
    for _ in range(600):
        now += dt
        tr._last_face_at = now  # 顔は見え続けている
        new = tr.step(dt, now=now)
        if new is None:
            break
        assert abs(new.head_yaw - tr.gaze.head_yaw) <= t.head_rate_deg_s * DEG * dt + 1e-9
        assert abs(new.body_yaw - tr.gaze.body_yaw) <= t.body_rate_deg_s * DEG * dt + 1e-9
        assert abs(new.neck) <= t.head_max_deg * DEG + 1e-9
        tr.gaze = new
    assert tr.gaze.head_yaw == pytest.approx(44 * DEG, abs=1e-6)
    assert tr.gaze.body_yaw == pytest.approx(t.body_max_deg * DEG, abs=1e-6)  # 腰が上限まで担う
    assert now == pytest.approx(t.body_max_deg / t.body_rate_deg_s, abs=0.1)  # 腰の速さで決まる

    # 小さなずれ(6.6°)は首だけで済ませ、腰は動かない
    tr, s, _ = make()
    tr.ingest(FaceObs(True, -0.15, 0.0, 1.0), 0.1, now=0.0)
    now = 0.0
    for _ in range(200):
        now += dt
        tr._last_face_at = now
        new = tr.step(dt, now=now)
        if new is None:
            break
        tr.gaze = new
    assert tr.gaze.body_yaw == 0.0 and tr.gaze.head_yaw == pytest.approx(0.15 * 44 * DEG, abs=1e-6)


def test_lost_holds_then_returns_to_neutral():
    tr, s, _ = make()
    t = s.motion.tracking
    tr.sync_from(Pose(yaw=30 * DEG, pitch=5 * DEG, body_yaw=20 * DEG))
    tr._last_face_at = 0.0
    tr.state = "tracking"
    assert tr.step(0.02, now=1.0) is None and tr.state == "tracking"  # 保持中は動かない
    new = tr.step(0.02, now=2.0)
    assert new is not None and tr.state == "lost"
    assert 0 < 30 * DEG - new.head_yaw <= t.return_rate_deg_s * DEG * 0.02 + 1e-9
    now = 2.0
    for _ in range(3000):
        now += 0.02
        new = tr.step(0.02, now=now)
        if new is None:
            break
        tr.gaze = new
    assert tr.gaze.is_near(NEUTRAL_GAZE) and tr.state == "idle"
    tr.ingest(FaceObs(False, None, None, 9.0), 0.1, now=now)
    assert not tr.face_detected


def test_park_returns_to_neutral_and_ignores_faces():
    tr, _, _ = make()
    tr.sync_from(Pose(yaw=20 * DEG, body_yaw=10 * DEG))
    tr._last_face_at = 0.0
    tr.park(True)
    tr.ingest(FaceObs(True, -1.0, 0.0, 1.0), 0.1, now=0.1)
    now = 0.1
    for _ in range(2000):
        now += 0.02
        new = tr.step(0.02, now=now)
        if new is None:
            break
        tr.gaze = new
    assert tr.gaze.is_near(NEUTRAL_GAZE) and tr.state == "parked"


async def test_loops_send_suspend_resume_and_survive_errors():
    tr, s, robot = make()
    s.motion.stream_hz = 50
    s.motion.tracking.poll_hz = 25
    robot.face = FaceObs(True, -0.8, 0.0, 0.0)
    tr.start()
    await asyncio.sleep(0.4)
    assert tr.running and tr.state == "tracking" and tr.frames > 5 and tr.face_detected
    assert all(p.yaw > 0 for p in robot.sent[-3:])  # 左の顔 → 正の yaw
    n = len(robot.sent)
    tr.suspend()
    await asyncio.sleep(0.2)
    assert len(robot.sent) == n  # 停止中は何も送らない
    tr.resume(sync_to=Pose(yaw=0.1, body_yaw=0.05))
    assert tr.gaze == Gaze(body_yaw=0.05, head_yaw=0.1, pitch=0.0)
    robot.fail = True
    await asyncio.sleep(0.2)
    assert tr.running and tr.errors > 0  # 通信エラーでループは死なない
    robot.fail = False
    await asyncio.sleep(0.2)
    assert len(robot.sent) > n
    await tr.stop()
    assert not tr.running and tr.state == "off" and tr.snapshot() == NEUTRAL_GAZE and not tr.face_detected


async def test_allowed_gate_blocks_polling_and_sending():
    tr, s, robot = make(allowed=lambda: False)
    s.motion.stream_hz = 50
    robot.face = FaceObs(True, -0.8, 0.0, 0.0)
    tr.start()
    await asyncio.sleep(0.3)
    assert tr.frames == 0 and robot.sent == [] and not tr.face_detected
    await tr.stop()
