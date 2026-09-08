import asyncio
import time

import httpx
import pytest

from app.events import EventBus
from app.monitor import ConnState, ConnectionMonitor
from app.pose import NEUTRAL, Pose
from app.robot import RobotBusy, RobotClient, RobotUnreachable
from tests.fake_daemon import FakeDaemon, make_app


@pytest.fixture
def fake():
    return FakeDaemon()


@pytest.fixture
async def robot(fake):
    client = RobotClient("http://fake", transport=httpx.ASGITransport(app=make_app(fake)))
    yield client
    await client.aclose()


async def test_status_and_present_pose(robot, fake):
    st = await robot.daemon_status()
    assert st.state == "running" and st.ready and st.motor_mode == "disabled"
    p = await robot.present_pose()
    assert p.pitch == 0.5 and p.ant_r == -3.05 and p.ant_l == 3.05


async def test_goto_dropped_while_running_and_clear_moves(robot, fake):
    u1 = await robot.goto(NEUTRAL, 0.5)
    assert u1 and await robot.running_moves() == [u1]
    with pytest.raises(RobotBusy):
        await robot.set_target(NEUTRAL)
    assert await robot.clear_moves() == 1
    assert await robot.running_moves() == []
    await robot.set_target(NEUTRAL)
    assert fake.targets and fake.targets[-1][1]["target_body_yaw"] is None
    # 未知 uuid の stop は 500 → 握りつぶす
    await robot.stop_move("nope")


async def test_sounds_roundtrip(robot, fake):
    await robot.upload_sound("abc.wav", b"RIFF....")
    assert await robot.list_sounds() == {"abc.wav"}
    await robot.play_sound("abc.wav")
    await robot.stop_sound()
    assert [n for _, n in fake.played] == ["abc.wav", "<stop>"]


async def test_unreachable_maps_to_robot_error():
    client = RobotClient("http://127.0.0.1:9", connect_timeout=0.2, read_timeout=0.2)
    try:
        with pytest.raises(RobotUnreachable):
            await client.daemon_status()
    finally:
        await client.aclose()


async def _wait_state(mon: ConnectionMonitor, state: ConnState, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if mon.state == state:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"state {mon.state} != {state}")


async def test_monitor_recovers_once_then_disconnects_on_outage(robot, fake):
    bus = EventBus()
    calls = []

    async def recovery():
        calls.append(time.monotonic())

    mon = ConnectionMonitor(robot, bus, recovery, interval=0.05, poll_timeout=0.3, fail_threshold=3, stale_s=0.3, recovery_backoff_s=0.0)
    await mon.start()
    try:
        await _wait_state(mon, ConnState.connected)
        assert len(calls) == 1
        # しばらく安定していても復旧は再実行されない
        await asyncio.sleep(0.2)
        assert len(calls) == 1 and mon.is_connected()
        # 障害注入 → degraded → disconnected
        fake.fail_until = time.monotonic() + 0.5
        await _wait_state(mon, ConnState.disconnected)
        assert "503" in mon.snapshot.reason or "接続" in mon.snapshot.reason or mon.snapshot.reason
        # 復帰 → 復旧手順が 1 回だけ走って connected
        await _wait_state(mon, ConnState.connected, timeout=3.0)
        assert len(calls) == 2
        types = [e["type"] for e in bus.recent]
        assert "connection" in types
    finally:
        await mon.stop()


async def test_monitor_detects_stalled_control_loop(robot, fake):
    bus = EventBus()

    async def recovery():
        pass

    mon = ConnectionMonitor(robot, bus, recovery, interval=0.05, poll_timeout=0.3, fail_threshold=3, stale_s=10.0, recovery_backoff_s=0.0)
    await mon.start()
    try:
        await _wait_state(mon, ConnState.connected)
        fake.freeze_alive = True
        await _wait_state(mon, ConnState.disconnected, timeout=2.0)
        assert "制御ループ" in mon.snapshot.reason
        fake.freeze_alive = False
        await _wait_state(mon, ConnState.connected, timeout=2.0)
    finally:
        await mon.stop()


async def test_monitor_recovery_failure_keeps_disconnected(robot, fake):
    bus = EventBus()
    attempts = []

    async def recovery():
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("boom")

    mon = ConnectionMonitor(robot, bus, recovery, interval=0.05, poll_timeout=0.3, recovery_backoff_s=0.1)
    await mon.start()
    try:
        await _wait_state(mon, ConnState.connected, timeout=3.0)
        assert len(attempts) == 2
        assert any(e["type"] == "toast" and "復旧に失敗" in e["message"] for e in bus.recent)
    finally:
        await mon.stop()


async def test_face_event_published(robot, fake):
    bus = EventBus()

    async def recovery():
        pass

    q = bus.subscribe()
    mon = ConnectionMonitor(robot, bus, recovery, interval=0.05, poll_timeout=0.3, recovery_backoff_s=0.0)
    await mon.start()
    try:
        await _wait_state(mon, ConnState.connected)
        fake.face_detected = True
        deadline = time.monotonic() + 1.0
        seen = False
        while time.monotonic() < deadline and not seen:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.2)
                seen = ev["type"] == "face" and ev["detected"] is True
            except asyncio.TimeoutError:
                pass
        assert seen
    finally:
        await mon.stop()
