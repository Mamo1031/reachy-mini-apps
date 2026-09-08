"""Probe tests for suspected protocol-semantics defects (not part of the suite)."""
import asyncio
import time

import httpx
import pytest

from app import main
from tests.fake_daemon import FakeDaemon, make_app
from tests.test_audio import FakeTTS
from tests.test_performer import Harness


@pytest.fixture
async def h(tmp_path):
    hh = Harness(tmp_path)
    await hh.start()
    yield hh
    await hh.stop()


def played(h):
    return [n for _, n in h.fake.played]


async def test_probe_interrupt_during_audio_lead_orphans_gesture(h):
    """Press B2 (long gesture), then BC2 within audio_lead_ms. Expect: B2 gesture never plays."""
    h.settings.motion.audio_lead_ms = 300
    await h.performer.play_phrase("B2")
    await asyncio.sleep(0.05)  # inside the 300 ms lead: player.is_playing is False
    assert not h.player.is_playing
    await h.performer.play_phrase("BC2")
    await asyncio.sleep(0.5)
    print("player.current after interrupt:", h.player.current, "played:", played(h))
    # BC2 gesture is shake_small; B2 is calm_antennas (serenity1). Which one is streaming?
    assert h.player.current != "calm_antennas", "orphaned B2 gesture task is playing after being interrupted"


async def test_probe_stop_during_audio_lead_gesture_plays_after_stop(h):
    h.settings.motion.audio_lead_ms = 300
    await h.performer.play_phrase("B2")
    await asyncio.sleep(0.05)
    await h.performer.stop()
    assert h.performer.status == "idle"
    n = len(h.fake.targets)
    await asyncio.sleep(0.6)
    print("targets sent after STOP:", len(h.fake.targets) - n, "player.current:", h.player.current)
    assert len(h.fake.targets) == n, "gesture streamed after STOP"


async def test_probe_apply_tracking_mid_gesture_overrides_player_pause(h):
    h.settings.motion.tracking_enabled = True
    h.settings.motion.audio_lead_ms = 0
    await h.performer.play_phrase("B3")  # head_lower, moves head
    await asyncio.sleep(0.3)
    assert h.player.tracking_paused and h.fake.tracking_weight == 0.0
    # experimenter toggles tracking (or settings PUT / recover) mid-gesture
    await h.performer.set_tracking(True)
    print("weight mid-gesture after apply_tracking:", h.fake.tracking_weight, "player paused:", h.player.tracking_paused)
    assert h.fake.tracking_weight == 0.0, "tracking weight restored to 1.0 while head gesture still streaming -> head part silently ignored"


@pytest.fixture
async def env(tmp_path):
    fake = FakeDaemon()
    tts = FakeTTS()
    await main.build_state(data_dir=tmp_path, robot_transport=httpx.ASGITransport(app=make_app(fake)), tts=tts, monitor_interval=0.05)
    main.state.settings.motion.audio_lead_ms = 30
    main.state.settings.motion.ramp_in_s = 0.1
    main.state.motion_ramp = 0.1
    main.state.settings.motion.ramp_out_s = 0.1
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
        yield c, fake, tts
    await main.shutdown_state()


async def wait_preflight(c, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pf = (await c.get("/api/state")).json()["preflight"]
        if pf["synth"]["status"] in ("ok", "fail") and pf["volume"]["status"] in ("ok", "fail"):
            return pf
        await asyncio.sleep(0.1)
    raise AssertionError("preflight")


async def test_probe_idle_runs_while_resting_and_kills_wake_goto(env):
    c, fake, tts = env
    await wait_preflight(c)
    main.state.settings.motion.idle.enabled = True
    main.state.settings.motion.idle.interval_s = 2.0
    main.state.settings.motion.idle.jitter_s = 0.0
    r = await c.post("/api/robot/rest")
    assert r.json()["resting"] is True and fake.motor_mode == "disabled"
    n = len(fake.targets)
    await asyncio.sleep(3.5)
    print("set_target calls while resting (motors disabled):", len(fake.targets) - n)
    idle_while_resting = len(fake.targets) - n
    # Now wake; the fake goto lasts 2.0 s. Make idle due right away.
    main.state.performer.last_idle_at = time.monotonic() - 100
    stops_before = fake.calls.count("POST /api/move/stop")
    r = await c.post("/api/robot/wake")
    stops_after = fake.calls.count("POST /api/move/stop")
    print("move/stop calls during wake:", stops_after - stops_before, "head after wake:", fake.head, "resting:", r.json())
    assert idle_while_resting == 0, "idle loop streams set_target while robot is resting with motors disabled"
