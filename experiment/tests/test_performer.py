import asyncio
import time

import httpx
import pytest

from app.audio import AudioStore
from app.config import DATA_DIR, Settings, default_gestures, default_phrases
from app.events import EventBus
from app.gestures import GestureLibrary
from app.monitor import ConnectionMonitor, ConnState
from app.performer import PerformError, Performer
from app.player import TrajectoryPlayer
from app.robot import RobotClient
from app.session import SessionManager
from tests.fake_daemon import FakeDaemon, make_app
from tests.test_audio import FakeTTS


class Harness:
    def __init__(self, tmp_path):
        self.fake = FakeDaemon()
        self.fake.motor_mode = "enabled"
        self.settings = Settings()
        self.settings.motion.ramp_in_s = 0.1
        self.settings.motion.ramp_out_s = 0.1
        self.settings.motion.audio_lead_ms = 50
        self.settings.motion.idle.enabled = False
        self.phrases = default_phrases()
        self.gestures = default_gestures()
        self.bus = EventBus()
        self.robot = RobotClient("http://fake", transport=httpx.ASGITransport(app=make_app(self.fake)))
        self.tts = FakeTTS()
        self.audio = AudioStore(tmp_path / "cache", self.robot, self.tts, lambda: self.settings)
        self.lib = GestureLibrary(lambda: self.gestures, lambda: self.settings, DATA_DIR / "moves")
        self.player = TrajectoryPlayer(self.robot, lambda: self.settings, self.bus)
        self.session = SessionManager(tmp_path / "logs", self.bus, lambda: self.settings, lambda: self.phrases)
        self.recoveries = 0

        async def recovery():
            self.recoveries += 1

        self.monitor = ConnectionMonitor(self.robot, self.bus, recovery, interval=0.05, poll_timeout=0.3, recovery_backoff_s=0.0)
        self.performer = Performer(self.robot, self.player, self.audio, lambda: self.phrases, self.lib, self.session, self.monitor, self.bus, lambda: self.settings)

    async def start(self):
        await self.monitor.start()
        deadline = time.monotonic() + 3
        while not self.monitor.is_connected() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        assert self.monitor.is_connected()
        self.session.start("はな", "ちゃん", "robot_first", "empathy")

    async def stop(self):
        await self.performer.stop_idle()
        await self.monitor.stop()
        await self.robot.aclose()


@pytest.fixture
async def h(tmp_path):
    hh = Harness(tmp_path)
    await hh.start()
    yield hh
    await hh.stop()


def played(h):
    return [n for _, n in h.fake.played]


async def test_play_phrase_plays_sound_then_gesture_and_finishes(h):
    res = await h.performer.play_phrase("BC1")
    assert res["accepted"] and res["duration"] > 0
    assert h.performer.status == "playing" and h.performer.current["id"] == "BC1"
    assert h.performer.current["text"] == "うん"
    assert len(played(h)) == 1 and played(h)[0].endswith(".wav")
    # 音 → lead → 軌道の順(最初の set_target は音より後)
    await asyncio.sleep(0.15)
    assert h.fake.targets and h.fake.targets[0][0] >= h.fake.played[0][0] + 0.04
    await asyncio.sleep(res["duration"] + 0.6)
    assert h.performer.status == "idle" and h.performer.current is None
    kinds = [e["kind"] for e in h.bus.recent if e["type"] == "log"]
    assert kinds[-3:] == ["button", "playing", "done"]


async def test_name_expansion_and_intro_done(h):
    await h.performer.play_phrase("A2")
    assert h.performer.current["text"].startswith("はなちゃんっていうんだね")
    assert h.session.snapshot()["intro_done"] == ["A2"]
    # 励ましだけが「前回発話」を更新する
    assert h.session.snapshot()["since_last_utterance_s"] is None
    await asyncio.sleep(0.6)
    await h.performer.play_phrase("B1")
    assert h.session.snapshot()["since_last_utterance_s"] is not None


async def test_debounce_same_button(h):
    r1 = await h.performer.play_phrase("BC3")
    r2 = await h.performer.play_phrase("BC3")
    assert r1["accepted"] and r2 == {"accepted": False, "reason": "debounce"}


async def test_new_button_interrupts_current(h):
    await h.performer.play_phrase("B2")  # 長め(serenity1 4.6 s)
    await asyncio.sleep(0.3)
    assert h.player.is_playing
    await h.performer.play_phrase("BC2")
    assert h.performer.current["id"] == "BC2"
    names = played(h)
    assert names[1] == "<stop>" and len(names) == 3
    kinds = [e["kind"] for e in h.bus.recent if e["type"] == "log"]
    assert "interrupted" in kinds


async def test_stop_returns_to_neutral_and_restores_tracking(h):
    h.settings.motion.tracking_enabled = True
    await h.performer.play_phrase("B3")
    await asyncio.sleep(0.4)
    await h.performer.stop()
    assert h.performer.status == "idle" and h.performer.current is None
    assert not h.player.is_playing
    last = h.fake.targets[-1][1]
    assert abs(last["target_head_pose"]["pitch"]) < 1e-6
    assert h.fake.tracking_enabled and h.fake.tracking_weight == 1.0
    assert played(h)[-1] == "<stop>"


async def test_pause_blocks_tracking_and_resume_restores(h):
    h.settings.motion.tracking_enabled = True
    await h.performer.pause()
    assert h.performer.paused and h.fake.tracking_weight == 0.0
    # 一時停止中でもフレーズは再生できるが、追跡は止まったまま
    await h.performer.play_phrase("BC1")
    await asyncio.sleep(0.5)
    assert h.fake.tracking_weight == 0.0
    await h.performer.resume()
    assert not h.performer.paused and h.fake.tracking_weight == 1.0


async def test_tracking_toggle_applies_and_persists_in_settings(h):
    await h.performer.set_tracking(False)
    assert h.fake.tracking_enabled is False and h.settings.motion.tracking_enabled is False
    await h.performer.set_tracking(True)
    assert h.fake.tracking_enabled is True


async def test_play_refused_when_disconnected(h):
    h.fake.fail_until = time.monotonic() + 1.0
    deadline = time.monotonic() + 2
    while h.monitor.state != ConnState.disconnected and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    with pytest.raises(PerformError) as ei:
        await h.performer.play_phrase("BC1")
    assert ei.value.status == 503


async def test_unknown_phrase(h):
    with pytest.raises(PerformError) as ei:
        await h.performer.play_phrase("ZZ")
    assert ei.value.status == 404


async def test_idle_gesture_runs_when_idle_and_yields(h):
    h.settings.motion.idle.enabled = True
    h.settings.motion.idle.interval_s = 1.2
    h.settings.motion.idle.jitter_s = 0.0
    h.settings.motion.tracking_enabled = True
    h.performer.start_idle()
    await asyncio.sleep(2.6)
    heads = [t[1]["target_head_pose"] for t in h.fake.targets]
    assert h.fake.targets and all(hp is None for hp in heads)  # 追跡 ON → アンテナだけ
    await h.performer.stop_idle()


async def test_prewarm_texts(h):
    texts = h.performer.texts_for_prewarm(include_child=False)
    assert all("{" not in t for t in texts)
    assert not any("はなちゃん" in t for t in texts)
    all_texts = h.performer.texts_for_prewarm(include_child=True)
    assert any("はなちゃんっていうんだね" in t for t in all_texts)
    assert any("ドラちゃん" in t for t in texts) and any("はるかお姉さん" in t for t in texts)
