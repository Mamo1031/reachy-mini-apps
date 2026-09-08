import io
import wave

import httpx
import pytest

from app.audio import AudioStore, wav_duration
from app.config import Settings, VoiceParams
from app.robot import RobotClient
from app.tts.base import TTSBackend, Voice
from tests.fake_daemon import FakeDaemon, make_app


def make_wav(seconds: float, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))
    return buf.getvalue()


class FakeTTS(TTSBackend):
    name = "fake"

    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    async def ensure_ready(self, progress=None):
        return None

    async def is_ready(self):
        return True

    async def list_voices(self):
        return [Voice("1", "fake voice", "credit")]

    async def synthesize(self, text, voice_id, params):
        self.calls.append((text, voice_id))
        return make_wav(0.5 + 0.1 * len(text))


@pytest.fixture
def fake():
    return FakeDaemon()


@pytest.fixture
async def store(fake, tmp_path):
    robot = RobotClient("http://fake", transport=httpx.ASGITransport(app=make_app(fake)))
    settings = Settings()
    tts = FakeTTS()
    s = AudioStore(tmp_path / "cache", robot, tts, lambda: settings)
    yield s, tts, settings, robot
    await robot.aclose()


def test_wav_duration():
    assert wav_duration(make_wav(1.25)) == pytest.approx(1.25)


async def test_key_depends_on_voice_params_and_text(store):
    s, _, settings, _ = store
    k1 = s.key("こんにちは")
    assert len(k1) == 16
    assert s.key("こんにちは") == k1
    assert s.key("こんばんは") != k1
    assert s.key("こんにちは", voice_id="3") != k1
    assert s.key("こんにちは", params=VoiceParams(speed=1.1)) != k1
    settings.tts.voice_id = "8"
    assert s.key("こんにちは") != k1


async def test_ensure_synthesizes_once_and_uploads_once(store, fake):
    s, tts, _, _ = store
    name, dur = await s.ensure("うん")
    assert name.endswith(".wav") and dur > 0
    assert name in fake.sounds
    upload_calls = [c for c in fake.calls if "upload" in c]
    name2, _ = await s.ensure("うん")
    assert name2 == name
    assert len(tts.calls) == 1
    assert len([c for c in fake.calls if "upload" in c]) == len(upload_calls)


async def test_reupload_missing_after_daemon_restart(store, fake):
    s, tts, _, _ = store
    await s.prewarm(["うん", "そうだね"])
    assert len(fake.sounds) == 2
    fake.sounds.clear()  # デーモン再起動で /tmp が消えた
    n = await s.reupload_missing()
    assert n == 2 and len(fake.sounds) == 2
    assert len(tts.calls) == 2  # キャッシュから再アップロード、再合成はしない
    assert await s.reupload_missing() == 0


async def test_prewarm_progress_and_cache_files(store, tmp_path):
    s, _, _, _ = store
    seen = []
    await s.prewarm(["a", "b", "c"], progress=lambda i, n, t: seen.append((i, n, t)))
    assert seen == [(1, 3, "a"), (2, 3, "b"), (3, 3, "c")]
    assert len(list((tmp_path / "cache").glob("*.wav"))) == 3
    assert not list((tmp_path / "cache").glob("*.tmp"))
