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


# ---------------------------------------------------------------- 録音した肉声


def _library(tmp_path, stems=("A1", "B3"), names=("ひとみちゃん",)):
    from tests.test_recordings import tone, write_wav
    from app.recordings import RecordingLibrary

    base = tmp_path / "recordings" / "ryu"
    for stem in stems:
        write_wav(tone(0.4), path=base / f"{stem}.wav")
    for nm in names:
        write_wav(tone(0.6), path=base / "names" / f"{nm}.wav")
    return RecordingLibrary(base, "りゅうさん（録音）")


async def test_recorded_voice_is_used_and_uploaded_once(store, tmp_path, fake):
    from app.audio import SpeechItem

    s, tts, settings, _ = store
    s.recordings = _library(tmp_path)
    settings.tts.source = "recorded"
    name, dur = await s.ensure_item(SpeechItem(key="B3", text="難しく感じることもあるよね"))
    assert name and name in fake.sounds and 0.3 < dur < 0.7
    assert tts.calls == []  # 合成は呼ばれない
    assert (s.cache_dir / name).exists() and s.wanted[name] == "難しく感じることもあるよね"
    again, _ = await s.ensure_item(SpeechItem(key="B3", text="難しく感じることもあるよね"))
    assert again == name and len(fake.sounds) == 1  # 2 回目はアップロードしない
    # 名前入りの台詞は names/ から探す
    child, _ = await s.ensure_item(SpeechItem(key="names/ひとみちゃん", text="ひとみちゃんっていうんだね"))
    assert child != name and child in fake.sounds
    assert s.is_ready(SpeechItem(key="B3", text="難しく感じることもあるよね"))


async def test_missing_recording_is_silent_or_synthesized(store, tmp_path):
    from app.audio import SpeechItem

    s, tts, settings, _ = store
    s.recordings = _library(tmp_path)
    settings.tts.source = "recorded"
    item = SpeechItem(key="BC1", text="うん")
    name, dur = await s.ensure_item(item)
    assert name is None and dur == 0.0 and tts.calls == []  # 無音
    assert s.is_ready(item)  # 用意するものが無いので「準備中」にしない
    settings.tts.recorded.fallback = "voicevox"
    name, dur = await s.ensure_item(item)
    assert name and dur > 0 and tts.calls == [("うん", "1")]
    # 合成に切り替えれば録音があっても合成を使う
    settings.tts.source = "synth"
    name2, _ = await s.ensure_item(SpeechItem(key="B3", text="難しく感じることもあるよね"))
    assert name2 and ("難しく感じることもあるよね", "1") in tts.calls


async def test_recorded_audio_is_restored_after_daemon_restart(store, tmp_path, fake):
    from app.audio import SpeechItem

    s, _, settings, _ = store
    s.recordings = _library(tmp_path)
    settings.tts.source = "recorded"
    name, _ = await s.ensure_item(SpeechItem(key="A1", text="初めまして"))
    fake.sounds.clear()  # ロボットが再起動して音声が消えた
    (s.cache_dir / name).unlink()  # キャッシュも消えた
    assert await s.reupload_missing() == 1
    assert name in fake.sounds and (s.cache_dir / name).exists()  # 元の録音から作り直す


async def test_recording_change_makes_a_new_file(store, tmp_path, fake):
    from app.audio import SpeechItem
    from tests.test_recordings import tone, write_wav

    s, _, settings, _ = store
    lib = _library(tmp_path)
    s.recordings = lib
    settings.tts.source = "recorded"
    item = SpeechItem(key="A1", text="初めまして")
    first, _ = await s.ensure_item(item)
    write_wav(tone(0.9), path=lib.base / "A1.wav")  # 録り直し
    second, dur = await s.ensure_item(item)
    assert second != first and dur > 0.7 and second in fake.sounds
