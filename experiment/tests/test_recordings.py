import io
import math
import struct
import wave
from pathlib import Path

import pytest

from app.recordings import (
    RecordingError,
    RecordingLibrary,
    duration_of,
    levels,
    prepare,
    read_pcm,
    trim_silence,
)

DATA = Path(__file__).resolve().parents[1]


def tone(seconds: float, rate: int = 24000, amp: float = 0.3, freq: float = 300.0) -> list[float]:
    return [amp * math.sin(2 * math.pi * freq * i / rate) for i in range(int(seconds * rate))]


def noise(seconds: float, rate: int = 24000, amp: float = 0.001) -> list[float]:
    """−60 dBFS 前後の暗騒音(スマホ録音の無音部分に相当)。"""
    out, x = [], 12345
    for _ in range(int(seconds * rate)):
        x = (1103515245 * x + 12345) % (1 << 31)
        out.append(amp * (x / (1 << 30) - 1.0))
    return out


def write_wav(samples: list[float], rate: int = 24000, channels: int = 1, path: Path | None = None) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *(int(max(-32768, min(32767, v * 32767))) for v in samples)))
    data = buf.getvalue()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def test_read_pcm_handles_standard_and_extensible_wav():
    data = write_wav(tone(0.1))
    ch, rate, samples = read_pcm(data)
    assert ch == 1 and rate == 24000 and len(samples) == 2400
    assert max(samples) == pytest.approx(0.3, abs=0.01)
    # afconvert が書く WAVE_FORMAT_EXTENSIBLE(tag 0xFFFE)も読める
    ext = bytearray(data)
    ext[ext.index(b"fmt ") + 8 : ext.index(b"fmt ") + 10] = struct.pack("<H", 0xFFFE)
    assert read_pcm(bytes(ext))[2] == samples
    with pytest.raises(RecordingError):
        read_pcm(b"not a wav at all")
    with pytest.raises(RecordingError):
        read_pcm(b"RIFF____WAVE")


def test_trim_silence_finds_speech_between_quiet_noise():
    rate = 24000
    samples = noise(1.5, rate) + tone(1.0, rate) + noise(1.2, rate)
    out = trim_silence(samples, rate)
    # 声の 1.0 秒 + 前 0.05 秒 + 後ろ 0.15 秒だけ残る(窓 20 ms の粒度)
    assert len(out) / rate == pytest.approx(1.2, abs=0.05)
    assert max(abs(x) for x in out) == pytest.approx(0.3, abs=0.01)
    assert trim_silence(noise(0.5, rate), rate) == noise(0.5, rate)  # 全部無音なら触らない
    assert trim_silence([], rate) == []
    short = tone(0.01, rate)
    assert trim_silence(short, rate) == short


def test_prepare_trims_and_levels_the_audio(tmp_path):
    rate = 24000
    src = tmp_path / "B1.wav"
    write_wav(noise(1.0, rate) + tone(0.8, rate, amp=0.05) + noise(0.8, rate), rate, path=src)
    raw = prepare(src, loudness_db=0.0, trim=False)
    assert duration_of(raw) == pytest.approx(2.6, abs=0.02)
    out = prepare(src, loudness_db=10.0, trim=True)
    assert duration_of(out) == pytest.approx(1.0, abs=0.06)
    peak, rms = levels(out)
    assert -1.2 <= peak <= -0.8  # −1 dBFS まで持ち上げる(割れない)
    assert rms > levels(raw)[1] + 10
    with pytest.raises(RecordingError):
        prepare(tmp_path / "missing.wav")


def test_prepare_mixes_stereo_to_mono(tmp_path):
    rate = 24000
    left = tone(0.5, rate, amp=0.4)
    inter = [v for x in left for v in (x, -x)]  # 逆相にすると混ぜたとき 0 になる
    src = tmp_path / "st.wav"
    write_wav(inter, rate, channels=2, path=src)
    _, _, mono = read_pcm(prepare(src, trim=False))
    assert len(mono) == len(left)
    assert max(abs(x) for x in mono) < 0.01


def test_library_find_ids_names_and_traversal(tmp_path):
    base = tmp_path / "ryu"
    for stem in ("A1", "B3"):
        write_wav(tone(0.2), path=base / f"{stem}.wav")
    write_wav(tone(0.2), path=base / "names" / "ひとみちゃん.wav")
    write_wav(tone(0.2), path=tmp_path / "secret.wav")
    lib = RecordingLibrary(base, "りゅうさん")
    assert lib.find("A1") == base / "A1.wav"
    assert lib.find("names/ひとみちゃん") == base / "names" / "ひとみちゃん.wav"
    assert lib.find("BC1") is None
    assert lib.find("../secret") is None and lib.find("other/x") is None and lib.find("") is None
    assert lib.ids() == ["A1", "B3"] and lib.names() == ["ひとみちゃん"]
    rep = lib.report(["A1", "B3", "BC1", "C5"], child="ひとみちゃん")
    assert rep["missing"] == ["BC1", "C5"] and rep["child_ready"] is True and rep["label"] == "りゅうさん"
    assert lib.report(["A1"], child="たろうくん")["child_ready"] is False


def test_cache_key_tracks_file_and_settings(tmp_path):
    base = tmp_path / "ryu"
    src = base / "A1.wav"
    write_wav(tone(0.2), path=src)
    lib = RecordingLibrary(base)
    k = lib.cache_key(src, loudness_db=10.0, trim=True)
    assert k == lib.cache_key(src, loudness_db=10.0, trim=True)
    assert k != lib.cache_key(src, loudness_db=6.0, trim=True)
    assert k != lib.cache_key(src, loudness_db=10.0, trim=False)
    write_wav(tone(0.4), path=src)  # 録り直したら別のキーになる
    assert k != lib.cache_key(src, loudness_db=10.0, trim=True)


def test_real_recordings_are_usable():
    """実際に置いた録音が読めて、押してすぐ声が出る状態になっていること。"""
    lib = RecordingLibrary(DATA / "recordings" / "ryu", "りゅうさん（録音）")
    if not lib.base.is_dir() or not lib.ids():
        pytest.skip("録音が置かれていません")
    assert "A1" in lib.ids() and lib.names()
    src = lib.find("A1")
    assert src is not None
    wav = prepare(src, loudness_db=10.0, trim=True)
    peak, rms = levels(wav)
    assert 1.0 < duration_of(wav) < 20.0
    assert -1.2 <= peak <= -0.8 and rms > -20
    _, rate, samples = read_pcm(wav)
    head = samples[: int(0.1 * rate)]
    assert max(abs(x) for x in head) > 0.001  # 先頭の無音が詰められている
