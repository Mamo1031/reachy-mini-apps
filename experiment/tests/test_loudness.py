import io
import math
import struct
import wave

import pytest

from app.loudness import CEILING, process_wav, rms_dbfs


def _wav(samples, rate=24000, channels=1):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *(int(max(-32768, min(32767, s))) for s in samples)))
    return buf.getvalue()


def _peak(data):
    with wave.open(io.BytesIO(data)) as w:
        frames = w.readframes(w.getnframes())
    n = len(frames) // 2
    return max(abs(x) for x in struct.unpack(f"<{n}h", frames)) / 32768.0


def _speechlike(rate=24000):
    """声の代わり: 大きさの違うトーンバースト(−12 / −24 dBFS)と無音を交互に(抑揚のある声に近い)。"""
    out = []
    for k in range(6):
        amp = 32768 * 10 ** ((-12 if k % 2 == 0 else -24) / 20)
        for i in range(int(0.25 * rate)):
            t = i / rate
            out.append(amp * (0.6 * math.sin(2 * math.pi * 220 * t) + 0.4 * math.sin(2 * math.pi * 1800 * t)))
        out.extend([0.0] * int(0.15 * rate))
    return out


def test_boost_raises_level_without_clipping():
    src = _wav(_speechlike())
    out = process_wav(src, 8.0)
    assert len(out) == len(src)
    with wave.open(io.BytesIO(out)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    gain = rms_dbfs(out) - rms_dbfs(src)
    assert 8.0 <= gain <= 22.0, gain  # 圧縮 + 正規化でしっかり持ち上がる
    assert _peak(out) <= CEILING + 0.002  # −1 dBFS を超えない
    assert _peak(out) >= CEILING - 0.05  # ピークはほぼ上限まで使う
    # 張りを強くするほど平均レベルは上がる
    assert rms_dbfs(process_wav(src, 12.0)) > rms_dbfs(process_wav(src, 3.0))


def test_loud_input_is_limited_not_clipped():
    loud = [30000 * math.sin(2 * math.pi * 300 * i / 24000) for i in range(24000)]
    out = process_wav(_wav(loud), 12.0)
    assert _peak(out) <= CEILING + 0.002
    with wave.open(io.BytesIO(out)) as w:
        frames = w.readframes(w.getnframes())
    ints = struct.unpack(f"<{len(frames) // 2}h", frames)
    assert max(abs(x) for x in ints) < 32767  # 飽和サンプルなし


def test_silence_and_zero_boost_are_untouched():
    silent = _wav([0] * 2400)
    assert rms_dbfs(process_wav(silent, 8.0)) < -80
    src = _wav(_speechlike())
    assert process_wav(src, 0.0) == src


def test_stereo_and_other_widths():
    stereo = _wav(_speechlike()[:4800] * 2, channels=2)
    out = process_wav(stereo, 6.0)
    with wave.open(io.BytesIO(out)) as w:
        assert w.getnchannels() == 2
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(bytes([128] * 800))
    eight = buf.getvalue()
    assert process_wav(eight, 6.0) == eight  # 対象外はそのまま


def test_cache_key_and_store_apply_loudness(tmp_path):
    import asyncio

    from app.audio import AudioStore
    from app.config import Settings
    from tests.test_audio import FakeTTS

    settings = Settings()
    store = AudioStore(tmp_path, None, FakeTTS(), lambda: settings)  # type: ignore[arg-type]
    k0 = store.key("こんにちは")
    settings.tts.params.loudness_db = 8.0
    k1 = store.key("こんにちは")
    assert k0 != k1  # 張りを変えたら別の音声として作り直す
    key, wav = asyncio.run(store.synthesize_cached("こんにちは"))
    assert key == k1 and (tmp_path / f"{key}.wav").exists()
    assert _peak(wav) <= CEILING + 0.002
