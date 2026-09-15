"""合成した声の「音量感」を、クリップさせずに上げる(大きな部屋向け)。

ロボットのスピーカーは小さく、音量はデーモン側でも上限に張り付いている。単純に振幅を 2 倍にしても
ピークがクリップするだけで大きくは聞こえない。ここでは
  1. 100 Hz 以下を落とす(スピーカーが再生できず、ヘッドルームだけ食う成分)
  2. コンプレッサで山を抑えて谷を持ち上げる(平均レベル = 音量感が上がる)
  3. ソフトリミッタで −1 dBFS を超えないようにする
を純 Python で行う(24 kHz・16 bit・モノラル、5 秒で 0.3 秒程度)。数値計算ライブラリは使わない。
"""

from __future__ import annotations

import io
import math
import struct
import wave

CEILING = 10 ** (-1.0 / 20.0)  # −1 dBFS: 出力のピーク上限
RATIO = 4.0
ATTACK_S = 0.003
RELEASE_S = 0.15
HIGHPASS_HZ = 100.0
LIMIT_RELEASE_S = 0.05
MAX_MAKEUP = 10 ** (30.0 / 20.0)  # 無音に近いファイルを増幅しすぎない


def process_samples(samples: list[float], rate: int, boost_db: float) -> list[float]:
    """float(−1〜1)のサンプル列を処理して返す。

    boost_db が大きいほどコンプレッサのしきい値を下げて山を強く抑え、そのぶん全体を持ち上げる。
    持ち上げ量は自動(処理後のピークがちょうど −1 dBFS になるように)。
    """
    if not samples or boost_db <= 0:
        return samples
    # 1) 1 次ハイパス
    rc = 1.0 / (2.0 * math.pi * HIGHPASS_HZ)
    dt = 1.0 / rate
    alpha = rc / (rc + dt)
    hp: list[float] = [0.0] * len(samples)
    prev_x = prev_y = 0.0
    for i, x in enumerate(samples):
        y = alpha * (prev_y + x - prev_x)
        hp[i] = y
        prev_x, prev_y = x, y
    # 2) コンプレッサ(ピーク検出、アタック / リリース付き)。しきい値は張りの量で下げる
    att = math.exp(-1.0 / (ATTACK_S * rate))
    rel = math.exp(-1.0 / (RELEASE_S * rate))
    peak_in = max(abs(x) for x in hp)
    if peak_in <= 1e-6:
        return samples  # 無音
    thr = peak_in * 10 ** (-(4.0 + boost_db) / 20.0)  # 入力ピーク基準で 4 + boost dB 下
    slope = 1.0 - 1.0 / RATIO
    env = 0.0
    env_peak = 0.0  # 圧縮後の包絡線のピーク(立ち上がりの一瞬の飛び出しは含めない)
    out: list[float] = [0.0] * len(hp)
    for i, x in enumerate(hp):
        a = abs(x)
        env = a + (att if a > env else rel) * (env - a)
        gain = (thr / env) ** slope if env > thr else 1.0
        out[i] = x * gain
        if env * gain > env_peak:
            env_peak = env * gain
    # 3) 包絡線のピークが −1 dBFS になるように持ち上げ、飛び出しは瞬時応答のリミッタで押さえる
    makeup = min(MAX_MAKEUP, CEILING / env_peak) if env_peak > 0 else 1.0
    rel_l = math.exp(-1.0 / (LIMIT_RELEASE_S * rate))
    lim_env = 0.0
    for i, x in enumerate(out):
        y = x * makeup
        a = abs(y)
        lim_env = a if a > lim_env else lim_env * rel_l
        out[i] = y * (CEILING / lim_env) if lim_env > CEILING else y
    return out


def process_wav(data: bytes, boost_db: float) -> bytes:
    """16 bit PCM の WAV に process_samples を適用する。対象外の形式(8 bit / 32 bit)はそのまま返す。"""
    if boost_db <= 0:
        return data
    with wave.open(io.BytesIO(data)) as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    if params.sampwidth != 2 or params.nchannels < 1:
        return data
    n = len(frames) // 2
    ints = struct.unpack(f"<{n}h", frames[: n * 2])
    ch = params.nchannels
    result: list[float] = [0.0] * n
    for c in range(ch):
        chan = [ints[i] / 32768.0 for i in range(c, n, ch)]
        done = process_samples(chan, params.framerate, boost_db)
        for k, i in enumerate(range(c, n, ch)):
            result[i] = done[k]
    packed = struct.pack(f"<{n}h", *(max(-32768, min(32767, int(round(v * 32767.0)))) for v in result))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setparams(params)
        out.writeframes(packed)
    return buf.getvalue()


def rms_dbfs(data: bytes) -> float:
    """検証用: 16 bit WAV の RMS(dBFS)。"""
    with wave.open(io.BytesIO(data)) as w:
        frames = w.readframes(w.getnframes())
    n = len(frames) // 2
    if n == 0:
        return -math.inf
    ints = struct.unpack(f"<{n}h", frames[: n * 2])
    return 20 * math.log10(max(1e-9, math.sqrt(sum(x * x for x in ints) / n) / 32768.0))
