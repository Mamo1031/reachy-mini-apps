"""録音した肉声を台詞に対応づけ、ロボットで鳴らせる WAV に整える。

- 探索: `recordings/<声>/<台詞 ID>.<拡張子>`。名前入りの台詞は名前で探す(`name_key()`):
  `names/<なまえ+呼び方>` = 名前を含む文(A2「〇〇ちゃんっていうんだね…」)、
  `names_only/<なまえ+呼び方>` = 名前だけ(相づち「〇〇ちゃん！」)。
- 変換: m4a などは macOS 標準の afconvert(無ければ ffmpeg)で 24 kHz モノラルの PCM にする。
- 整形: 前後の無音を詰めて(押してすぐ声が出るように)、音量を揃える(loudness と同じ処理)。

録音が無い台詞は None を返す。呼び出し側が「無音のまま動作だけ再生」か「合成音声で代用」を決める。
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import math
import shutil
import struct
import subprocess
import unicodedata
import wave
from pathlib import Path
from typing import Callable

from .loudness import process_samples

log = logging.getLogger(__name__)

AUDIO_EXTS = (".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".opus")
NAME_DIR = "names"  # 名前を含む文の録音(A2)
NAME_ONLY_DIR = "names_only"  # 名前だけの録音(相づち)
NAME_DIRS = (NAME_DIR, NAME_ONLY_DIR)
TARGET_RATE = 24000
TRIM_FLOOR_DB = -30.0  # いちばん大きい区間からこれだけ下を無音とみなす
TRIM_WINDOW_S = 0.02
LEAD_PAD_S = 0.05
TAIL_PAD_S = 0.15


class RecordingError(Exception):
    pass


def name_folder(template: str) -> str:
    """名前入りの台詞(展開前の文面)が使う録音フォルダ。名前だけなら names_only、文なら names。

    「名前だけ」= {child} を除くと句読点・記号・空白(長音「ー」を含む)しか残らない台詞。
    「{child}！」「{child}〜」「「{child}」」はどれも名前だけ。正規表現にすると記号の列挙漏れで
    文の側に落ち、A2 の録音が鳴ってしまうので、文字種(Unicode カテゴリ)で判定する。
    """
    if "{child}" not in template:
        return NAME_DIR
    rest = template.replace("{child}", "")
    if all(c == "ー" or unicodedata.category(c)[0] in "PSZC" for c in rest):
        return NAME_ONLY_DIR
    return NAME_DIR


def name_key(template: str, child: str) -> str:
    """名前入りの台詞の録音を探す鍵("names/ひとみちゃん" / "names_only/ひとみちゃん")。"""
    return f"{name_folder(template)}/{child}"


def _find_decoder() -> list[str] | None:
    for name in ("afconvert", "ffmpeg"):
        if shutil.which(name):
            return [name]
    return None


def _decode_to_wav(src: Path) -> bytes:
    """音声ファイルを 24 kHz モノラルの WAV バイト列にする(WAV はそのまま読む)。"""
    if src.suffix.lower() == ".wav":
        try:
            return src.read_bytes()
        except OSError as e:
            raise RecordingError(f"{src.name} を読めません: {e}") from e
    tool = _find_decoder()
    if tool is None:
        raise RecordingError(f"{src.name} を変換できません(afconvert / ffmpeg が見つかりません)")
    out = src.with_suffix(".decoded.tmp.wav")
    if tool[0] == "afconvert":
        cmd = ["afconvert", "-f", "WAVE", "-d", f"LEI16@{TARGET_RATE}", "-c", "1", str(src), str(out)]
    else:
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", str(TARGET_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
        if r.returncode != 0:
            raise RecordingError(f"{src.name} を変換できません: {r.stderr.decode(errors='replace').strip()[:120]}")
        return out.read_bytes()
    except (OSError, subprocess.SubprocessError) as e:
        raise RecordingError(f"{src.name} を変換できません: {e}") from e
    finally:
        out.unlink(missing_ok=True)


def read_pcm(data: bytes) -> tuple[int, int, list[float]]:
    """WAV(16 bit)を (チャンネル数, サンプリングレート, −1〜1 のサンプル列)にする。

    afconvert は WAVE_FORMAT_EXTENSIBLE を書くので、標準ライブラリの wave では読めない。
    RIFF のチャンクを自分で辿る。
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise RecordingError("WAV ではありません")
    pos, fmt, raw = 12, None, None
    while pos + 8 <= len(data):
        cid, size = data[pos : pos + 4], struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        body = data[pos + 8 : pos + 8 + size]
        if cid == b"fmt " and len(body) >= 16:
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            raw = body
        pos += 8 + size + (size & 1)
    if fmt is None or raw is None:
        raise RecordingError("WAV の中身を読めません")
    tag, channels, rate, _, _, bits = fmt
    if tag not in (1, 0xFFFE) or bits != 16:
        raise RecordingError(f"対応していない WAV 形式です(tag={tag}, {bits} bit)")
    n = len(raw) // 2
    ints = struct.unpack(f"<{n}h", raw[: n * 2])
    return channels, rate, [x / 32768.0 for x in ints]


def _to_mono(samples: list[float], channels: int) -> list[float]:
    if channels <= 1:
        return samples
    return [sum(samples[i : i + channels]) / channels for i in range(0, len(samples) - channels + 1, channels)]


def trim_silence(samples: list[float], rate: int) -> list[float]:
    """前後の無音を詰める(前 0.05 秒 / 後ろ 0.15 秒だけ残す)。

    20 ms ごとの実効値で声の区間を探す。試し録りの雑音は −60 dBFS 前後で、瞬間的な山が
    サンプル単位のしきい値に引っかかるため、窓で均してから「いちばん大きい窓の 30 dB 下」で切る。
    """
    n = int(TRIM_WINDOW_S * rate)
    if not samples or n <= 0 or len(samples) < n * 2:
        return samples
    env = [math.sqrt(sum(x * x for x in samples[i : i + n]) / n) for i in range(0, len(samples) - n + 1, n)]
    peak = max(env)
    if peak <= 0:
        return samples
    thr = peak * 10 ** (TRIM_FLOOR_DB / 20.0)
    first = next((i for i, v in enumerate(env) if v > thr), None)
    if first is None:
        return samples
    last = len(env) - next(i for i, v in enumerate(reversed(env)) if v > thr)
    lo = max(0, first * n - int(LEAD_PAD_S * rate))
    hi = min(len(samples), last * n + int(TAIL_PAD_S * rate))
    return samples[lo:hi]


def _write_wav(samples: list[float], rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *(max(-32768, min(32767, int(round(v * 32767.0)))) for v in samples)))
    return buf.getvalue()


def prepare(src: Path, *, loudness_db: float = 0.0, trim: bool = True) -> bytes:
    """録音ファイル → ロボットへ送れる WAV(無音を詰めて音量を揃えたもの)。"""
    channels, rate, samples = read_pcm(_decode_to_wav(src))
    samples = _to_mono(samples, channels)
    if not samples:
        raise RecordingError(f"{src.name} は空です")
    if trim:
        samples = trim_silence(samples, rate)
    if loudness_db > 0:
        samples = process_samples(samples, rate, loudness_db)
    return _write_wav(samples, rate)


def duration_of(wav: bytes) -> float:
    _, rate, samples = read_pcm(wav)
    return len(samples) / rate if rate else 0.0


class RecordingLibrary:
    """1 人分の録音フォルダ。台詞 ID(と名前)からファイルを探す。"""

    def __init__(self, base: Path, label: str = "") -> None:
        self.base = base
        self.label = label

    def find(self, key: str) -> Path | None:
        """key は台詞 ID("B3")か、名前入りの台詞("names/ひとみちゃん"、"names_only/ひとみちゃん")。"""
        if "/" in key:
            folder, stem = key.split("/", 1)
            if folder not in NAME_DIRS:
                return None
            d = self.base / folder
        else:
            d, stem = self.base, key
        if not stem or "/" in stem or stem in (".", ".."):
            return None
        for ext in AUDIO_EXTS:
            p = d / f"{stem}{ext}"
            if p.is_file():
                return p
        return None

    @staticmethod
    def _stems(d: Path) -> list[str]:
        """フォルダ内の音声ファイル名(拡張子なし)。

        macOS は濁点・半濁点を分解形(NFD)で保存するため、開始画面で入力した合成形(NFC)と
        そのまま比べると一致しない。表示・照合用に NFC へそろえる(探索自体は OS が吸収する)。
        """
        if not d.is_dir():
            return []
        return sorted({unicodedata.normalize("NFC", p.stem) for p in d.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS})

    def ids(self) -> list[str]:
        """録音がある台詞 ID。"""
        return self._stems(self.base)

    def names(self) -> list[str]:
        """名前を含む文の録音がある名前(「ひとみちゃん」のような呼び方込みの表記)。"""
        return self._stems(self.base / NAME_DIR)

    def names_only(self) -> list[str]:
        """名前だけの録音がある名前。"""
        return self._stems(self.base / NAME_ONLY_DIR)

    def cache_key(self, src: Path, *, loudness_db: float, trim: bool) -> str:
        """録音ファイルの中身と整形条件が変わったら別のキーになる。"""
        st = src.stat()
        sig = f"rec|{src.resolve()}|{st.st_size}|{st.st_mtime_ns}|{loudness_db}|{int(trim)}"
        return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]

    async def prepare_async(self, src: Path, *, loudness_db: float, trim: bool) -> bytes:
        return await asyncio.to_thread(prepare, src, loudness_db=loudness_db, trim=trim)

    def report(self, wanted_ids: list[str], child: str | None = None) -> dict[str, object]:
        """UI 用: どの台詞の録音があり、どれが無いか。"""
        have = set(self.ids())
        missing = [i for i in wanted_ids if i not in have]
        out: dict[str, object] = {
            "label": self.label,
            "dir": str(self.base),
            "ids": sorted(have),
            "missing": missing,
            "names": self.names(),
            "names_only": self.names_only(),
        }
        if child is not None:
            out["child_ready"] = self.find(f"{NAME_DIR}/{child}") is not None
            out["child_name_only_ready"] = self.find(f"{NAME_ONLY_DIR}/{child}") is not None
        return out


def levels(wav: bytes) -> tuple[float, float]:
    """検証用: (ピーク, RMS) を dBFS で返す。"""
    _, _, samples = read_pcm(wav)
    if not samples:
        return (-math.inf, -math.inf)
    peak = max(abs(x) for x in samples)
    rms = math.sqrt(sum(x * x for x in samples) / len(samples))
    to_db: Callable[[float], float] = lambda v: 20 * math.log10(v) if v > 0 else -math.inf
    return to_db(peak), to_db(rms)
