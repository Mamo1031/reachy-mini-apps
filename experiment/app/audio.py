"""音声ストア: 合成 WAV のキャッシュと、ロボットへのアップロード管理。

キャッシュキー = hash(バックエンド名 | 話者 | 音声パラメータ | 本文に含まれる辞書登録 | 展開後テキスト)。
ロボット上のファイル名はキー + ".wav"(同名上書きなので再アップロードは冪等)。
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import wave
from pathlib import Path
from typing import Callable

from .config import Settings, VoiceParams
from .robot import RobotClient
from .tts.base import TTSBackend, TTSError

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


class AudioError(Exception):
    pass


def wav_duration(data: bytes) -> float:
    """WAV の長さ(秒)。壊れていれば AudioError。"""
    try:
        with wave.open(io.BytesIO(data)) as w:
            frames = w.getnframes()
            rate = w.getframerate() or 1
    except (wave.Error, EOFError, ValueError) as e:
        raise AudioError(f"WAV を読めません: {e}") from e
    if frames <= 0:
        raise AudioError("WAV が空です")
    return frames / rate


class AudioStore:
    def __init__(self, cache_dir: Path, robot: RobotClient, tts: TTSBackend, settings_ref: Callable[[], Settings]) -> None:
        self.cache_dir = cache_dir
        self.robot = robot
        self.tts = tts
        self.settings_ref = settings_ref
        self._durations: dict[str, float] = {}
        self._uploaded: set[str] = set()  # ロボット上にあると分かっている basename
        self.wanted: dict[str, str] = {}  # basename → テキスト(今のセッションで必要なもの)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ keys
    def _voice(self) -> tuple[str, VoiceParams]:
        s = self.settings_ref().tts
        return s.voice_id, s.params

    def key(self, text: str, voice_id: str | None = None, params: VoiceParams | None = None) -> str:
        vid, prm = self._voice()
        sig = self.tts.cache_signature(voice_id or vid, params or prm)
        # テキストに含まれる単語の辞書登録(読み・アクセント)が変わったら別の音声になる
        dict_sig = "|".join(
            f"{e.surface}={e.pronunciation}/{e.accent_type}" for e in sorted(self.settings_ref().tts.user_dict, key=lambda e: e.surface) if e.surface and e.surface in text
        )
        return hashlib.sha256(f"{sig}|{dict_sig}|{text}".encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def basename(key: str) -> str:
        return f"{key}.wav"

    def path(self, key: str) -> Path:
        return self.cache_dir / self.basename(key)

    # ------------------------------------------------------------ synth
    async def synthesize_cached(self, text: str, voice_id: str | None = None, params: VoiceParams | None = None) -> tuple[str, bytes]:
        """キャッシュにあればそれを、無ければ合成して保存する。戻り値: (key, wav)。

        キャッシュが壊れていれば捨てて合成し直す。合成結果も検証してから保存する。
        """
        key = self.key(text, voice_id, params)
        p = self.path(key)
        if p.exists():
            data = p.read_bytes()
            try:
                self._durations[key] = wav_duration(data)
                return key, data
            except AudioError as e:
                log.warning("corrupt cache %s (%s) — re-synthesizing", p.name, e)
                p.unlink(missing_ok=True)
        vid, prm = self._voice()
        wav = await self.tts.synthesize(text, voice_id or vid, params or prm)
        try:
            self._durations[key] = wav_duration(wav)
        except AudioError as e:
            raise TTSError(f"音声合成の結果が不正です({text[:12]}…): {e}") from e
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(wav)
        tmp.replace(p)
        return key, wav

    def duration(self, key: str) -> float:
        if key not in self._durations:
            self._durations[key] = wav_duration(self.path(key).read_bytes())
        return self._durations[key]

    def is_synthesized(self, text: str) -> bool:
        return self.path(self.key(text)).exists()

    # ------------------------------------------------------------ upload
    async def ensure(self, text: str) -> tuple[str, float]:
        """合成(必要なら)→ アップロード(必要なら)。戻り値: (ロボット上の basename, 秒)。"""
        async with self._lock:
            key, wav = await self.synthesize_cached(text)
            name = self.basename(key)
            self.wanted[name] = text
            if name not in self._uploaded:
                await self.robot.upload_sound(name, wav)
                self._uploaded.add(name)
            return name, self._durations[key]

    def forget(self, name: str) -> None:
        """再生で 404 になった等、ロボット上に無いと分かった音声を忘れる(次の ensure で再アップロード)。"""
        self._uploaded.discard(name)

    async def prewarm(self, texts: list[str], progress: ProgressFn | None = None, upload: bool = True) -> list[str]:
        """まとめて合成(+アップロード)。失敗したテキストは飛ばして続け、失敗メッセージの一覧を返す。"""
        failures: list[str] = []
        total = len(texts)
        for i, text in enumerate(texts, 1):
            if progress:
                progress(i, total, text)
            try:
                if upload:
                    await self.ensure(text)
                else:
                    async with self._lock:
                        key, _ = await self.synthesize_cached(text)
                        self.wanted[self.basename(key)] = text
            except (TTSError, AudioError) as e:
                failures.append(f"{text[:14]}…: {e}")
                log.warning("prewarm failed for %r: %s", text, e)
        return failures

    async def reupload_missing(self, progress: ProgressFn | None = None) -> int:
        """ロボット上の一覧と突き合わせ、必要な音声で欠けているものを再アップロードする。

        一覧にあるものは「アップロード済み」として覚え直す(次の再生で無駄に再アップロードしない)。
        """
        async with self._lock:
            present = await self.robot.list_sounds()
            self._uploaded = {n for n in self.wanted if n in present}
            missing = [n for n in self.wanted if n not in present]
            for i, name in enumerate(missing, 1):
                if progress:
                    progress(i, len(missing), self.wanted[name])
                p = self.cache_dir / name
                if p.exists():
                    wav = p.read_bytes()
                else:  # キャッシュが消えていれば合成し直す(名前が変わるので wanted を付け替える)
                    key, wav = await self.synthesize_cached(self.wanted[name])
                    if self.basename(key) != name:
                        self.wanted[self.basename(key)] = self.wanted.pop(name)
                        name = self.basename(key)
                await self.robot.upload_sound(name, wav)
                self._uploaded.add(name)
            return len(missing)

    def forget_uploads(self) -> None:
        """デーモン再起動などでロボット上の音声が消えたときに呼ぶ。"""
        self._uploaded.clear()
