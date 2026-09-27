"""音声ストア: 合成 WAV と録音した肉声のキャッシュ、ロボットへのアップロード管理。

声の出どころは 2 つある(設定 `tts.source`)。
- synth   : VOICEVOX で合成する。キー = hash(バックエンド | 話者 | パラメータ | 辞書 | 展開後テキスト)
- recorded: `recordings/<声>/` の録音を使う。キー = hash(ファイルの中身と整形条件)。録音が無い台詞は
            設定 `tts.recorded.fallback` に従って合成で代用するか、無音(name=None)を返す

ロボット上のファイル名はキー + ".wav"(同名上書きなので再アップロードは冪等)。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import logging
import wave
from pathlib import Path
from typing import Callable

from dataclasses import dataclass

from .config import Settings, VoiceParams
from .loudness import process_wav
from .recordings import RecordingLibrary, RecordingError
from .robot import RobotClient
from .tts.base import TTSBackend, TTSError

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


class AudioError(Exception):
    pass


@dataclass(frozen=True)
class SpeechItem:
    """再生する 1 つの音声。key は録音を探すための鍵(台詞 ID か "names/ひとみちゃん")。"""

    key: str
    text: str


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
        self._sources: dict[str, Path] = {}  # basename → 録音ファイル(合成ではなく録音を使ったもの)
        self.recordings: RecordingLibrary | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ source
    def recording_for(self, key: str) -> Path | None:
        """この台詞に使う録音ファイル(録音を使わない設定・録音が無い場合は None)。"""
        if self.recordings is None or self.settings_ref().tts.source != "recorded":
            return None
        return self.recordings.find(key)

    def _fallback_to_synth(self) -> bool:
        return self.settings_ref().tts.recorded.fallback == "voicevox"

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
        prm = params or prm
        wav = await self.tts.synthesize(text, voice_id or vid, prm)
        if prm.loudness_db > 0:
            wav = await asyncio.to_thread(process_wav, wav, prm.loudness_db)
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

    def is_ready(self, item: SpeechItem) -> bool:
        """すぐ鳴らせるか(準備中の表示を出すかの判断)。"""
        src = self.recording_for(item.key)
        if src is not None:
            s = self.settings_ref().tts
            with contextlib.suppress(OSError):
                return self.path(self.recordings.cache_key(src, loudness_db=s.params.loudness_db, trim=s.recorded.trim_silence)).exists()  # type: ignore[union-attr]
            return False
        if self.recordings is not None and self.settings_ref().tts.source == "recorded" and not self._fallback_to_synth():
            return True  # 無音: 用意するものが無い
        return self.is_synthesized(item.text)

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

    async def ensure_item(self, item: SpeechItem) -> tuple[str | None, float]:
        """録音か合成を用意してアップロードする。戻り値: (ロボット上の basename か None, 秒)。

        None は「録音が無く、合成で代用しない設定」= 無音。呼び出し側は動作だけ再生する。
        """
        src = self.recording_for(item.key)
        if src is not None:
            async with self._lock:
                return await self._ensure_recorded(src, item.text)
        if self._fallback_to_synth() or self.settings_ref().tts.source != "recorded":
            return await self.ensure(item.text)
        return None, 0.0

    async def _ensure_recorded(self, src: Path, text: str) -> tuple[str, float]:
        """録音を整形してキャッシュ・アップロードする(ロックの中で呼ぶ)。"""
        assert self.recordings is not None
        s = self.settings_ref().tts
        try:
            key = self.recordings.cache_key(src, loudness_db=s.params.loudness_db, trim=s.recorded.trim_silence)
        except OSError as e:
            raise AudioError(f"録音を読めません({src.name}): {e}") from e
        p, name = self.path(key), self.basename(key)
        wav: bytes | None = None
        if p.exists():
            try:
                self._durations[key] = wav_duration(p.read_bytes())
            except AudioError:
                log.warning("corrupt cache %s — re-preparing", p.name)
                p.unlink(missing_ok=True)
        if not p.exists():
            try:
                wav = await self.recordings.prepare_async(src, loudness_db=s.params.loudness_db, trim=s.recorded.trim_silence)
            except RecordingError as e:
                raise AudioError(str(e)) from e
            self._durations[key] = wav_duration(wav)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(wav)
            tmp.replace(p)
        self.wanted[name] = text
        self._sources[name] = src
        if name not in self._uploaded:
            await self.robot.upload_sound(name, wav if wav is not None else p.read_bytes())
            self._uploaded.add(name)
        return name, self._durations[key]

    def forget(self, name: str) -> None:
        """再生で 404 になった等、ロボット上に無いと分かった音声を忘れる(次の ensure で再アップロード)。"""
        self._uploaded.discard(name)

    async def prewarm_items(self, items: list[SpeechItem], progress: ProgressFn | None = None) -> list[str]:
        """まとめて用意(録音の整形 or 合成 + アップロード)。失敗は飛ばして続け、メッセージ一覧を返す。"""
        failures: list[str] = []
        for i, item in enumerate(items, 1):
            if progress:
                progress(i, len(items), item.text)
            try:
                await self.ensure_item(item)
            except (TTSError, AudioError) as e:
                failures.append(f"{item.text[:14]}…: {e}")
                log.warning("prewarm failed for %r: %s", item.text, e)
        return failures

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
                elif name in self._sources:  # 録音: 元ファイルから作り直す(キーは変わらない)
                    s = self.settings_ref().tts
                    assert self.recordings is not None
                    wav = await self.recordings.prepare_async(self._sources[name], loudness_db=s.params.loudness_db, trim=s.recorded.trim_silence)
                    p.write_bytes(wav)
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
