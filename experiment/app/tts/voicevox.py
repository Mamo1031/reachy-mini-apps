"""VOICEVOX エンジン(ローカル HTTP)バックエンド。未起動なら自動起動する。"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from ..config import VoiceParams, VoicevoxSettings
from .base import ProgressFn, TTSBackend, TTSError, Voice

log = logging.getLogger(__name__)


class VoicevoxBackend(TTSBackend):
    name = "voicevox"

    def __init__(self, settings: VoicevoxSettings, data_dir: Path, log_dir: Path) -> None:
        self.settings = settings
        self.data_dir = data_dir
        self.log_dir = log_dir
        self._proc: asyncio.subprocess.Process | None = None
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(connect=1.0, read=30.0, write=10.0, pool=1.0))

    @property
    def base_url(self) -> str:
        return f"http://{self.settings.host}:{self.settings.port}"

    @property
    def engine_path(self) -> Path:
        p = Path(self.settings.engine_dir)
        if not p.is_absolute():
            p = self.data_dir / p
        return p / "run"

    # ------------------------------------------------------------ lifecycle
    async def is_ready(self) -> bool:
        try:
            r = await self._client.get("/version", timeout=1.0)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure_ready(self, progress: ProgressFn | None = None) -> None:
        if await self.is_ready():
            return
        if not self.settings.autostart:
            raise TTSError(f"VOICEVOX エンジンが {self.base_url} で動いていません(自動起動は無効)。")
        if not self.engine_path.exists():
            raise TTSError(f"VOICEVOX エンジンが見つかりません: {self.engine_path}(experiment/setup_voicevox.sh を実行してください)")
        if self._proc is None or self._proc.returncode is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logfile = open(self.log_dir / "voicevox.log", "ab")
            log.info("starting VOICEVOX engine: %s", self.engine_path)
            self._proc = await asyncio.create_subprocess_exec(
                str(self.engine_path),
                "--host",
                self.settings.host,
                "--port",
                str(self.settings.port),
                cwd=str(self.engine_path.parent),
                stdout=logfile,
                stderr=asyncio.subprocess.STDOUT,
            )
        t0 = time.monotonic()
        while time.monotonic() - t0 < 120.0:
            if await self.is_ready():
                if progress:
                    progress("VOICEVOX 起動完了")
                return
            if self._proc.returncode is not None:
                raise TTSError(f"VOICEVOX エンジンが終了しました(code {self._proc.returncode})。cache/voicevox.log を確認してください。")
            if progress:
                progress(f"VOICEVOX 起動中… {int(time.monotonic() - t0)} 秒")
            await asyncio.sleep(1.0)
        raise TTSError("VOICEVOX エンジンの起動がタイムアウトしました(120 秒)。")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def stop_engine(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()

    # ------------------------------------------------------------ api
    async def _call(self, method: str, path: str, what: str, **kw: Any) -> httpx.Response:
        try:
            r = await self._client.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise TTSError(f"VOICEVOX との通信に失敗しました({what}): {type(e).__name__}") from e
        if r.status_code >= 400:
            raise TTSError(f"VOICEVOX がエラーを返しました({what}, HTTP {r.status_code}): {r.text[:200]}")
        return r

    async def list_voices(self) -> list[Voice]:
        r = await self._call("GET", "/speakers", "話者一覧")
        voices: list[Voice] = []
        for sp in r.json():
            for st in sp.get("styles", []):
                voices.append(Voice(id=str(st["id"]), name=f"{sp['name']} {st['name']}", credit=f"VOICEVOX:{sp['name']}"))
        return voices

    async def synthesize(self, text: str, voice_id: str, params: VoiceParams) -> bytes:
        speaker = int(voice_id)
        q = (await self._call("POST", "/audio_query", "audio_query", params={"text": text, "speaker": speaker})).json()
        q["speedScale"] = params.speed
        q["pitchScale"] = params.pitch
        q["intonationScale"] = params.intonation
        q["volumeScale"] = params.volume
        q["postPhonemeLength"] = params.post_phoneme
        if "pauseLengthScale" in q:
            q["pauseLengthScale"] = params.pause_scale
        q["outputSamplingRate"] = 24000
        q["outputStereo"] = False
        r = await self._call("POST", "/synthesis", "synthesis", params={"speaker": speaker}, json=q)
        wav = r.content
        if not wav.startswith(b"RIFF"):
            raise TTSError("VOICEVOX から WAV 以外の応答が返りました")
        return wav

    async def register_pronunciation(self, surface: str, pronunciation: str, accent_type: int = 0) -> None:
        existing = (await self._call("GET", "/user_dict", "辞書取得")).json()
        params = {"surface": surface, "pronunciation": pronunciation, "accent_type": int(accent_type), "word_type": "PROPER_NOUN", "priority": 8}
        for word_id, entry in existing.items():
            if entry.get("surface") == surface:
                await self._call("PUT", f"/user_dict_word/{word_id}", "辞書更新", params=params)
                return
        await self._call("POST", "/user_dict_word", "辞書登録", params=params)
