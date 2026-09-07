"""TTS バックエンドの共通インターフェース。

将来、実験者の録音からのゼロショット音声クローン(Style-Bert-VITS2 / Fish Speech 等)を
追加するときは、このインターフェースを実装したクラスを 1 つ足すだけで差し替えられる。
辞書登録(register_pronunciation)は VOICEVOX 系にしか無い機能なので、無いバックエンドでは no-op。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from ..config import VoiceParams

ProgressFn = Callable[[str], None]


class TTSError(Exception):
    """TTS 側のエラー(str() は日本語のユーザー向けメッセージ)。"""


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    credit: str = ""


class TTSBackend:
    name: str = "base"

    async def ensure_ready(self, progress: ProgressFn | None = None) -> None:
        """エンジンが使える状態にする(必要なら起動して待つ)。使えなければ TTSError。"""
        raise NotImplementedError

    async def is_ready(self) -> bool:
        raise NotImplementedError

    async def list_voices(self) -> list[Voice]:
        raise NotImplementedError

    async def synthesize(self, text: str, voice_id: str, params: VoiceParams) -> bytes:
        """WAV(PCM)のバイト列を返す。"""
        raise NotImplementedError

    async def register_pronunciation(self, surface: str, pronunciation: str, accent_type: int = 0) -> None:
        return None

    def cache_signature(self, voice_id: str, params: VoiceParams) -> str:
        """キャッシュキーに含める、声に関わる全パラメータの文字列表現。"""
        return f"{self.name}|{voice_id}|{json.dumps(params.model_dump(), sort_keys=True)}"

    async def aclose(self) -> None:
        return None


def hiragana_to_katakana(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0x3041 <= code <= 0x3096:
            out.append(chr(code + 0x60))
        else:
            out.append(ch)
    return "".join(out)
