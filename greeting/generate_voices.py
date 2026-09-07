"""挨拶音声(WAV)を生成する。初回のみ実行: uv run generate_voices.py

macOS標準のsayコマンド(日本語音声Kyoko)でAIFFを生成し、
afconvertでWAV(16bit/22.05kHz/モノラル)に変換する。ネット接続・APIキー不要。
"""

import subprocess
from pathlib import Path

SOUNDS_DIR = Path(__file__).parent / "sounds"
VOICE = "Kyoko"
PHRASES = {
    "hello": "こんにちは",
    "goodbye": "さようなら",
}


def main() -> None:
    SOUNDS_DIR.mkdir(exist_ok=True)
    for name, text in PHRASES.items():
        aiff = SOUNDS_DIR / f"{name}.aiff"
        wav = SOUNDS_DIR / f"{name}.wav"
        subprocess.run(["say", "-v", VOICE, "-o", str(aiff), text], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@22050", "-c", "1", str(aiff), str(wav)],
            check=True,
        )
        aiff.unlink()
        print(f"生成しました: {wav} (「{text}」)")


if __name__ == "__main__":
    main()
