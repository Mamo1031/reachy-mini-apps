#!/bin/zsh
# VOICEVOX エンジン(ヘッドレス)を tools/voicevox/ に導入する。冪等: 導入済みなら何もしない。
# 使い方: ./setup_voicevox.sh            (Apple Silicon Mac 向け)
set -euo pipefail
VERSION="${VOICEVOX_VERSION:-0.25.2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$HERE/tools/voicevox"
ENGINE_DIR="$DEST/macos-arm64"
ARCHIVE="voicevox_engine-macos-arm64-${VERSION}.7z.001"
URL="https://github.com/VOICEVOX/voicevox_engine/releases/download/${VERSION}/${ARCHIVE}"

if [[ -x "$ENGINE_DIR/run" ]]; then
  echo "VOICEVOX エンジンは導入済みです: $ENGINE_DIR/run"
  exit 0
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "このスクリプトは Apple Silicon (arm64) 用です。" >&2
  exit 1
fi
mkdir -p "$DEST"
if ! command -v 7zz >/dev/null 2>&1; then
  echo "7-Zip (sevenzip) を Homebrew で導入します..."
  brew install sevenzip
fi
if [[ ! -f "$DEST/$ARCHIVE" ]]; then
  echo "エンジンをダウンロードします(約 1.8 GB): $URL"
  curl -L --fail --retry 3 --retry-delay 5 -C - -o "$DEST/$ARCHIVE.part" "$URL"
  mv "$DEST/$ARCHIVE.part" "$DEST/$ARCHIVE"
fi
echo "展開中..."
(cd "$DEST" && 7zz x -y "$ARCHIVE" >/dev/null)
if [[ ! -x "$ENGINE_DIR/run" ]]; then
  # 展開先ディレクトリ名が異なる場合に備えて run を探す
  FOUND="$(find "$DEST" -maxdepth 3 -name run -type f | head -1 || true)"
  if [[ -z "$FOUND" ]]; then echo "展開後に run が見つかりません" >&2; exit 1; fi
  mv "$(dirname "$FOUND")" "$ENGINE_DIR"
fi
echo "Gatekeeper の隔離属性を解除します..."
xattr -rd com.apple.quarantine "$ENGINE_DIR" 2>/dev/null || true
chmod +x "$ENGINE_DIR/run"
rm -f "$DEST/$ARCHIVE"
echo "完了: $ENGINE_DIR/run"
