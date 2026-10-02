#!/bin/zsh
# Build the zip handed to the experimenter: double-click 「はじめる.command」 and the app starts.
#
#   cd experiment && ./tools/make_release.sh
#
# Uses ditto so the executable bit on the launcher survives zipping. Leaves out everything
# the experimenter does not need: git data, virtualenvs, the 2 GB VOICEVOX engine, caches,
# logs, tests and developer notes. Recorded audio IS included (it is not in git).
set -euo pipefail

cd "$(dirname "$0")/../.."                      # repository root
ROOT="$PWD"
NAME="実験コントローラ"
STAMP="$(date +%Y%m%d)"
OUT="$ROOT/dist"
STAGE="$(mktemp -d)/$NAME"
ZIP="$OUT/${NAME}-${STAMP}.zip"

trap 'rm -rf "$(dirname "$STAGE")"' EXIT

mkdir -p "$STAGE/experiment" "$OUT"
cp "$ROOT/はじめる.command" "$ROOT/README.md" "$ROOT/はじめに読んでください.txt" "$STAGE/"
chmod +x "$STAGE/はじめる.command"

# tools/ holds the 2 GB VOICEVOX engine and developer scripts; setup_voicevox.sh (kept, it sits
# one level up) downloads the engine on demand for anyone who wants the synthesized voice.
rsync -a \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '.DS_Store' --exclude '.pytest_cache/' \
  --exclude 'cache/' --exclude 'logs/' --exclude 'tools/' --exclude 'tests/' \
  --exclude '*.bak' --exclude '*.broken-*' \
  "$ROOT/experiment/" "$STAGE/experiment/"

recordings=$(find "$STAGE/experiment/recordings" -type f 2>/dev/null | wc -l | tr -d ' ')
[[ "$recordings" -gt 0 ]] || { echo "!! 録音が 1 本も入っていません。experiment/recordings/ を確認してください。" >&2; exit 1; }
[[ -x "$STAGE/はじめる.command" ]] || { echo "!! はじめる.command に実行権限がありません。" >&2; exit 1; }

rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "$STAGE" "$ZIP"

echo "できました: $ZIP"
echo "  大きさ: $(du -h "$ZIP" | cut -f1)   録音: ${recordings} 本"
echo
echo "渡すときに伝えること:"
echo "  1. zip を展開して、出てきたフォルダの「はじめる」をダブルクリック"
echo "  2. 初回だけ macOS が止めるので、同梱の「はじめに読んでください.txt」の手順で 1 回だけ許可する"
echo "     (macOS 15: システム設定 > プライバシーとセキュリティ > このまま開く / 14 以前: 右クリック > 開く)"
