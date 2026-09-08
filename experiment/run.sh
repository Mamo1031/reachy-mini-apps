#!/bin/zsh
# 実験コントローラを起動する。ブラウザで http://localhost:8080 (iPad からは http://<MacのIP>:8080)
# VOICEVOX エンジンはサーバーが自動起動する(初回は ./setup_voicevox.sh が必要)。
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8080}"
if [[ ! -x tools/voicevox/macos-arm64/run ]]; then
  echo "VOICEVOX エンジンが未導入です。先に ./setup_voicevox.sh を実行してください。" >&2
fi
IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
echo "操作画面: http://localhost:${PORT}   ${IP:+(同じネットワークの iPad から: http://${IP}:${PORT})}"
exec uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --no-access-log
