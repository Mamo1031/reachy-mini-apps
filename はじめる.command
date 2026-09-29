#!/bin/zsh
# ダブルクリックで実験コントローラを起動する。ターミナルの知識は不要。
#
# やること: 必要なライブラリを用意 → サーバーを起動 → ブラウザで操作画面を開く。
# 止めるときはこのウィンドウで Control + C を押すか、ウィンドウを閉じる。

cd "$(dirname "$0")/experiment" || { echo "experiment フォルダが見つかりません"; read -r; exit 1; }
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
unset VIRTUAL_ENV   # 別の仮想環境が有効でも、このフォルダのものを使う
PORT="${PORT:-8080}"

printf '\033]0;Reachy Mini 実験コントローラ\007'   # ウィンドウのタイトル
echo "============================================"
echo " Reachy Mini 実験コントローラ"
echo "============================================"
echo

fail() {
  echo
  echo "!! $1"
  echo
  echo "このウィンドウを閉じて構いません。Enter キーで閉じます。"
  read -r
  exit 1
}

if [[ "$(uname -m)" != "arm64" ]]; then
  fail "この Mac では動きません(Apple シリコンの Mac が必要です)。"
fi

# --- 1) uv(必要なライブラリを入れる道具)
if ! command -v uv > /dev/null 2>&1; then
  echo "初回の準備をします(必要な道具を入れます。数分かかります)…"
  if ! curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1; then
    fail "準備に失敗しました。インターネットにつながっているか確認してください。"
  fi
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv > /dev/null 2>&1 || fail "準備に失敗しました(uv を入れられませんでした)。"
fi

# --- 2) ライブラリ(2 回目以降はすぐ終わる)
echo "準備を確認しています…"
if ! uv sync > /tmp/reachy_setup.log 2>&1; then
  echo
  tail -5 /tmp/reachy_setup.log
  fail "準備に失敗しました。インターネットにつながっているか確認してください。"
fi

# --- 3) 声の設定に応じた案内(録音だけなら VOICEVOX は不要)
SOURCE="$(uv run python -c 'import json;print(json.load(open("settings.json")).get("tts",{}).get("source","synth"))' 2>/dev/null || echo synth)"
if [[ "$SOURCE" != "recorded" && ! -x tools/voicevox/macos-arm64/run ]]; then
  echo
  echo "!! 合成の声を使う設定ですが、VOICEVOX が入っていません。"
  echo "   ターミナルで experiment フォルダの ./setup_voicevox.sh を実行するか、"
  echo "   設定 > 声 で「りゅうさん(録音)」を選んでください。"
  echo
fi

# --- 4) 起動して、つながったらブラウザを開く
IP="$(ipconfig getifaddr en0 2>/dev/null || true)"
echo
echo "操作画面を開きます: http://localhost:${PORT}"
[[ -n "$IP" ]] && echo "iPad から使うとき:   http://${IP}:${PORT}"
echo
echo "終わるときは、このウィンドウで Control + C を押してください。"
echo "--------------------------------------------"
echo

(
  for _ in $(seq 1 60); do
    if curl -s -m 1 "http://127.0.0.1:${PORT}/api/state" > /dev/null 2>&1; then
      open "http://localhost:${PORT}"
      exit 0
    fi
    sleep 1
  done
) &

exec uv run uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --no-access-log
