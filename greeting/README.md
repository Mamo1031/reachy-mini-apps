# greeting — Reachy Mini 挨拶アプリ

ターミナルから Reachy Mini に挨拶させる最小アプリ。

- `1` → 「こんにちは」
- `2` → 「さようなら」
- その他 → 再入力を促す
- `q` / Ctrl+C → スリープ姿勢に戻して終了

発話はロボット本体のスピーカーから。発話中はアンテナが揺れる。

## 仕組み

ロボット内デーモンのREST API(`http://reachy-mini.local:8000`)を直接叩く。
公式SDKは使わない — 無線版をEthernet接続で使う場合、SDKのメディア経路(WebRTC)が
Wi-Fi前提の構造で不安定になるため。依存は `requests` のみ。

音声はmacOS標準の `say`(Kyoko)で事前生成したWAVを、起動時にロボットへ
アップロードして使う(ロボット側の保存先が `/tmp` のため毎回アップロード)。

## 使い方

```bash
cd greeting

# 初回のみ: 音声ファイルを生成
uv run generate_voices.py

# 実行(ロボットの電源とLAN接続が必要)
uv run main.py
```

接続先を変える場合:

```bash
REACHY_MINI_URL=http://192.168.x.x:8000 uv run main.py
```

## 注意

- ダッシュボード(Reachy Mini Control)から公式アプリを実行中の場合は、
  ロボットの取り合いになるため先に停止しておくこと。
- `reachy-mini.local` が解決できない場合はルーターのDHCP一覧などでIPを調べ、
  `REACHY_MINI_URL` で指定する。
