"""Reachy Mini 挨拶アプリ(ターミナル対話型)。

1 → 「こんにちは」、2 → 「さようなら」をロボット本体のスピーカーで発話し、
発話中はアンテナを揺らす。q または Ctrl+C で終了(スリープ姿勢に戻す)。

ロボット内デーモンのREST API(デフォルト http://reachy-mini.local:8000)を直接叩く。
接続先は環境変数 REACHY_MINI_URL で変更できる。
"""

import os
import sys
import time
from pathlib import Path

import requests

BASE_URL = os.environ.get("REACHY_MINI_URL", "http://reachy-mini.local:8000")
SOUNDS_DIR = Path(__file__).parent / "sounds"
TIMEOUT = (3.0, 10.0)  # (接続, 読み取り) 秒

NEUTRAL_HEAD = {"x": 0.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}

# 発話中のアンテナ揺らし: (アンテナ [右, 左] rad, 頭のロール rad, 所要秒)
WIGGLE_KEYFRAMES = [
    ([0.45, -0.45], 0.09, 0.25),
    ([-0.45, 0.45], -0.09, 0.25),
    ([0.45, -0.45], 0.09, 0.25),
    ([-0.45, 0.45], -0.09, 0.25),
    ([0.0, 0.0], 0.0, 0.3),
]

MENU = "1: こんにちは / 2: さようなら / q: 終了"


class ReachyClient:
    """デーモンREST APIの薄いラッパー。"""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")

    def _get(self, path: str):
        r = requests.get(f"{self.base}{path}", timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, **kwargs):
        r = requests.post(f"{self.base}{path}", timeout=TIMEOUT, **kwargs)
        r.raise_for_status()
        return r.json()

    def daemon_status(self) -> dict:
        return self._get("/api/daemon/status")

    def motors_mode(self) -> str:
        return self._get("/api/motors/status").get("mode", "")

    def enable_motors(self) -> None:
        self._post("/api/motors/set_mode/enabled")

    def wake_up(self) -> None:
        self._post("/api/move/play/wake_up")
        self.wait_moves_done(timeout=15.0)

    def goto_sleep(self) -> None:
        self._post("/api/move/play/goto_sleep")
        self.wait_moves_done(timeout=10.0)

    def wait_moves_done(self, timeout: float) -> None:
        """実行中のムーブがなくなるまでポーリングする。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._get("/api/move/running"):
                return
            time.sleep(0.2)

    def upload_sound(self, path: Path) -> str:
        """WAVをロボットへアップロードし、ロボット上の保存パスを返す。

        保存先はロボットの /tmp 配下のためロボット再起動で消える。
        アプリ起動のたびにアップロードし直す前提。
        """
        with open(path, "rb") as f:
            res = self._post(
                "/api/media/sounds/upload",
                files={"file": (path.name, f, "audio/wav")},
            )
        return res["path"]

    def play_sound(self, robot_path: str) -> None:
        """ロボットのスピーカーで再生する(非同期・即座に返る)。"""
        self._post("/api/media/play_sound", json={"file": robot_path})

    def goto(self, antennas=None, head=None, duration: float = 0.5) -> None:
        # body_yaw はデフォルト 0.0 だと毎回胴体が再センタリングされるため null を明示
        self._post(
            "/api/move/goto",
            json={
                "head_pose": head,
                "antennas": antennas,
                "body_yaw": None,
                "duration": duration,
                "interpolation": "minjerk",
            },
        )


def greet(client: ReachyClient, robot_path: str) -> None:
    """発話(非同期)と同時にアンテナ揺らしアニメーションを実行する。"""
    client.play_sound(robot_path)
    for antennas, roll, duration in WIGGLE_KEYFRAMES:
        start = time.monotonic()
        client.goto(antennas=antennas, head={**NEUTRAL_HEAD, "roll": roll}, duration=duration)
        # gotoが同期でも非同期でも次のキーフレームまで duration 秒空ける
        time.sleep(max(0.0, duration - (time.monotonic() - start)))


def connect() -> ReachyClient | None:
    print(f"Reachy Mini に接続中... ({BASE_URL})")
    client = ReachyClient(BASE_URL)
    try:
        status = client.daemon_status()
    except requests.exceptions.RequestException as e:
        print("Reachy Mini に接続できませんでした。")
        print("  - ロボットの電源とLANケーブルの接続を確認してね")
        print(f"  - 接続先: {BASE_URL}(環境変数 REACHY_MINI_URL で変更可)")
        print(f"  詳細: {e}")
        return None
    if status.get("state") != "running":
        print(f"デーモンが動作していません(state: {status.get('state')})")
        return None
    print(f"接続OK: {status.get('robot_name', 'reachy_mini')}")
    return client


def setup(client: ReachyClient) -> dict[str, str] | None:
    """ウェイクアップと音声アップロードを行い、{入力キー: ロボット上のパス} を返す。"""
    # ロボットを起こす前にローカルのWAVを確認する(無駄な起床動作を防ぐ)
    local_files: dict[str, Path] = {}
    for key, filename in (("1", "hello.wav"), ("2", "goodbye.wav")):
        local = SOUNDS_DIR / filename
        if not local.exists():
            print(f"{local} がありません。先に `uv run generate_voices.py` を実行してね。")
            return None
        local_files[key] = local

    if client.motors_mode() != "enabled":
        print("ウェイクアップ中...")
        # wake_up はトルクを入れないため、先にモーターを有効化する必要がある
        client.enable_motors()
        client.wake_up()

    print("音声ファイルをアップロード中...")
    return {key: client.upload_sound(path) for key, path in local_files.items()}


def interact(client: ReachyClient, sound_paths: dict[str, str]) -> None:
    print()
    print(f"準備完了! {MENU}")
    while True:
        try:
            choice = input("> ").strip()
        except EOFError:
            return
        if choice == "q":
            return
        if choice in sound_paths:
            try:
                greet(client, sound_paths[choice])
            except requests.exceptions.RequestException as e:
                print(f"通信エラー(入力待ちに戻ります): {e}")
        else:
            print(f"再度入力してね({MENU})")


def shutdown(client: ReachyClient) -> None:
    print("スリープ姿勢に戻して終了します...")
    try:
        client.goto(antennas=[0.0, 0.0], head=NEUTRAL_HEAD, duration=1.0)
        time.sleep(1.0)
        client.goto_sleep()
    except requests.exceptions.RequestException:
        print("(ロボットと通信できないため、スリープ移行はスキップしました)")


def main() -> int:
    client = connect()
    if client is None:
        return 1
    woke = False
    try:
        sound_paths = setup(client)
        if sound_paths is None:
            return 1
        woke = True
        interact(client, sound_paths)
    except KeyboardInterrupt:
        print()
    except requests.exceptions.RequestException as e:
        print(f"通信エラーが発生しました: {e}")
        return 1
    finally:
        # 起こしていない(セットアップ前に失敗した)場合は寝かせ直さない
        if woke:
            shutdown(client)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # 接続試行中・終了処理中のCtrl+Cもトレースバックなしで終了する
        print()
        sys.exit(130)
