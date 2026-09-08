"""ロボット無しで操作画面を試すデモサーバー(偽デーモンを内蔵)。

使い方: cd experiment && uv run --all-groups python tools/demo_server.py [--port 8082] [--fake-tts]
- ロボット API は tests/fake_daemon.py の偽物に接続する(実機の癖を再現)。
- 音声は VOICEVOX が動いていれば本物、無ければ無音 WAV(--fake-tts で強制)。
- 設定・キャッシュ・ログは cache/demo/ 以下に書く(本番の settings.json を汚さない)。
- 故障注入: POST /demo/outage?seconds=5(通信断)、POST /demo/face?on=1(顔検出)、POST /demo/reboot(音声消失)
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import Request
from fastapi.routing import APIRoute

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import main  # noqa: E402
from app.config import DATA_DIR  # noqa: E402
from tests.fake_daemon import FakeDaemon, make_app  # noqa: E402
from tests.test_audio import FakeTTS  # noqa: E402


def build(port: int, fake_tts: bool) -> None:
    fake = FakeDaemon()
    data_dir = DATA_DIR / "cache" / "demo"
    data_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        tts = FakeTTS() if fake_tts else None
        await main.build_state(data_dir=data_dir, robot_transport=httpx.ASGITransport(app=make_app(fake)), tts=tts, monitor_interval=0.5)
        main.state.settings.motion.idle.interval_s = 6.0
        yield
        await main.shutdown_state()

    main.app.router.lifespan_context = lifespan

    async def outage(request: Request):
        seconds = float(request.query_params.get("seconds", "5"))
        fake.fail_until = time.monotonic() + seconds
        return {"outage_s": seconds}

    async def face(request: Request):
        fake.face_detected = request.query_params.get("on", "1") == "1"
        return {"face": fake.face_detected}

    async def reboot(request: Request):
        fake.sounds.clear()
        fake.motor_mode = "disabled"
        fake.fail_until = time.monotonic() + 3.0
        return {"rebooted": True}

    async def state(request: Request):
        return {
            "motor_mode": fake.motor_mode,
            "sounds": len(fake.sounds),
            "played": [n for _, n in fake.played][-10:],
            "targets": len(fake.targets),
            "last_target": fake.targets[-1][1] if fake.targets else None,
            "tracking": [fake.tracking_enabled, fake.tracking_weight],
            "head": fake.head,
        }

    # 静的ファイルのマウント("/")より前に入れないと届かない
    for path, fn, methods in (("/demo/outage", outage, ["POST"]), ("/demo/face", face, ["POST"]), ("/demo/reboot", reboot, ["POST"]), ("/demo/state", state, ["GET"])):
        main.app.router.routes.insert(0, APIRoute(path, fn, methods=methods))

    print(f"デモサーバー: http://127.0.0.1:{port}  (偽ロボット / {'偽音声' if fake_tts else 'VOICEVOX があれば本物の音声'})")
    uvicorn.run(main.app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--fake-tts", action="store_true")
    a = ap.parse_args()
    build(a.port, a.fake_tts)
