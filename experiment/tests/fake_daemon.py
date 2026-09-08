"""テスト用の偽デーモン(実機の REST API の挙動を必要な範囲で再現する)。

再現している実機の癖:
- goto は非同期で uuid を返す。実行中に別の goto を送ると黙って捨てられる(uuid は返る)。
- set_target はムーブ実行中 {"status": "ignored"} を返す。
- /api/move/stop は未知の uuid に 500 を返す。
- daemon status の last_alive は制御ループが動いている限り進む。
"""

from __future__ import annotations

import time
import uuid as uuidlib
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse


@dataclass
class FakeMove:
    uuid: str
    ends_at: float
    kind: str


@dataclass
class FakeDaemon:
    motor_mode: str = "disabled"
    head: dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": -0.045, "roll": 0.0, "pitch": 0.5, "yaw": 0.0})
    antennas: list[float] = field(default_factory=lambda: [-3.05, 3.05])
    body_yaw: float = 0.0
    sounds: set[str] = field(default_factory=set)
    volume: int = 80
    tracking_enabled: bool = False
    tracking_weight: float = 1.0
    wobbling: bool = False
    face_detected: bool = False
    moves: dict[str, FakeMove] = field(default_factory=dict)
    # 記録
    targets: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    played: list[tuple[float, str]] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    # 故障注入
    fail_until: float = 0.0  # この時刻まで全リクエストに 503
    state: str = "running"
    error: str | None = None
    freeze_alive: bool = False
    _alive: float = field(default_factory=time.time)
    version: str = "fake-1.10.0"

    def running(self) -> list[str]:
        now = time.monotonic()
        for k in [k for k, m in self.moves.items() if m.ends_at <= now]:
            del self.moves[k]
        return list(self.moves)

    def last_alive(self) -> float:
        if not self.freeze_alive:
            self._alive = time.time()
        return self._alive


def make_app(fd: FakeDaemon) -> FastAPI:
    app = FastAPI()

    @app.middleware("http")
    async def failure_injection(request: Request, call_next):
        fd.calls.append(f"{request.method} {request.url.path}")
        if time.monotonic() < fd.fail_until:
            return JSONResponse({"detail": "injected failure"}, status_code=503)
        return await call_next(request)

    @app.get("/api/daemon/status")
    def status():
        return {
            "type": "daemon_status",
            "robot_name": "fake",
            "state": fd.state,
            "wireless_version": True,
            "backend_status": {
                "ready": fd.state == "running",
                "motor_control_mode": fd.motor_mode,
                "last_alive": fd.last_alive(),
                "control_loop_stats": {},
                "error": fd.error,
            },
            "error": fd.error,
            "version": fd.version,
            "face_target": {"detected": fd.face_detected, "x": 0.0 if fd.face_detected else None, "y": None, "roll": None, "ts": None},
        }

    @app.get("/api/motors/status")
    def motors_status():
        return {"mode": fd.motor_mode}

    @app.post("/api/motors/set_mode/{mode}")
    def set_mode(mode: str):
        fd.motor_mode = mode
        return {"status": "ok"}

    @app.get("/api/state/full")
    def state_full():
        return {"control_mode": fd.motor_mode, "head_pose": dict(fd.head), "body_yaw": fd.body_yaw, "antennas_position": list(fd.antennas)}

    @app.post("/api/move/goto")
    def goto(body: dict):
        u = str(uuidlib.uuid4())
        if fd.running():
            return {"uuid": u}  # 黙って捨てる
        fd.moves[u] = FakeMove(uuid=u, ends_at=time.monotonic() + float(body.get("duration", 0.0)), kind="goto")
        if body.get("head_pose"):
            fd.head = dict(body["head_pose"])
        if body.get("antennas"):
            fd.antennas = list(body["antennas"])
        if body.get("body_yaw") is not None:
            fd.body_yaw = float(body["body_yaw"])
        return {"uuid": u}

    @app.post("/api/move/play/goto_sleep")
    def goto_sleep():
        u = str(uuidlib.uuid4())
        if not fd.running():
            fd.moves[u] = FakeMove(uuid=u, ends_at=time.monotonic() + 2.0, kind="sleep")
            fd.head = {"x": 0.0, "y": 0.0, "z": -0.045, "roll": 0.0, "pitch": 0.5, "yaw": 0.0}
            fd.antennas = [-3.05, 3.05]
        return {"uuid": u}

    @app.post("/api/move/play/wake_up")
    def wake_up():
        u = str(uuidlib.uuid4())
        if not fd.running():
            fd.moves[u] = FakeMove(uuid=u, ends_at=time.monotonic() + 1.0, kind="wake")
            fd.played.append((time.monotonic(), "wake_up.wav"))
        return {"uuid": u}

    @app.get("/api/move/running")
    def running():
        return [{"uuid": u} for u in fd.running()]

    @app.post("/api/move/stop")
    def stop(body: dict):
        u = body.get("uuid")
        if u not in fd.moves:
            return JSONResponse({"detail": "KeyError"}, status_code=500)
        del fd.moves[u]
        return {"status": "ok"}

    @app.post("/api/move/set_target")
    def set_target(body: dict):
        if fd.running():
            return {"status": "ignored", "reason": "move_running"}
        fd.targets.append((time.monotonic(), body))
        if body.get("target_head_pose"):
            fd.head = dict(body["target_head_pose"])
        if body.get("target_antennas"):
            fd.antennas = list(body["target_antennas"])
        return {"status": "ok"}

    @app.post("/api/media/tracking/enable")
    def tracking_enable(body: dict | None = None):
        fd.tracking_enabled = True
        fd.tracking_weight = float((body or {}).get("weight", 1.0))
        return {"status": "ok"}

    @app.post("/api/media/tracking/disable")
    def tracking_disable():
        fd.tracking_enabled = False
        return {"status": "ok"}

    @app.get("/api/media/tracking/face")
    def tracking_face():
        return {"status": "ok", "face_target": {"detected": fd.face_detected}}

    @app.post("/api/media/wobbling/enable")
    def wob_on():
        fd.wobbling = True
        return {"status": "ok"}

    @app.post("/api/media/wobbling/disable")
    def wob_off():
        fd.wobbling = False
        return {"status": "ok"}

    @app.get("/api/media/sounds")
    def sounds():
        return {"files": sorted(fd.sounds)}

    @app.post("/api/media/sounds/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        if not data:
            return JSONResponse({"detail": "empty"}, status_code=400)
        fd.sounds.add(file.filename)
        return {"status": "ok", "path": f"/tmp/reachy_mini_sounds/{file.filename}"}

    @app.post("/api/media/play_sound")
    def play_sound(body: dict):
        name = str(body.get("file", ""))
        if name.rsplit("/", 1)[-1] not in fd.sounds and not name.startswith("/"):
            return JSONResponse({"detail": "not found"}, status_code=404)
        fd.played.append((time.monotonic(), name))
        return {"status": "ok"}

    @app.post("/api/media/stop_sound")
    def stop_sound():
        fd.played.append((time.monotonic(), "<stop>"))
        return {"status": "ok"}

    @app.get("/api/volume/current")
    def vol():
        return {"volume": fd.volume, "platform": "fake", "device": "fake"}

    @app.post("/api/volume/set")
    def vol_set(body: dict):
        fd.volume = int(body["volume"])
        fd.played.append((time.monotonic(), "<test-sound>"))
        return {"volume": fd.volume, "platform": "fake", "device": "fake"}

    return app
