"""ロボット内デーモンの REST API クライアント(httpx、keep-alive、呼び出しごとのタイムアウト)。

例外は全て RobotError の派生で、str() が日本語のユーザー向けメッセージになる。
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .pose import Pose, antennas_payload, head_payload, pose_to_matrix_flat

log = logging.getLogger(__name__)


def _looks_like_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return host in ("localhost",)


class RobotError(Exception):
    """ロボット通信エラーの基底。"""


class RobotUnreachable(RobotError):
    """接続不可・タイムアウト・名前解決失敗。"""


class RobotHttpError(RobotError):
    def __init__(self, status: int, body: str, what: str) -> None:
        super().__init__(f"{what} に失敗しました(HTTP {status}): {body[:200]}")
        self.status = status
        self.body = body


class RobotBusy(RobotError):
    """ムーブ実行中のため set_target が無視された。"""


@dataclass
class DaemonStatus:
    state: str
    ready: bool
    motor_mode: str | None
    last_alive: float | None
    error: str | None
    version: str | None
    face_detected: bool
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "DaemonStatus":
        bs = d.get("backend_status") or {}
        ft = d.get("face_target") or {}
        return cls(
            state=str(d.get("state", "")),
            ready=bool(bs.get("ready", False)),
            motor_mode=bs.get("motor_control_mode"),
            last_alive=bs.get("last_alive"),
            error=d.get("error") or bs.get("error"),
            version=d.get("version"),
            face_detected=bool(ft.get("detected", False)),
            raw=d,
        )


class TargetStream:
    """デーモンの WebSocket(/ws/sdk)へ `set_full_target` を送りっぱなしにする経路。

    HTTP は 1 フレームごとに新しい TCP 接続(往復 2 回)が要り、Wi-Fi の揺らぎで 30〜150 ms かかる。
    WebSocket なら応答を待たずに送れるので、送信レートが往復遅延に縛られない。
    接続が切れたら次のフレームで張り直し、張れなければ呼び出し側が HTTP に落とす。
    """

    def __init__(self, robot: "RobotClient") -> None:
        self.robot = robot
        self._ws: Any = None
        self._drain_task: asyncio.Task | None = None
        self._failed_at = 0.0
        self.sent = 0

    def _url(self) -> str:
        base = self.robot.resolved_url
        scheme = "wss" if base.startswith("https") else "ws"
        return scheme + base.split("://", 1)[1].join(["://", "/ws/sdk"])

    async def _connect(self) -> bool:
        if self._ws is not None:
            return True
        if time.monotonic() - self._failed_at < 2.0:  # 失敗直後は連続で試さない
            return False
        try:
            import websockets

            self._ws = await asyncio.wait_for(websockets.connect(self._url(), open_timeout=1.0, ping_interval=None, max_queue=4), timeout=1.5)
        except Exception as e:  # 接続不可はフォールバックで吸収
            log.info("ws stream unavailable (%s); using HTTP", type(e).__name__)
            self._failed_at = time.monotonic()
            self._ws = None
            return False
        self._drain_task = asyncio.create_task(self._drain(), name="ws-target-drain")
        log.info("ws target stream connected: %s", self._url())
        return True

    async def _drain(self) -> None:
        """デーモンが流してくる状態通知を読み捨てる(読まないと送信バッファが詰まる)。"""
        ws = self._ws
        try:
            async for _ in ws:
                pass
        except Exception:
            pass
        finally:
            if self._ws is ws:
                self._ws = None

    async def send(self, pose: Pose, *, head: bool, antennas: bool) -> bool:
        if not await self._connect():
            return False
        msg = {
            "type": "set_full_target",
            "head": pose_to_matrix_flat(pose) if head else None,
            "antennas": antennas_payload(pose) if antennas else None,
            "body_yaw": None,
        }
        try:
            await asyncio.wait_for(self._ws.send(json.dumps(msg)), timeout=0.5)
        except Exception as e:
            log.warning("ws send failed (%s); reconnecting next frame", type(e).__name__)
            await self.close()
            self._failed_at = 0.0  # すぐ張り直してよい
            return False
        self.sent += 1
        return True

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if self._drain_task is not None:
            self._drain_task.cancel()
            self._drain_task = None
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=1.0)
            except Exception:
                pass


class RobotClient:
    def __init__(
        self,
        base_url: str,
        *,
        connect_timeout: float = 1.0,
        read_timeout: float = 3.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._resolved_url: str | None = None
        self._stream: TargetStream | None = None
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------ lifecycle
    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def resolved_url(self) -> str:
        """実際に接続している URL(ホスト名を IP に置き換えたもの)。"""
        return self._resolved_url or self._base_url

    def _make_client(self, url: str) -> httpx.AsyncClient:
        # 実機のデーモン(v1.10.0)は keep-alive で再利用した接続の 2 リクエスト目に応答しない
        # (成功と ReadTimeout が交互に起きる)ため、毎回新しい接続を張る。LAN では 1 接続 数十 ms。
        return httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(connect=self._connect_timeout, read=self._read_timeout, write=self._read_timeout, pool=1.0),
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=8),
            headers={"Connection": "close"},
            transport=self._transport,
        )

    async def _resolve(self) -> str:
        """`reachy-mini.local` のような名前を IP に解決した URL を返す(mDNS を毎回引くと 1 接続 +10 ms)。"""
        parts = urlsplit(self._base_url)
        host = parts.hostname or ""
        if self._transport is not None or not host or _looks_like_ip(host):
            return self._base_url
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, parts.port or 80, type=socket.SOCK_STREAM)
        except (socket.gaierror, OSError) as e:
            raise RobotUnreachable(f"ロボットの名前解決に失敗しました({host}): {e}") from e
        ipv4 = [i[4][0] for i in infos if i[0] == socket.AF_INET]
        addr = ipv4[0] if ipv4 else infos[0][4][0]
        if ":" in addr:  # IPv6
            addr = f"[{addr}]"
        netloc = f"{addr}:{parts.port}" if parts.port else addr
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._resolved_url = await self._resolve()
            self._client = self._make_client(self._resolved_url)
        return self._client

    async def _drop_client(self) -> None:
        """接続失敗時: 次回は名前解決からやり直す(再起動で IP が変わった場合に備える)。"""
        old, self._client, self._resolved_url = self._client, None, None
        if self._stream is not None:
            await self._stream.close()
        if old is not None:
            await old.aclose()

    async def set_base_url(self, url: str, connect_timeout: float | None = None, read_timeout: float | None = None) -> None:
        self._base_url = url.rstrip("/")
        if connect_timeout is not None:
            self._connect_timeout = connect_timeout
        if read_timeout is not None:
            self._read_timeout = read_timeout
        await self._drop_client()

    async def aclose(self) -> None:
        await self._drop_client()

    # ------------------------------------------------------------ transport
    async def _request(self, method: str, path: str, what: str, *, timeout: float | None = None, **kw: Any) -> Any:
        if timeout is not None:
            kw["timeout"] = httpx.Timeout(connect=self._connect_timeout, read=timeout, write=timeout, pool=1.0)
        try:
            client = await self._get_client()
            r = await client.request(method, path, **kw)
        except httpx.ConnectError as e:  # 接続不可: 次回は名前解決からやり直す
            await self._drop_client()
            raise RobotUnreachable(f"ロボットに接続できません({self._base_url}): {type(e).__name__}") from e
        except httpx.TransportError as e:  # タイムアウト・切断
            raise RobotUnreachable(f"ロボットに接続できません({self._base_url}): {type(e).__name__}") from e
        if r.status_code >= 400:
            raise RobotHttpError(r.status_code, r.text, what)
        if not r.content:
            return None
        try:
            return r.json()
        except ValueError:
            return r.text

    async def _get(self, path: str, what: str, **kw: Any) -> Any:
        return await self._request("GET", path, what, **kw)

    async def _post(self, path: str, what: str, **kw: Any) -> Any:
        return await self._request("POST", path, what, **kw)

    # ------------------------------------------------------------ status / motors
    async def daemon_status(self, timeout: float | None = None) -> DaemonStatus:
        d = await self._get("/api/daemon/status", "状態取得", timeout=timeout)
        if not isinstance(d, dict):
            raise RobotHttpError(200, str(d), "状態取得")
        return DaemonStatus.from_json(d)

    async def motor_mode(self) -> str:
        d = await self._get("/api/motors/status", "モーター状態取得")
        return str(d.get("mode", ""))

    async def enable_motors(self) -> None:
        await self._post("/api/motors/set_mode/enabled", "モーター有効化")

    async def disable_motors(self) -> None:
        await self._post("/api/motors/set_mode/disabled", "モーター無効化")

    async def present_pose(self, timeout: float | None = None) -> Pose:
        d = await self._get("/api/state/full", "現在姿勢取得", timeout=timeout)
        hp = d.get("head_pose")
        ant = d.get("antennas_position")
        if not isinstance(hp, dict) or not ant or len(ant) < 2:
            raise RobotError("現在姿勢を取得できませんでした(head_pose / antennas_position が空)")
        return Pose(
            roll=float(hp["roll"]),
            pitch=float(hp["pitch"]),
            yaw=float(hp["yaw"]),
            x=float(hp["x"]),
            y=float(hp["y"]),
            z=float(hp["z"]),
            ant_r=float(ant[0]),
            ant_l=float(ant[1]),
        )

    # ------------------------------------------------------------ moves
    async def running_moves(self) -> list[str]:
        d = await self._get("/api/move/running", "実行中ムーブ取得")
        if not isinstance(d, list):
            return []
        return [m["uuid"] if isinstance(m, dict) else str(m) for m in d]

    async def stop_move(self, uuid: str) -> None:
        try:
            await self._post("/api/move/stop", "ムーブ停止", json={"uuid": uuid})
        except RobotHttpError as e:
            if e.status in (404, 500):  # 既に終了している uuid はデーモンが 500 を返す
                return
            raise

    async def clear_moves(self) -> int:
        uuids = await self.running_moves()
        for u in uuids:
            await self.stop_move(u)
        return len(uuids)

    async def goto(self, pose: Pose, duration: float, body_yaw: float | None = None) -> str:
        d = await self._post(
            "/api/move/goto",
            "goto",
            json={
                "head_pose": head_payload(pose),
                "antennas": antennas_payload(pose),
                "body_yaw": body_yaw,
                "duration": duration,
                "interpolation": "minjerk",
            },
        )
        return str(d.get("uuid", "")) if isinstance(d, dict) else ""

    async def wait_moves_done(self, timeout: float, poll: float = 0.1) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not await self.running_moves():
                return True
            await asyncio.sleep(poll)
        return not await self.running_moves()

    async def set_target(self, pose: Pose, *, head: bool = True, antennas: bool = True) -> None:
        body = {
            "target_head_pose": head_payload(pose) if head else None,
            "target_antennas": antennas_payload(pose) if antennas else None,
            "target_body_yaw": None,
        }
        d = await self._post("/api/move/set_target", "set_target", json=body, timeout=0.5)
        if isinstance(d, dict) and d.get("status") == "ignored":
            raise RobotBusy("ムーブ実行中のため目標を送れませんでした")

    async def stream_target(self, pose: Pose, *, head: bool = True, antennas: bool = True) -> str:
        """軌道のフレーム送信用。WebSocket(応答待ちなし)を優先し、使えなければ HTTP set_target に落とす。

        戻り値: 使った経路 "ws" / "http"。WebSocket は応答が無いので「ムーブ実行中で無視された」ことは
        分からない — ストリーミング開始前に HTTP の set_target で確認しておくこと。
        """
        if self._transport is None:  # テスト用の ASGI トランスポートでは WebSocket は使えない
            stream = self._stream or TargetStream(self)
            self._stream = stream
            if await stream.send(pose, head=head, antennas=antennas):
                return "ws"
        await self.set_target(pose, head=head, antennas=antennas)
        return "http"

    async def close_stream(self) -> None:
        if self._stream is not None:
            await self._stream.close()

    # ------------------------------------------------------------ tracking / wobbling
    async def set_tracking(self, enabled: bool, weight: float = 1.0) -> None:
        if enabled:
            await self._post("/api/media/tracking/enable", "顔追跡", json={"weight": max(0.0, min(1.0, weight))})
        else:
            await self._post("/api/media/tracking/disable", "顔追跡停止")

    async def set_wobbling(self, enabled: bool) -> None:
        await self._post("/api/media/wobbling/enable" if enabled else "/api/media/wobbling/disable", "頭揺れ設定")

    # ------------------------------------------------------------ audio
    async def list_sounds(self) -> set[str]:
        """ロボット上の音声ファイル名(basename)。実機は {"files": [...]} だが、キー名に依存しない。"""
        d = await self._get("/api/media/sounds", "音声一覧取得")
        names: set[str] = set()
        if isinstance(d, dict):
            for v in d.values():
                if isinstance(v, list):
                    names.update(str(f).rsplit("/", 1)[-1] for f in v)
        elif isinstance(d, list):
            names.update(str(f).rsplit("/", 1)[-1] for f in d)
        return names

    async def upload_sound(self, name: str, wav: bytes) -> None:
        await self._post("/api/media/sounds/upload", f"音声アップロード({name})", files={"file": (name, wav, "audio/wav")}, timeout=20.0)

    async def play_sound(self, name: str) -> None:
        await self._post("/api/media/play_sound", "音声再生", json={"file": name}, timeout=2.0)

    async def stop_sound(self) -> None:
        await self._post("/api/media/stop_sound", "音声停止", timeout=2.0)

    async def get_volume(self) -> int:
        d = await self._get("/api/volume/current", "音量取得")
        return int(d.get("volume", 0))

    async def set_volume(self, volume: int) -> None:
        await self._post("/api/volume/set", "音量設定", json={"volume": int(max(0, min(100, volume)))}, timeout=10.0)

    async def goto_sleep(self) -> str:
        d = await self._post("/api/move/play/goto_sleep", "スリープ", timeout=5.0)
        return str(d.get("uuid", "")) if isinstance(d, dict) else ""
