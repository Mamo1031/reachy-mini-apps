"""FastAPI サーバー: 操作画面の配信、UI 用 REST、SSE、起動時プリフライト、復旧手順。

起動: cd experiment && ./run.sh   (http://<MacのIP>:8080)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .audio import AudioError, AudioStore
from .config import (
    CACHE_DIR,
    DATA_DIR,
    GESTURES_PATH,
    LOGS_DIR,
    MOVES_DIR,
    PHRASES_PATH,
    SETTINGS_PATH,
    DictEntry,
    Gestures,
    Phrases,
    Settings,
    VoiceParams,
    default_gestures,
    default_phrases,
    load_or_create,
    save_atomic,
    validate_phrases,
)
from .events import EventBus, sse_stream
from .gestures import GestureError, GestureLibrary
from .monitor import ConnectionMonitor
from .performer import PerformError, Performer
from .player import TrajectoryPlayer
from .pose import DEG, NEUTRAL, antenna_distance, head_distance
from .robot import RobotClient, RobotError
from .session import SessionError, SessionManager
from .tts.base import TTSBackend, TTSError, hiragana_to_katakana
from .tts.voicevox import VoicevoxBackend

APP_VERSION = "0.1.0"
STATIC_DIR = Path(__file__).resolve().parent / "static"
PREFLIGHT_STEPS = ["voicevox", "robot", "motors", "synth", "upload", "tracking", "volume"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # 30 Hz の set_target で INFO ログが溢れるのを防ぐ
log = logging.getLogger("experiment")


# ================================================================ state


class AppState:
    def __init__(self) -> None:
        self.settings: Settings = Settings()
        self.phrases: Phrases = default_phrases()
        self.gestures: Gestures = default_gestures()
        self.warnings: list[str] = []
        self.bus = EventBus()
        self.robot: RobotClient
        self.monitor: ConnectionMonitor
        self.tts: TTSBackend
        self.audio: AudioStore
        self.gesture_lib: GestureLibrary
        self.player: TrajectoryPlayer
        self.session: SessionManager
        self.performer: Performer
        self.preflight: dict[str, dict[str, Any]] = {s: {"status": "pending", "message": ""} for s in PREFLIGHT_STEPS}
        self.preflight_task: asyncio.Task | None = None
        self.prewarm_task: asyncio.Task | None = None
        self.volume: int | None = None
        self.resting = False
        self.started_at = time.time()


state = AppState()


def make_tts(settings: Settings) -> TTSBackend:
    if settings.tts.backend == "voicevox":
        return VoicevoxBackend(settings.tts.voicevox, DATA_DIR, CACHE_DIR)
    raise TTSError(f"未知の TTS バックエンド: {settings.tts.backend}")


def step(name: str, status: str, message: str = "", **extra: Any) -> None:
    state.preflight[name] = {"status": status, "message": message, **extra}
    state.bus.publish("preflight", step=name, **state.preflight[name])


def set_resting(value: bool) -> None:
    if state.resting != value:
        state.resting = value
        state.bus.publish("resting", resting=value)


# ================================================================ recovery / preflight


async def recover(*, wake: bool | None = None) -> None:
    """接続(再)確立時の復旧手順。冪等。プリフライトの motors / upload / tracking を兼ねる。

    wake=None: 監視ループからの自動復旧。休止中ならモーターは触らない(勝手に起き上がらない)。
    wake=True: 「起こす」ボタン。休止を解除してニュートラルへ。
    各ステップは独立に失敗を記録して続行する。モーターのステップだけは失敗を復旧失敗として扱う。
    """
    robot, audio, perf = state.robot, state.audio, state.performer
    do_wake = wake if wake is not None else not state.resting
    errors: list[str] = []
    async with perf.suspended():
        # --- 1) モーター / 姿勢
        if do_wake:
            step("motors", "running", "モーターを確認中")
            try:
                if await robot.motor_mode() != "enabled":
                    await robot.enable_motors()
                    await asyncio.sleep(0.3)
                pose = await robot.present_pose()
                if head_distance(pose, NEUTRAL) > 10 * DEG or antenna_distance(pose, NEUTRAL) > 60 * DEG:
                    step("motors", "running", "ニュートラル姿勢へ移動中")
                    await robot.clear_moves()
                    # 追跡が weight 1.0 のままだと goto の頭部が無視されるので、先に止める
                    with contextlib.suppress(RobotError):
                        await robot.set_tracking(True, 0.0)
                    await robot.goto(NEUTRAL, 2.0, body_yaw=0.0)
                    if not await robot.wait_moves_done(4.0):
                        state.bus.toast("warn", "ニュートラルへの移動が時間内に終わりませんでした")
                else:
                    await robot.clear_moves()
                set_resting(False)
                step("motors", "ok", "モーター有効・ニュートラル")
            except RobotError as e:
                step("motors", "fail", str(e))
                state.session.row("recovery", result="error", detail=f"motors: {e}")
                raise
        else:
            step("motors", "ok", "休止中(モーター OFF のまま)")
        # --- 2) 音声
        step("upload", "running", "ロボット上の音声を確認中")
        try:
            audio.forget_uploads()
            n = await audio.reupload_missing(progress=lambda i, t, txt: step("upload", "running", f"音声を再アップロード中 {i}/{t}", progress=[i, t]))
            step("upload", "ok", f"音声 {len(audio.wanted)} 件を確認({n} 件を再アップロード)")
        except (RobotError, TTSError, AudioError) as e:
            errors.append(f"音声: {e}")
            step("upload", "fail", str(e))
        # --- 3) 追跡 / 頭揺れ
        step("tracking", "running", "顔追跡・頭揺れ設定を適用中")
        try:
            if do_wake:
                await perf.apply_tracking()
            else:
                await robot.set_tracking(False)
            step("tracking", "ok", "顔追跡 " + ("ON" if (state.settings.motion.tracking_enabled and do_wake and not perf.paused) else "OFF"))
        except RobotError as e:
            errors.append(f"顔追跡: {e}")
            step("tracking", "fail", str(e))
        try:
            await perf.apply_wobbling()
        except RobotError as e:
            errors.append(f"頭揺れ: {e}")
        await perf.on_reconnected()
    state.session.row("recovery", result="ok" if not errors else "partial", detail="; ".join(errors))
    if errors:
        state.bus.toast("warn", "復旧の一部に失敗しました: " + "; ".join(errors))


async def preflight() -> None:
    """起動時(と「再試行」時)の準備。失敗しても UI に理由を出して待つ。"""
    for s in PREFLIGHT_STEPS:
        if state.preflight[s]["status"] != "ok":
            step(s, "pending")
    # 1) VOICEVOX
    step("voicevox", "running", "VOICEVOX を確認中")
    try:
        await state.tts.ensure_ready(progress=lambda m: step("voicevox", "running", m))
        for e in state.settings.tts.user_dict:
            try:
                await state.tts.register_pronunciation(e.surface, e.pronunciation, e.accent_type)
            except TTSError as ex:
                log.warning("user dict %s: %s", e.surface, ex)
        step("voicevox", "ok", "VOICEVOX 準備完了")
    except TTSError as e:
        step("voicevox", "fail", str(e))
    # 2) ロボット接続(監視ループが接続すると復旧手順 = motors/upload/tracking が走る)
    step("robot", "running", f"ロボットに接続中 {state.robot.base_url}")
    tries = 0
    state.monitor.poll_now()
    while not state.monitor.is_connected():
        tries += 1
        step("robot", "running", f"ロボット接続待ち({state.robot.base_url})… {state.monitor.snapshot.reason or ''} [{tries}]")
        await asyncio.sleep(1.0)
    step("robot", "ok", f"接続 OK(daemon {state.monitor.snapshot.version or '?'})")
    # 3) 音声の事前合成(名前を含まないもの)
    if state.preflight["voicevox"]["status"] == "ok":
        await prewarm_all(include_child=False)
    else:
        step("synth", "fail", "VOICEVOX が使えないため音声を準備できません")
    # 4) 音量
    try:
        state.volume = await state.robot.get_volume()
        step("volume", "ok", f"音量 {state.volume}")
    except RobotError as e:
        step("volume", "fail", str(e))


async def prewarm_all(*, include_child: bool) -> list[str]:
    """台本の音声をまとめて合成・転送し、進捗を preflight の synth ステップに出す。失敗一覧を返す。"""
    step("synth", "running", "音声を準備中")
    texts = state.performer.texts_for_prewarm(include_child=include_child)
    try:
        failures = await state.audio.prewarm(texts, progress=lambda i, t, txt: step("synth", "running", f"音声を準備中 {i}/{t}: {txt[:18]}…", progress=[i, t]))
    except RobotError as e:
        step("synth", "fail", str(e))
        return [str(e)]
    if failures:
        step("synth", "fail", f"{len(failures)} 件の音声を作れませんでした: " + failures[0])
        for f in failures[:3]:
            state.bus.toast("error", f"音声を作れませんでした: {f}")
    else:
        step("synth", "ok", f"音声 {len(texts)} 件を準備済み")
        step("upload", "ok", f"音声 {len(state.audio.wanted)} 件をロボットへ転送済み")
    return failures


def start_prewarm_background() -> None:
    """声や台本が変わったあと、次のボタンで待たされないよう裏で合成し直す。"""
    if state.prewarm_task is not None and not state.prewarm_task.done():
        state.prewarm_task.cancel()
    if not state.monitor.can_control():
        return
    include_child = state.session.current is not None
    state.prewarm_task = asyncio.create_task(supervised(prewarm_all(include_child=include_child), "prewarm"), name="prewarm")


def start_preflight() -> None:
    if state.preflight_task is not None and not state.preflight_task.done():
        return
    state.preflight_task = asyncio.create_task(supervised(preflight(), "preflight"), name="preflight")


async def supervised(coro: Any, name: str) -> None:
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception as e:  # 想定外でもサーバーを落とさない
        log.exception("%s crashed", name)
        state.bus.toast("error", f"{name} で予期しないエラー: {e}")


# ================================================================ lifespan


async def build_state(
    *,
    data_dir: Path = DATA_DIR,
    robot_transport: Any = None,
    tts: TTSBackend | None = None,
    monitor_interval: float = 1.0,
) -> AppState:
    """全コンポーネントを組み立てて監視・プリフライト・アイドルを開始する(テストからは偽物を注入)。"""
    global CACHE_DIR, LOGS_DIR, SETTINGS_PATH, PHRASES_PATH, GESTURES_PATH  # noqa: PLW0603
    if data_dir != DATA_DIR:
        CACHE_DIR, LOGS_DIR = data_dir / "cache", data_dir / "logs"
        SETTINGS_PATH, PHRASES_PATH, GESTURES_PATH = data_dir / "settings.json", data_dir / "phrases.json", data_dir / "gestures.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    state.warnings = []
    state.resting = False
    state.bus = EventBus()  # 組み立てのたびに新しく(テストで直前の状態を持ち越さない)
    for path, model, default, attr in (
        (SETTINGS_PATH, Settings, Settings(), "settings"),
        (PHRASES_PATH, Phrases, default_phrases(), "phrases"),
        (GESTURES_PATH, Gestures, default_gestures(), "gestures"),
    ):
        obj, warn = load_or_create(path, model, default)
        setattr(state, attr, obj)
        if warn:
            state.warnings.append(warn)
    s = state.settings
    state.robot = RobotClient(s.robot.base_url, connect_timeout=s.robot.connect_timeout_s, read_timeout=s.robot.read_timeout_s, transport=robot_transport)
    state.tts = tts or make_tts(s)
    state.audio = AudioStore(CACHE_DIR, state.robot, state.tts, lambda: state.settings)
    state.gesture_lib = GestureLibrary(lambda: state.gestures, lambda: state.settings, MOVES_DIR)
    try:
        state.warnings.extend(state.gesture_lib.validate())
    except Exception as e:  # ジェスチャー定義が壊れていても起動はする
        state.warnings.append(f"ジェスチャー定義の検証に失敗しました: {e}")
    state.warnings.extend(validate_phrases(state.phrases, set(state.gestures.names())))
    state.player = TrajectoryPlayer(state.robot, lambda: state.settings, state.bus)
    state.session = SessionManager(LOGS_DIR, state.bus, lambda: state.settings, lambda: state.phrases, APP_VERSION)
    state.monitor = ConnectionMonitor(
        state.robot,
        state.bus,
        recover,
        interval=monitor_interval,
        resting_ref=lambda: state.resting,
        on_change=lambda st, reason: state.session.row("connection", result=st, detail=reason),
    )
    state.performer = Performer(
        state.robot,
        state.player,
        state.audio,
        lambda: state.phrases,
        state.gesture_lib,
        state.session,
        state.monitor,
        state.bus,
        lambda: state.settings,
        resting_ref=lambda: state.resting,
    )
    state.preflight = {st: {"status": "pending", "message": ""} for st in PREFLIGHT_STEPS}
    for w in state.warnings:
        log.warning(w)
    await state.monitor.start()
    start_preflight()
    state.performer.start_idle()
    log.info("experiment controller %s started", APP_VERSION)
    return state


async def shutdown_state() -> None:
    await state.performer.stop_idle()
    for t in (state.preflight_task, state.prewarm_task):
        if t is not None:
            t.cancel()
    await state.monitor.stop()
    with contextlib.suppress(Exception):
        state.session.end()
    await state.robot.aclose()
    await state.tts.aclose()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    await build_state()
    try:
        yield
    finally:
        await shutdown_state()


app = FastAPI(title="Reachy Mini 実験コントローラ", version=APP_VERSION, lifespan=lifespan)


# ================================================================ errors


@app.exception_handler(PerformError)
async def _perform_error(_: Request, e: PerformError):
    state.bus.toast("error", str(e))
    return JSONResponse({"error": str(e)}, status_code=e.status)


@app.exception_handler(SessionError)
async def _session_error(_: Request, e: SessionError):
    return JSONResponse({"error": str(e)}, status_code=400)


@app.exception_handler(RobotError)
async def _robot_error(_: Request, e: RobotError):
    state.bus.toast("error", str(e))
    return JSONResponse({"error": str(e)}, status_code=503)


@app.exception_handler(TTSError)
async def _tts_error(_: Request, e: TTSError):
    state.bus.toast("error", str(e))
    return JSONResponse({"error": str(e)}, status_code=503)


@app.exception_handler(AudioError)
async def _audio_error(_: Request, e: AudioError):
    state.bus.toast("error", str(e))
    return JSONResponse({"error": str(e)}, status_code=503)


@app.exception_handler(GestureError)
async def _gesture_error(_: Request, e: GestureError):
    return JSONResponse({"error": str(e)}, status_code=400)


@app.exception_handler(ValidationError)
async def _validation_error(_: Request, e: ValidationError):
    return JSONResponse({"error": "入力が正しくありません: " + "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5])}, status_code=400)


@app.exception_handler(Exception)
async def _unexpected_error(request: Request, e: Exception):
    """想定外の例外も日本語の JSON で返し、画面とログに残す。"""
    log.exception("unhandled error on %s", request.url.path)
    msg = f"内部エラー: {type(e).__name__}: {e}"
    with contextlib.suppress(Exception):
        state.bus.toast("error", msg)
        state.session.row("error", detail=f"{request.url.path}: {msg}")
    return JSONResponse({"error": msg}, status_code=500)


# ================================================================ snapshot / events


def heartbeat() -> dict[str, Any]:
    sess = state.session.snapshot()
    return {
        "server_time": time.time(),
        "connection": state.monitor.snapshot.state,
        "main_remaining_s": sess.get("main_remaining_s"),
        "since_last_utterance_s": sess.get("since_last_utterance_s"),
        "performer": state.performer.status,
    }


def snapshot() -> dict[str, Any]:
    s = state.settings
    return {
        "server_time": time.time(),
        "app_version": APP_VERSION,
        "connection": state.monitor.snapshot.to_dict(),
        "session": state.session.snapshot(),
        "performer": state.performer.state(),
        "preflight": state.preflight,
        "resting": state.resting,
        "volume": state.volume,
        "warnings": state.warnings,
        "settings": s.model_dump(),
        "phrases": state.phrases.model_dump(),
        "leaves": {k: v.model_dump() for k, v in state.phrases.leaves().items()},
        "gesture_names": state.gestures.names(),
        "gesture_info": {name: (g.description or "") for name, g in state.gestures.root.items()},
        "recent_events": state.bus.recent[-50:],
    }


@app.get("/api/state")
async def get_state():
    return snapshot()


@app.get("/api/events")
async def events():
    gen = sse_stream(state.bus, snapshot(), heartbeat, interval=3.0)
    return StreamingResponse(gen, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"})


@app.post("/api/preflight")
async def post_preflight():
    start_preflight()
    return {"ok": True}


# ================================================================ session


class SessionStart(BaseModel):
    child_name: str
    suffix: str
    order: str
    condition: str


@app.post("/api/session/start")
async def session_start(body: SessionStart):
    if not state.monitor.can_control():
        raise PerformError(f"ロボットと接続されていません({state.monitor.snapshot.reason})", 503)
    if state.resting:
        raise PerformError("ロボットが休止中です。「起こす」を押してください", 409)
    if state.prewarm_task is not None and not state.prewarm_task.done():
        state.prewarm_task.cancel()
    await state.performer.reset()  # 前の子どもの一時停止などを持ち越さない
    state.session.start(body.child_name, body.suffix, body.order, body.condition)
    try:
        failures = await prewarm_all(include_child=True)
        if failures:
            raise PerformError("音声の準備に失敗したためセッションを開始できません: " + failures[0], 503)
        present = await state.robot.list_sounds()
        if any(n not in present for n in state.audio.wanted):
            await state.audio.reupload_missing()
    except Exception as e:
        state.session.end()
        if isinstance(e, PerformError):
            raise
        step("synth", "fail", str(e))
        raise PerformError(f"音声の準備に失敗したためセッションを開始できません: {e}", 503) from e
    state.session.row("system", detail=f"audio ready ({len(state.audio.wanted)} files)")
    return state.session.snapshot()


class PhaseBody(BaseModel):
    phase: str


@app.post("/api/session/phase")
async def session_phase(body: PhaseBody):
    state.session.set_phase(body.phase)
    return state.session.snapshot()


class IntroDone(BaseModel):
    part_id: str


@app.post("/api/session/intro_done")
async def session_intro_done(body: IntroDone):
    state.session.mark_intro_done(body.part_id)
    return state.session.snapshot()


@app.post("/api/session/end")
async def session_end():
    await state.performer.reset()
    state.session.end()
    return state.session.snapshot()


# ================================================================ play / control


class PlayBody(BaseModel):
    id: str


@app.post("/api/play")
async def play(body: PlayBody):
    return await state.performer.play_phrase(body.id)


@app.post("/api/phrases/test")
async def phrases_test(body: PlayBody):
    return await state.performer.play_phrase(body.id, log_it=False)


@app.post("/api/stop")
async def stop():
    await state.performer.stop()
    return state.performer.state()


@app.post("/api/pause")
async def pause():
    await state.performer.pause()
    return state.performer.state()


@app.post("/api/resume")
async def resume():
    await state.performer.resume()
    return state.performer.state()


class TrackingBody(BaseModel):
    enabled: bool


@app.post("/api/tracking")
async def tracking(body: TrackingBody):
    await state.performer.set_tracking(body.enabled)  # 適用できたときだけ設定が変わる
    save_atomic(SETTINGS_PATH, state.settings)
    return state.performer.state()


class GestureTest(BaseModel):
    name: str


@app.post("/api/gestures/test")
async def gestures_test(body: GestureTest):
    await state.performer.test_gesture(body.name)
    return {"ok": True}


@app.post("/api/robot/rest")
async def robot_rest():
    """ロボットを休ませる: 休止フラグを先に立て(ボタンとアイドルを止め)、スリープ姿勢 → モーター OFF。"""
    if state.resting:
        return {"resting": True}
    set_resting(True)
    try:
        async with state.performer.suspended():
            await state.robot.stop_sound()
            await state.robot.clear_moves()
            await state.robot.set_tracking(False)
            await state.robot.goto_sleep()
            if not await state.robot.wait_moves_done(8.0):
                raise RobotError("スリープ姿勢への移動が終わりませんでした(モーターは ON のままにします)")
            await state.robot.disable_motors()
    except RobotError as e:
        set_resting(False)
        with contextlib.suppress(RobotError):
            await state.performer.apply_tracking()
        state.session.row("error", detail=f"rest failed: {e}")
        raise
    state.session.row("system", detail="robot rest")
    return {"resting": True}


@app.post("/api/robot/wake")
async def robot_wake():
    await recover(wake=True)
    state.session.row("system", detail="robot wake")
    return {"resting": state.resting}


# ================================================================ settings / phrases / gestures


@app.get("/api/settings")
async def get_settings():
    return state.settings.model_dump()


@app.put("/api/settings")
async def put_settings(body: dict[str, Any]):
    new = Settings.model_validate(body)
    old = state.settings
    names = set(state.gestures.names())
    problems = [f"アイドル動作 '{g}' がありません" for g in (new.motion.idle.gesture_tracking_on, new.motion.idle.gesture_tracking_off) if g not in names]
    if problems:
        return JSONResponse({"error": "; ".join(problems)}, status_code=400)
    # 先に新しい部品を作り、失敗したら何も変えない
    new_tts: TTSBackend | None = None
    if new.tts.voicevox != old.tts.voicevox or new.tts.backend != old.tts.backend:
        new_tts = make_tts(new)
    state.settings = new
    save_atomic(SETTINGS_PATH, new)
    if new.robot != old.robot:
        await state.robot.set_base_url(new.robot.base_url, new.robot.connect_timeout_s, new.robot.read_timeout_s)
        state.monitor.snapshot.robot_url = new.robot.base_url
        state.monitor.poll_now()
    if new_tts is not None:
        old_tts = state.tts
        state.tts = new_tts
        state.audio.tts = new_tts
        await old_tts.aclose()
    warnings = state.gesture_lib.validate()
    if state.monitor.can_control():
        try:
            await state.performer.apply_tracking()
            await state.performer.apply_wobbling()
        except RobotError as e:
            state.bus.toast("error", f"設定の適用に失敗しました: {e}")
    voice_changed = new.tts != old.tts or new.names != old.names
    state.bus.publish("settings_changed", warnings=warnings)
    for w in warnings:
        state.bus.toast("warn", w)
    if voice_changed:
        state.bus.toast("info", "声または名前が変わったので、音声を裏で作り直しています")
        start_prewarm_background()
    return {"ok": True, "warnings": warnings}


@app.get("/api/phrases")
async def get_phrases():
    return state.phrases.model_dump()


@app.put("/api/phrases")
async def put_phrases(body: dict[str, Any]):
    new = Phrases.model_validate(body)
    problems = validate_phrases(new, set(state.gestures.names()))
    if problems:
        return JSONResponse({"error": "台本を保存できません: " + "; ".join(problems[:5])}, status_code=400)
    changed_text = new.model_dump() != state.phrases.model_dump()
    state.phrases = new
    save_atomic(PHRASES_PATH, new)
    state.bus.publish("settings_changed", warnings=[])
    if changed_text:
        start_prewarm_background()
    return {"ok": True}


@app.get("/api/gestures")
async def get_gestures():
    return state.gestures.model_dump()


@app.put("/api/gestures")
async def put_gestures(body: dict[str, Any]):
    new = Gestures.model_validate(body)
    old = state.gestures
    names = set(new.names())
    missing = [f"{k}: {v.gesture}" for k, v in state.phrases.leaves().items() if v.gesture not in names]
    idle = state.settings.motion.idle
    missing += [f"アイドル: {g}" for g in (idle.gesture_tracking_on, idle.gesture_tracking_off) if g not in names]
    if missing:
        return JSONResponse({"error": "台本や設定から使われているジェスチャーを消せません: " + ", ".join(missing)}, status_code=400)
    state.gestures = new
    try:
        warnings = state.gesture_lib.validate()
    except Exception as e:
        state.gestures = old
        return JSONResponse({"error": f"ジェスチャー定義を検証できません: {e}"}, status_code=400)
    save_atomic(GESTURES_PATH, new)
    state.bus.publish("settings_changed", warnings=warnings)
    return {"ok": True, "warnings": warnings}


# ================================================================ voice


@app.get("/api/voices")
async def voices():
    return [{"id": v.id, "name": v.name, "credit": v.credit} for v in await state.tts.list_voices()]


class PreviewBody(BaseModel):
    text: str
    voice_id: str | None = None
    params: VoiceParams | None = None
    target: str = "browser"


@app.post("/api/tts/preview")
async def tts_preview(body: PreviewBody):
    text = state.performer.expand_text(body.text)
    key, wav = await state.audio.synthesize_cached(text, body.voice_id, body.params)
    if body.target == "robot":
        if not state.monitor.can_control():
            raise PerformError("ロボットと接続されていません", 503)
        if state.performer.status != "idle":
            raise PerformError("再生中は試聴できません。ストップしてから試してください", 409)
        name = state.audio.basename(key)
        await state.robot.upload_sound(name, wav)
        await state.robot.play_sound(name)
        return {"ok": True, "duration": state.audio.duration(key)}
    return Response(content=wav, media_type="audio/wav")


class DictBody(BaseModel):
    surface: str
    pronunciation: str | None = None
    accent_type: int = 0


@app.post("/api/dict/word")
async def dict_word(body: DictBody):
    surface = body.surface.strip()
    if not surface:
        return JSONResponse({"error": "単語が空です"}, status_code=400)
    pron = hiragana_to_katakana((body.pronunciation or surface).strip())
    await state.tts.register_pronunciation(surface, pron, body.accent_type)  # 先にエンジンへ。失敗したら設定は変えない
    entries = [e for e in state.settings.tts.user_dict if e.surface != surface]
    entries.append(DictEntry(surface=surface, pronunciation=pron, accent_type=body.accent_type))
    state.settings.tts.user_dict = entries
    save_atomic(SETTINGS_PATH, state.settings)
    start_prewarm_background()
    return {"ok": True, "entries": [e.model_dump() for e in entries]}


class VolumeBody(BaseModel):
    volume: int


@app.post("/api/volume")
async def set_volume(body: VolumeBody):
    if state.performer.status != "idle":
        raise PerformError("再生中は音量を変えられません(テスト音が鳴るため)", 409)
    await state.robot.set_volume(body.volume)
    state.volume = await state.robot.get_volume()
    state.bus.publish("volume", volume=state.volume)
    return {"volume": state.volume}


# ================================================================ static UI(API より後にマウント)

app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
