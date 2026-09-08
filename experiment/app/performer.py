"""Performer: ボタン 1 回分の「音声 + 動作」を同期再生し、ストップ / 一時停止 / アイドルを司る。

- 再生中に別のボタンが押されたら、今のものを中断して新しいものを再生する(同一ボタンは 500 ms デバウンス)。
- ストップ: 音声停止 → 軌道をニュートラルへ戻して中断 → 実行中ムーブ停止 → 追跡復元。
- 一時停止: ストップ相当 + 顔追跡を weight 0 + アイドル動作停止。再開で追跡を戻す。
- アイドル: 一定間隔で小さな動作(追跡 ON のときはアンテナだけ / OFF のときは小さな頷き)。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable

from .audio import AudioStore
from .config import Phrases, Settings, expand
from .events import EventBus
from .gestures import GestureError, GestureLibrary, Trajectory
from .monitor import ConnectionMonitor
from .player import TrajectoryPlayer
from .robot import RobotClient, RobotError
from .session import SessionManager
from .tts.base import TTSError

log = logging.getLogger(__name__)


class PerformError(Exception):
    """ユーザー向けメッセージ付き(str())。"""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


class Performer:
    def __init__(
        self,
        robot: RobotClient,
        player: TrajectoryPlayer,
        audio: AudioStore,
        phrases_ref: Callable[[], Phrases],
        gestures: GestureLibrary,
        session: SessionManager,
        monitor: ConnectionMonitor,
        bus: EventBus,
        settings_ref: Callable[[], Settings],
    ) -> None:
        self.robot = robot
        self.player = player
        self.audio = audio
        self.phrases_ref = phrases_ref
        self.gestures = gestures
        self.session = session
        self.monitor = monitor
        self.bus = bus
        self.settings_ref = settings_ref

        self.paused = False
        self.current: dict[str, Any] | None = None
        self.status: str = "idle"  # idle | preparing | playing
        self._token = 0
        self._play_task: asyncio.Task | None = None
        self._finish_task: asyncio.Task | None = None
        self._last_press: dict[str, float] = {}
        self._start_lock = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None
        self.last_idle_at = time.monotonic()

    # ------------------------------------------------------------ state
    def state(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "paused": self.paused,
            "current": self.current,
            "tracking_enabled": self.settings_ref().motion.tracking_enabled,
            "server_time": time.time(),
            "motion_stats": dict(self.player.stats),
        }

    def _publish(self) -> None:
        self.bus.publish("performer", **self.state())

    def _set_status(self, status: str) -> None:
        self.status = status
        self._publish()

    def expand_text(self, text: str) -> str:
        s = self.settings_ref()
        child = self.session.current.child_display if self.session.current else "〇〇ちゃん"
        return expand(text, robot=s.names.robot, child=child, experimenter=s.names.experimenter)

    def texts_for_prewarm(self, include_child: bool) -> list[str]:
        """事前合成すべきテキスト。include_child=False なら名前を含むものは除く。"""
        out: list[str] = []
        for leaf in self.phrases_ref().leaves().values():
            if "{child}" in leaf.text and not include_child:
                continue
            out.append(self.expand_text(leaf.text))
        return out

    # ------------------------------------------------------------ play
    async def play_phrase(self, phrase_id: str, *, log_it: bool = True) -> dict[str, Any]:
        leaf = self.phrases_ref().leaves().get(phrase_id)
        if leaf is None:
            raise PerformError(f"台本にありません: {phrase_id}", 404)
        if not self.monitor.is_connected():
            raise PerformError(f"ロボットと接続されていません({self.monitor.snapshot.reason})", 503)
        now = time.monotonic()
        debounce = self.settings_ref().ui.debounce_ms / 1000.0
        if now - self._last_press.get(phrase_id, -1e9) < debounce:
            return {"accepted": False, "reason": "debounce"}
        self._last_press[phrase_id] = now
        text = self.expand_text(leaf.text)
        traj: Trajectory | None = None
        try:
            traj = self.gestures.build(leaf.gesture)
        except GestureError as e:
            self.bus.toast("error", f"ジェスチャー '{leaf.gesture}' を作れません: {e}(音声のみ再生します)")
            self.session.row("error", item_id=phrase_id, gesture=leaf.gesture, detail=str(e))
        async with self._start_lock:
            await self._interrupt()
            self._token += 1
            token = self._token
            if log_it:
                self.session.row("button", item_id=phrase_id, text=text, gesture=leaf.gesture)
            try:
                if not self.audio.is_synthesized(text):
                    self._set_status("preparing")
                name, duration = await self.audio.ensure(text)
                await self.robot.play_sound(name)
            except (TTSError, RobotError) as e:
                self._set_status("idle")
                self.session.row("error", item_id=phrase_id, text=text, detail=str(e))
                raise PerformError(f"再生できませんでした: {e}", 503) from e
            self.current = {
                "id": phrase_id,
                "text": text,
                "gesture": leaf.gesture,
                "category": leaf.category,
                "started_at": time.time(),
                "duration": duration,
            }
            self._set_status("playing")
            if leaf.category in ("empathy", "logical"):
                self.session.note_utterance()
            if leaf.category == "intro":
                try:
                    self.session.mark_intro_done(phrase_id)
                except Exception:
                    pass
            self.session.row("playing", item_id=phrase_id, text=text, gesture=leaf.gesture, detail=f"{duration:.2f}s")
            if traj is not None:
                self._play_task = asyncio.create_task(self._run_gesture(traj), name=f"gesture-{phrase_id}")
            self._finish_task = asyncio.create_task(self._finish_after(duration + 0.2, token, phrase_id), name=f"finish-{phrase_id}")
            return {"accepted": True, "id": phrase_id, "duration": duration}

    async def _run_gesture(self, traj: Trajectory) -> None:
        m = self.settings_ref().motion
        lead = max(0.0, m.audio_lead_ms / 1000.0)
        if lead:
            await asyncio.sleep(lead)
        await self.player.play(traj, pause_tracking=self._tracking_active(), tracking_weight=m.tracking_weight)

    async def _finish_after(self, delay: float, token: int, phrase_id: str) -> None:
        await asyncio.sleep(delay)
        if self._token != token:
            return
        self.session.row("done", item_id=phrase_id)
        self.current = None
        self.last_idle_at = time.monotonic()
        self._set_status("idle")

    async def _interrupt(self) -> None:
        """今の再生を静かに打ち切る(次の再生の直前に呼ぶ)。"""
        if self._finish_task is not None and not self._finish_task.done():
            self._finish_task.cancel()
        self._finish_task = None
        was_playing = self.current is not None
        if self.current is not None:
            self.session.row("interrupted", item_id=self.current["id"])
            self.current = None
        self._token += 1
        if was_playing:  # 何も鳴っていなければ余計な往復をしない(ボタン応答を速く)
            try:
                await self.robot.stop_sound()
            except RobotError as e:
                log.warning("stop_sound failed: %s", e)
        self.player.cancel(to_neutral=False)
        if not await self.player.wait_idle(1.0) and self._play_task is not None:
            self._play_task.cancel()
            try:
                await self._play_task
            except (asyncio.CancelledError, Exception):
                pass
        self._play_task = None

    # ------------------------------------------------------------ stop / pause
    async def stop(self, *, reason: str = "stop") -> None:
        async with self._start_lock:
            if self._finish_task is not None and not self._finish_task.done():
                self._finish_task.cancel()
            self._finish_task = None
            self._token += 1
            was = self.current["id"] if self.current else ""
            self.current = None
            errors: list[str] = []
            try:
                await self.robot.stop_sound()
            except RobotError as e:
                errors.append(str(e))
            self.player.cancel(to_neutral=True)
            if not await self.player.wait_idle(1.5) and self._play_task is not None:
                self._play_task.cancel()
                try:
                    await self._play_task
                except (asyncio.CancelledError, Exception):
                    pass
            self._play_task = None
            try:
                await self.robot.clear_moves()
            except RobotError as e:
                errors.append(str(e))
            if not self.paused:
                try:
                    await self.apply_tracking()
                except RobotError as e:
                    errors.append(str(e))
            self.session.row(reason, item_id=was, result="ok" if not errors else "error", detail="; ".join(errors))
            self.last_idle_at = time.monotonic()
            self._set_status("idle")
            if errors:
                self.bus.toast("error", "ストップ処理の一部に失敗しました: " + "; ".join(errors))

    async def pause(self) -> None:
        await self.stop(reason="pause")
        self.paused = True
        try:
            await self.robot.set_tracking(True, 0.0)
        except RobotError as e:
            self.bus.toast("error", f"顔追跡の一時停止に失敗しました: {e}")
        self._publish()

    async def resume(self) -> None:
        self.paused = False
        self.session.row("resume")
        try:
            await self.apply_tracking()
        except RobotError as e:
            self.bus.toast("error", f"顔追跡の再開に失敗しました: {e}")
        self.last_idle_at = time.monotonic()
        self._publish()

    # ------------------------------------------------------------ tracking / wobbling
    def _tracking_active(self) -> bool:
        return bool(self.settings_ref().motion.tracking_enabled) and not self.paused

    async def apply_tracking(self) -> None:
        m = self.settings_ref().motion
        if self._tracking_active():
            await self.robot.set_tracking(True, m.tracking_weight)
        elif self.paused:
            await self.robot.set_tracking(True, 0.0)
        else:
            await self.robot.set_tracking(False)

    async def apply_wobbling(self) -> None:
        await self.robot.set_wobbling(bool(self.settings_ref().motion.wobbling_enabled))

    async def set_tracking(self, enabled: bool) -> None:
        self.settings_ref().motion.tracking_enabled = bool(enabled)
        self.session.row("tracking", detail="on" if enabled else "off")
        await self.apply_tracking()
        self._publish()

    # ------------------------------------------------------------ idle
    def start_idle(self) -> None:
        if self._idle_task is None:
            self._idle_task = asyncio.create_task(self.idle_loop(), name="idle-loop")

    async def stop_idle(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            try:
                await self._idle_task
            except (asyncio.CancelledError, Exception):
                pass
            self._idle_task = None

    async def idle_loop(self) -> None:
        while True:
            idle = self.settings_ref().motion.idle
            wait = max(2.0, idle.interval_s + random.uniform(-idle.jitter_s, idle.jitter_s))
            await asyncio.sleep(1.0)
            if not idle.enabled or self.paused or self.status != "idle" or not self.monitor.is_connected() or self.player.is_playing:
                continue
            if time.monotonic() - self.last_idle_at < wait:
                continue
            name = idle.gesture_tracking_on if self._tracking_active() else idle.gesture_tracking_off
            try:
                traj = self.gestures.build(name)
            except GestureError as e:
                log.warning("idle gesture %s: %s", name, e)
                self.last_idle_at = time.monotonic()
                continue
            self.last_idle_at = time.monotonic()
            await self.player.play(traj, pause_tracking=False)

    # ------------------------------------------------------------ misc
    async def test_gesture(self, name: str) -> None:
        if not self.monitor.is_connected():
            raise PerformError("ロボットと接続されていません", 503)
        try:
            traj = self.gestures.build(name)
        except GestureError as e:
            raise PerformError(str(e), 400) from e
        async with self._start_lock:
            await self._interrupt()
            self.session.row("system", gesture=name, detail="test gesture")
            self._play_task = asyncio.create_task(self.player.play(traj, pause_tracking=self._tracking_active(), tracking_weight=self.settings_ref().motion.tracking_weight))

    async def on_reconnected(self) -> None:
        """復旧後: 途中だった再生を idle に戻す。"""
        if self.current is not None or self.status != "idle":
            self.current = None
            self._token += 1
            self._set_status("idle")
