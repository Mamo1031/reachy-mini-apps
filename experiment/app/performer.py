"""Performer: ボタン 1 回分の「音声 + 動作」を同期再生し、ストップ / 一時停止 / アイドルを司る。

- 再生中に別のボタンが押されたら、今のものを中断して新しいものを再生する(同一ボタンは 500 ms デバウンス)。
- 音声の合成・アップロードはロックの外で行う(ストップや別ボタンが「準備中」に待たされない)。
  各操作は世代番号(_gen)を持ち、準備中に新しい操作が来たら古い方は静かに諦める。
- ストップ: 動作を中断してニュートラルへ → 音声停止 → 取り残されたタスクを確実に片付け → 追跡を現在の設定に戻す。
- 一時停止: paused を立ててからストップ相当(追跡は weight 0 のまま)。再開で追跡を戻す。
- アイドル: 一定間隔で小さな動作(追跡 ON のときはアンテナだけ / OFF のときは小さな頷き)。
  再生準備中・休止中・復旧中・一時停止中は動かない。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from typing import Any, AsyncIterator, Callable

from .audio import AudioError, AudioStore
from .config import Phrases, Settings, expand
from .events import EventBus
from .gestures import GestureError, GestureLibrary, Trajectory
from .monitor import ConnectionMonitor
from .player import TrajectoryPlayer
from .robot import RobotClient, RobotError, RobotHttpError
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
        resting_ref: Callable[[], bool] = lambda: False,
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
        self.resting_ref = resting_ref

        self.paused = False
        self.busy = False  # ボタン受付〜再生終了まで(アイドルを止める)
        self.current: dict[str, Any] | None = None
        self.status: str = "idle"  # idle | preparing | playing
        self._gen = 0
        self._play_task: asyncio.Task | None = None
        self._finish_task: asyncio.Task | None = None
        self._last_press: dict[str, float] = {}
        self._lock = asyncio.Lock()  # 再生開始 / ストップ / 一時停止を直列化
        self._idle_task: asyncio.Task | None = None
        self._suspend = 0  # >0 の間はアイドル禁止(復旧・休止処理中)
        self._tracking_dirty = False  # 追跡の適用に失敗した(接続回復後に再適用)
        self._idle_error: str | None = None
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

    def _next_gen(self) -> int:
        self._gen += 1
        return self._gen

    def expand_text(self, text: str) -> str:
        s = self.settings_ref()
        child = self.session.current.child_display if self.session.current else "〇〇ちゃん"
        return expand(text, robot=s.names.robot, child=child, experimenter=s.names.experimenter)

    def texts_for_prewarm(self, include_child: bool) -> list[str]:
        """事前合成すべきテキスト(重複なし)。include_child=False なら名前を含むものは除く。"""
        out: list[str] = []
        for leaf in self.phrases_ref().leaves().values():
            if "{child}" in leaf.text and not include_child:
                continue
            t = self.expand_text(leaf.text)
            if t not in out:
                out.append(t)
        return out

    @contextlib.asynccontextmanager
    async def suspended(self) -> AsyncIterator[None]:
        """復旧・休止処理の間、アイドル動作を止め、進行中の再生を静かに片付ける。"""
        self._suspend += 1
        try:
            async with self._lock:
                await self._interrupt_current()
            yield
        finally:
            self._suspend -= 1
            self.last_idle_at = time.monotonic()

    # ------------------------------------------------------------ play
    async def play_phrase(self, phrase_id: str, *, log_it: bool = True) -> dict[str, Any]:
        leaf = self.phrases_ref().leaves().get(phrase_id)
        if leaf is None:
            raise PerformError(f"台本にありません: {phrase_id}", 404)
        if not self.monitor.can_control():
            raise PerformError(f"ロボットと接続されていません({self.monitor.snapshot.reason})", 503)
        if self.resting_ref():
            raise PerformError("ロボットが休止中です。「起こす」を押してください", 409)
        now = time.monotonic()
        debounce = self.settings_ref().ui.debounce_ms / 1000.0
        if now - self._last_press.get(phrase_id, -1e9) < debounce:
            return {"accepted": False, "reason": "debounce"}
        self._last_press[phrase_id] = now

        gen = self._next_gen()
        self.busy = True
        text = self.expand_text(leaf.text)
        traj: Trajectory | None = None
        try:
            traj = self.gestures.build(leaf.gesture)
        except GestureError as e:
            self.bus.toast("error", f"ジェスチャー '{leaf.gesture}' を作れません: {e}(音声のみ再生します)")
            self.session.row("error", item_id=phrase_id, gesture=leaf.gesture, detail=str(e))

        # --- 音声の準備(ロックの外。ストップや次のボタンを待たせない)
        try:
            if not self.audio.is_synthesized(text):
                self._set_status("preparing")
            name, duration = await self.audio.ensure(text)
        except (TTSError, RobotError, AudioError) as e:
            if self._gen == gen:
                self.busy = False
                self._set_status("idle")
            self.session.row("error", item_id=phrase_id, text=text, detail=str(e))
            raise PerformError(f"再生できませんでした: {e}", 503) from e
        if self._gen != gen:  # 準備中にストップ / 別のボタンが来た
            return {"accepted": False, "reason": "superseded"}

        async with self._lock:
            if self._gen != gen:
                return {"accepted": False, "reason": "superseded"}
            await self._interrupt_current()
            if log_it:
                self.session.row("button", item_id=phrase_id, text=text, gesture=leaf.gesture)
            else:
                self.session.row("system", item_id=phrase_id, gesture=leaf.gesture, detail="test phrase")
            try:
                await self._play_sound_with_retry(name, text)
            except RobotError as e:
                self.busy = False
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
                "test": not log_it,
            }
            self._set_status("playing")
            if log_it:
                if leaf.category in ("empathy", "logical"):
                    self.session.note_utterance()
                if leaf.category == "intro":
                    with contextlib.suppress(Exception):
                        self.session.mark_intro_done(phrase_id)
                self.session.row("playing", item_id=phrase_id, text=text, gesture=leaf.gesture, detail=f"{duration:.2f}s")
            if traj is not None:
                self._play_task = asyncio.create_task(self._run_gesture(traj), name=f"gesture-{phrase_id}")
            self._finish_task = asyncio.create_task(self._finish_after(duration + 0.2, gen, phrase_id, log_it), name=f"finish-{phrase_id}")
            return {"accepted": True, "id": phrase_id, "duration": duration}

    async def _play_sound_with_retry(self, name: str, text: str) -> None:
        """再生。ロボット上から音声が消えていた(404)ら再アップロードして 1 回だけやり直す。"""
        try:
            await self.robot.play_sound(name)
        except RobotHttpError as e:
            if e.status != 404:
                raise
            log.warning("sound %s missing on robot; re-uploading", name)
            self.audio.forget(name)
            name2, _ = await self.audio.ensure(text)
            await self.robot.play_sound(name2)

    async def _run_gesture(self, traj: Trajectory) -> None:
        try:
            m = self.settings_ref().motion
            lead = max(0.0, m.audio_lead_ms / 1000.0)
            if lead:
                await asyncio.sleep(lead)
            await self.player.play(traj, pause_tracking=self._tracking_active(), restore_tracking=self.apply_tracking)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # 想定外でもタスクを静かに死なせない
            log.exception("gesture task failed")
            self.bus.toast("error", f"動作でエラー: {e}")

    async def _finish_after(self, delay: float, gen: int, phrase_id: str, log_it: bool) -> None:
        await asyncio.sleep(delay)
        if self._gen != gen:
            return
        if log_it:
            self.session.row("done", item_id=phrase_id)
        self.current = None
        self.busy = False
        self.last_idle_at = time.monotonic()
        self._set_status("idle")

    async def _interrupt_current(self) -> None:
        """今の再生を静かに打ち切る(ロック内で呼ぶ)。世代は進めない。"""
        if self._finish_task is not None and not self._finish_task.done():
            self._finish_task.cancel()
        self._finish_task = None
        was_playing = self.current is not None
        if self.current is not None:
            if not self.current.get("test"):
                self.session.row("interrupted", item_id=self.current["id"])
            self.current = None
        await self._retire_gesture(to_neutral=False)
        if was_playing:  # 何も鳴っていなければ余計な往復をしない(ボタン応答を速く)
            try:
                await self.robot.stop_sound()
            except RobotError as e:
                log.warning("stop_sound failed: %s", e)
        if self.status != "idle" and was_playing:
            self.status = "idle"

    async def _retire_gesture(self, *, to_neutral: bool) -> None:
        """動作タスクを確実に終わらせる。協調キャンセル → 待つ → だめなら Task.cancel。

        音声先行の待ち時間中(まだ player が動いていない)に呼ばれても、タスクを取り残さない。
        """
        task, self._play_task = self._play_task, None
        if self.player.is_playing:
            # 動作中: 協調キャンセル(to_neutral ならランプアウト)を待つ
            self.player.cancel(to_neutral=to_neutral)
            await self.player.wait_idle(1.5 if to_neutral else 1.0)
        if task is not None and not task.done():
            # まだ動いていない(音声先行の待ち中)か、協調キャンセルが間に合わなかった: 即座に止める
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            if to_neutral:
                await self.player.go_neutral()

    # ------------------------------------------------------------ stop / pause
    async def stop(self, *, reason: str = "stop") -> None:
        self._next_gen()  # 準備中の操作を無効化
        async with self._lock:
            if self._finish_task is not None and not self._finish_task.done():
                self._finish_task.cancel()
            self._finish_task = None
            was = self.current["id"] if self.current else ""
            self.current = None
            errors: list[str] = []
            await self._retire_gesture(to_neutral=True)
            try:
                await self.robot.stop_sound()
            except RobotError as e:
                errors.append(str(e))
            try:
                await self.robot.clear_moves()
            except RobotError as e:
                errors.append(str(e))
            try:
                await self.apply_tracking()
            except RobotError as e:
                errors.append(str(e))
            self.session.row(reason, item_id=was, result="ok" if not errors else "error", detail="; ".join(errors))
            self.busy = False
            self.last_idle_at = time.monotonic()
            self._set_status("idle")
            if errors:
                self.bus.toast("error", "ストップ処理の一部に失敗しました: " + "; ".join(errors))

    async def pause(self) -> None:
        self.paused = True  # 先に立てる: stop() 内の追跡適用が weight 0 になる(一瞬だけ追跡が戻る隙を作らない)
        await self.stop(reason="pause")
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

    async def reset(self) -> None:
        """セッションの境界で呼ぶ: 一時停止などの状態を次の子どもに持ち越さない。"""
        changed = self.paused
        self.paused = False
        if self.current is not None or self.player.is_playing or self._play_task is not None:
            await self.stop(reason="stop")
        if changed:
            try:
                await self.apply_tracking()
            except RobotError as e:
                self.bus.toast("error", f"顔追跡の再開に失敗しました: {e}")
        self._publish()

    # ------------------------------------------------------------ tracking / wobbling
    def _tracking_active(self) -> bool:
        return bool(self.settings_ref().motion.tracking_enabled) and not self.paused

    async def apply_tracking(self) -> None:
        """設定と一時停止状態から「今あるべき追跡状態」をロボットに適用する。

        ジェスチャー再生中で追跡を止めている間は何もしない(再生終了時に同じ関数が呼ばれる)。
        """
        if self.player.tracking_paused and self.player.is_playing:
            return
        m = self.settings_ref().motion
        try:
            if self._tracking_active():
                await self.robot.set_tracking(True, m.tracking_weight)
            elif self.paused:
                await self.robot.set_tracking(True, 0.0)
            else:
                await self.robot.set_tracking(False)
        except RobotError:
            self._tracking_dirty = True
            raise
        self._tracking_dirty = False

    async def apply_wobbling(self) -> None:
        await self.robot.set_wobbling(bool(self.settings_ref().motion.wobbling_enabled))

    async def set_tracking(self, enabled: bool) -> None:
        """UI のトグル。ロボットに適用できたときだけ設定を書き換える。"""
        s = self.settings_ref().motion
        before = s.tracking_enabled
        s.tracking_enabled = bool(enabled)
        try:
            await self.apply_tracking()
        except RobotError:
            s.tracking_enabled = before
            self._publish()
            raise
        self.session.row("tracking", detail="on" if enabled else "off")
        self._publish()

    # ------------------------------------------------------------ idle
    def start_idle(self) -> None:
        if self._idle_task is None:
            self._idle_task = asyncio.create_task(self.idle_loop(), name="idle-loop")

    async def stop_idle(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._idle_task
            self._idle_task = None

    def _idle_allowed(self) -> bool:
        return (
            self.settings_ref().motion.idle.enabled
            and not self.paused
            and not self.busy
            and self.status == "idle"
            and self._suspend == 0
            and not self.resting_ref()
            and self.monitor.is_connected()
            and not self.player.is_playing
        )

    async def idle_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                await self._idle_tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # ループは決して死なない
                log.exception("idle tick failed")
                self._toast_idle_error(f"アイドル動作でエラー: {e}")

    async def _idle_tick(self) -> None:
        # 追跡の適用に失敗していたら、接続が戻り次第やり直す(自己修復)
        if self._tracking_dirty and self.monitor.is_connected() and not self.player.is_playing:
            with contextlib.suppress(RobotError):
                await self.apply_tracking()
        if not self._idle_allowed():
            return
        idle = self.settings_ref().motion.idle
        wait = max(2.0, idle.interval_s + random.uniform(-idle.jitter_s, idle.jitter_s))
        if time.monotonic() - self.last_idle_at < wait:
            return
        name = idle.gesture_tracking_on if self._tracking_active() else idle.gesture_tracking_off
        try:
            traj = self.gestures.build(name)
        except GestureError as e:
            self.last_idle_at = time.monotonic()
            self._toast_idle_error(f"アイドル動作 '{name}' を作れません: {e}(設定 > 動作 を確認してください)")
            return
        self.last_idle_at = time.monotonic()
        if self._idle_allowed():
            await self.player.play(traj, pause_tracking=False)

    def _toast_idle_error(self, msg: str) -> None:
        if msg != self._idle_error:  # 同じエラーを毎秒出さない
            self._idle_error = msg
            self.bus.toast("warn", msg)

    # ------------------------------------------------------------ misc
    async def test_gesture(self, name: str) -> None:
        if not self.monitor.can_control():
            raise PerformError("ロボットと接続されていません", 503)
        if self.resting_ref():
            raise PerformError("ロボットが休止中です。「起こす」を押してください", 409)
        try:
            traj = self.gestures.build(name)
        except GestureError as e:
            raise PerformError(str(e), 400) from e
        self._next_gen()
        async with self._lock:
            await self._interrupt_current()
            self.busy = True
            self.session.row("system", gesture=name, detail="test gesture")
            self._set_status("idle")

            async def run() -> None:
                try:
                    await self.player.play(traj, pause_tracking=self._tracking_active(), restore_tracking=self.apply_tracking)
                finally:
                    self.busy = False
                    self.last_idle_at = time.monotonic()

            self._play_task = asyncio.create_task(run(), name=f"gesture-test-{name}")

    async def on_reconnected(self) -> None:
        """復旧後: 途中だった再生を idle に戻す。"""
        if self.current is not None or self.status != "idle" or self.busy:
            self._next_gen()
            self.current = None
            self.busy = False
            self._set_status("idle")
