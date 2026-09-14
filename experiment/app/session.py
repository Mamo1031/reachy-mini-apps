"""セッション(1 人の子どもの 1 回分)の状態・フェーズ・タイマーと、CSV ログ。

状態は変化のたびに `logs/.current_session.json` に書く。サーバーが途中で止まっても(Ctrl+C、クラッシュ)、
起動時にそのファイルが残っていれば「続きから再開」できる(同じ CSV に追記、タイマーは実時刻から再計算)。
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict

from .config import EncouragementCondition, OrderCondition, Phrases, Settings
from .events import EventBus

log = logging.getLogger(__name__)

HIRAGANA_RE = re.compile(r"^[ぁ-ゖー]+$")
Phase = Literal["intro", "baseline", "main", "ended"]
CSV_COLUMNS = ["time_iso", "elapsed_s", "phase", "kind", "item_id", "text", "gesture", "result", "detail"]
STATE_FILE = ".current_session.json"
RESUME_WINDOW_S = 30 * 60  # これより古い中断セッションは再開の候補にしない


class SessionError(Exception):
    pass


class Session(BaseModel):
    model_config = ConfigDict(extra="ignore")

    child_name: str
    suffix: Literal["ちゃん", "くん"]
    order: OrderCondition
    condition: EncouragementCondition
    phase: Phase = "intro"
    started_at: str
    started_mono: float
    started_wall: float = 0.0  # time.time()。再開時に monotonic を計算し直すため
    main_started_at: str | None = None
    main_started_mono: float | None = None
    main_started_wall: float | None = None
    last_utterance_mono: float | None = None
    last_utterance_wall: float | None = None
    intro_done: list[str] = []
    log_stem: str = ""

    @property
    def child_display(self) -> str:
        return f"{self.child_name}{self.suffix}"


class SessionLog:
    def __init__(self, stem: Path, meta: dict[str, Any] | None) -> None:
        self.stem = stem
        stem.parent.mkdir(parents=True, exist_ok=True)
        if meta is not None:  # 再開時は開始時の meta(設定と台本の写し)を残す
            stem.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._fh = open(stem.with_suffix(".csv"), "a", encoding="utf-8-sig", newline="")
        self._w = csv.writer(self._fh)
        if self._fh.tell() == 0:
            self._w.writerow(CSV_COLUMNS)
            self._fh.flush()

    def row(self, elapsed_s: float, phase: str, kind: str, item_id: str = "", text: str = "", gesture: str = "", result: str = "", detail: str = "") -> None:
        self._w.writerow([dt.datetime.now().isoformat(timespec="milliseconds"), f"{elapsed_s:.3f}", phase, kind, item_id, text, gesture, result, detail])
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


class SessionManager:
    def __init__(self, logs_dir: Path, bus: EventBus, settings_ref: Callable[[], Settings], phrases_ref: Callable[[], Phrases], app_version: str = "0.1.0") -> None:
        self.logs_dir = logs_dir
        self.bus = bus
        self.settings_ref = settings_ref
        self.phrases_ref = phrases_ref
        self.app_version = app_version
        self.current: Session | None = None
        self._log: SessionLog | None = None
        self.pending: dict[str, Any] | None = None  # 中断したセッションの保存内容(再開候補)

    # ------------------------------------------------------------ lifecycle
    def start(self, child_name: str, suffix: str, order: str, condition: str) -> Session:
        name = child_name.strip()
        if not name or not HIRAGANA_RE.match(name):
            raise SessionError("お名前はひらがなで入力してください")
        if suffix not in ("ちゃん", "くん"):
            raise SessionError("ちゃん / くん を選んでください")
        if order not in ("robot_first", "experimenter_first"):
            raise SessionError("順序条件を選んでください")
        if condition not in ("empathy", "logical"):
            raise SessionError("励まし条件を選んでください")
        if self.current is not None:
            self.end()
        if self.pending is not None:
            self.discard("superseded by a new session")
        now = dt.datetime.now()
        stem = self.logs_dir / f"{now.strftime('%Y%m%d_%H%M%S')}_{name}{suffix}"
        s = Session(
            child_name=name,
            suffix=suffix,  # type: ignore[arg-type]
            order=order,  # type: ignore[arg-type]
            condition=condition,  # type: ignore[arg-type]
            started_at=now.isoformat(timespec="seconds"),
            started_mono=time.monotonic(),
            started_wall=time.time(),
            log_stem=str(stem),
        )
        settings = self.settings_ref()
        meta = {
            "child": s.child_display,
            "order": order,
            "condition": condition,
            "started_at": s.started_at,
            "app_version": self.app_version,
            "settings": settings.model_dump(),
            "phrases": self.phrases_ref().model_dump(),
        }
        self._log = SessionLog(stem, meta)
        self.current = s
        self.row("session", detail=f"start {s.child_display} order={order} condition={condition}")
        self._persist()
        self.bus.publish("session", **self.snapshot())
        return s

    def end(self) -> None:
        if self.current is None:
            return
        self.current.phase = "ended"
        self.row("session", detail="end")
        if self._log:
            self._log.close()
        self._log = None
        self.current = None
        self._clear_state_file()
        self.bus.publish("session", **self.snapshot())

    def suspend(self) -> None:
        """サーバー停止時: セッションを終わらせず、次回起動で再開できるように残す。"""
        if self.current is None:
            return
        self.row("session", detail="suspended (server stopped)")
        self._persist()
        if self._log:
            self._log.close()
        self._log = None
        self.current = None

    # ------------------------------------------------------------ resume
    def load_pending(self, now: float | None = None) -> dict[str, Any] | None:
        """起動時: 中断セッションの保存内容を読む。古すぎるものは破棄する。"""
        path = self.logs_dir / STATE_FILE
        self.pending = None
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sess = Session.model_validate(data["session"])
            updated_at = float(data["updated_at"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("session state file unreadable: %s", e)
            self._clear_state_file()
            return None
        now = time.time() if now is None else now
        if now - updated_at > RESUME_WINDOW_S:
            self.pending = {"session": sess.model_dump(), "updated_at": updated_at}
            self.discard("too old to resume")
            return None
        self.pending = {"session": sess.model_dump(), "updated_at": updated_at}
        return self.pending_summary()

    def pending_summary(self, now: float | None = None) -> dict[str, Any] | None:
        if self.pending is None:
            return None
        s = Session.model_validate(self.pending["session"])
        now = time.time() if now is None else now
        total = self.settings_ref().session.main_minutes * 60.0
        remaining = None
        if s.main_started_wall is not None:
            remaining = max(0.0, total - (now - s.main_started_wall))
        return {
            "child": s.child_display,
            "child_name": s.child_name,
            "suffix": s.suffix,
            "order": s.order,
            "condition": s.condition,
            "phase": s.phase,
            "started_at": s.started_at,
            "main_remaining_s": remaining,
            "age_s": max(0.0, now - float(self.pending["updated_at"])),
            "log_stem": s.log_stem,
        }

    def resume(self) -> Session:
        """中断セッションを続きから再開する(同じ CSV に追記、タイマーは実時刻から計算し直す)。"""
        if self.pending is None:
            raise SessionError("再開できるセッションがありません")
        if self.current is not None:
            self.end()
        s = Session.model_validate(self.pending["session"])
        now_wall, now_mono = time.time(), time.monotonic()
        s.started_mono = now_mono - (now_wall - s.started_wall)
        if s.main_started_wall is not None:
            s.main_started_mono = now_mono - (now_wall - s.main_started_wall)
        if s.last_utterance_wall is not None:
            s.last_utterance_mono = now_mono - (now_wall - s.last_utterance_wall)
        if s.phase == "ended":
            s.phase = "intro"
        gap = now_wall - float(self.pending["updated_at"])
        self._log = SessionLog(Path(s.log_stem), None)
        self.current = s
        self.pending = None
        self.row("session", detail=f"resumed after restart (gap {gap:.0f}s)")
        self._persist()
        self.bus.publish("session", **self.snapshot())
        return s

    def discard(self, reason: str = "discarded") -> None:
        """中断セッションを再開せずに閉じる(古い CSV に end 行だけ付ける)。"""
        pending, self.pending = self.pending, None
        self._clear_state_file()
        if pending is None:
            return
        try:
            s = Session.model_validate(pending["session"])
            csv_path = Path(s.log_stem).with_suffix(".csv")
            if csv_path.exists():
                elapsed = time.time() - s.started_wall
                with open(csv_path, "a", encoding="utf-8-sig", newline="") as fh:
                    csv.writer(fh).writerow([dt.datetime.now().isoformat(timespec="milliseconds"), f"{elapsed:.3f}", s.phase, "session", "", "", "", "", f"end ({reason})"])
        except (OSError, ValueError, KeyError) as e:
            log.warning("could not close discarded session log: %s", e)
        self.bus.publish("session", **self.snapshot())

    def _persist(self) -> None:
        if self.current is None:
            return
        path = self.logs_dir / STATE_FILE
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({"version": 1, "updated_at": time.time(), "session": self.current.model_dump()}, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError as e:
            log.warning("session state save failed: %s", e)

    def _clear_state_file(self) -> None:
        try:
            (self.logs_dir / STATE_FILE).unlink(missing_ok=True)
        except OSError as e:
            log.warning("session state remove failed: %s", e)

    # ------------------------------------------------------------ state
    def set_phase(self, phase: str) -> None:
        s = self._require()
        if phase not in ("intro", "baseline", "main", "ended"):
            raise SessionError(f"不明なフェーズ: {phase}")
        if phase == "ended":
            self.end()
            return
        s.phase = phase  # type: ignore[assignment]
        if phase == "main" and s.main_started_mono is None:
            s.main_started_mono = time.monotonic()
            s.main_started_wall = time.time()
            s.main_started_at = dt.datetime.now().isoformat(timespec="seconds")
            # 本番開始から最初の声かけまでも「30 秒の目安」を出す
            if s.last_utterance_mono is None:
                s.last_utterance_mono = s.main_started_mono
                s.last_utterance_wall = s.main_started_wall
        self.row("phase", detail=phase)
        self._persist()
        self.bus.publish("session", **self.snapshot())

    def mark_intro_done(self, part_id: str) -> None:
        s = self._require()
        if part_id not in s.intro_done:
            s.intro_done.append(part_id)
            self._persist()
            self.bus.publish("session", **self.snapshot())

    def note_utterance(self) -> None:
        if self.current is not None:
            self.current.last_utterance_mono = time.monotonic()
            self.current.last_utterance_wall = time.time()
            self._persist()

    def intro_sequence(self) -> list[str]:
        s = self._require()
        return self.phrases_ref().intro_sequence(s.order)

    def _require(self) -> Session:
        if self.current is None:
            raise SessionError("セッションが始まっていません")
        return self.current

    # ------------------------------------------------------------ log
    def elapsed(self) -> float:
        return time.monotonic() - self.current.started_mono if self.current else 0.0

    def row(self, kind: str, item_id: str = "", text: str = "", gesture: str = "", result: str = "", detail: str = "") -> None:
        phase = self.current.phase if self.current else "-"
        if self._log is not None:
            try:
                self._log.row(self.elapsed(), phase, kind, item_id, text, gesture, result, detail)
            except OSError as e:
                log.error("log write failed: %s", e)
                self.bus.toast("error", f"ログの書き込みに失敗しました: {e}")
        self.bus.publish("log", kind=kind, item_id=item_id, text=text, gesture=gesture, result=result, detail=detail, phase=phase)

    # ------------------------------------------------------------ snapshot
    def snapshot(self) -> dict[str, Any]:
        s = self.current
        if s is None:
            return {"active": False, "pending": self.pending_summary()}
        now = time.monotonic()
        total = self.settings_ref().session.main_minutes * 60.0
        remaining = None
        if s.main_started_mono is not None:
            remaining = max(0.0, total - (now - s.main_started_mono))
        since = None if s.last_utterance_mono is None else now - s.last_utterance_mono
        return {
            "active": True,
            "child": s.child_display,
            "child_name": s.child_name,
            "suffix": s.suffix,
            "order": s.order,
            "condition": s.condition,
            "phase": s.phase,
            "started_at": s.started_at,
            "main_started_at": s.main_started_at,
            "main_total_s": total,
            "main_remaining_s": remaining,
            "since_last_utterance_s": since,
            "intro_done": list(s.intro_done),
            "intro_sequence": self.phrases_ref().intro_sequence(s.order),
            "log_stem": s.log_stem,
        }
