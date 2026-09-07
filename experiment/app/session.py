"""セッション(1 人の子どもの 1 回分)の状態・フェーズ・タイマーと、CSV ログ。"""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
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
    main_started_at: str | None = None
    main_started_mono: float | None = None
    last_utterance_mono: float | None = None
    intro_done: list[str] = []
    log_stem: str = ""

    @property
    def child_display(self) -> str:
        return f"{self.child_name}{self.suffix}"


class SessionLog:
    def __init__(self, stem: Path, meta: dict[str, Any]) -> None:
        self.stem = stem
        stem.parent.mkdir(parents=True, exist_ok=True)
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
        now = dt.datetime.now()
        stem = self.logs_dir / f"{now.strftime('%Y%m%d_%H%M%S')}_{name}{suffix}"
        s = Session(
            child_name=name,
            suffix=suffix,  # type: ignore[arg-type]
            order=order,  # type: ignore[arg-type]
            condition=condition,  # type: ignore[arg-type]
            started_at=now.isoformat(timespec="seconds"),
            started_mono=time.monotonic(),
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
        self.bus.publish("session", **self.snapshot())

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
            s.main_started_at = dt.datetime.now().isoformat(timespec="seconds")
        self.row("phase", detail=phase)
        self.bus.publish("session", **self.snapshot())

    def mark_intro_done(self, part_id: str) -> None:
        s = self._require()
        if part_id not in s.intro_done:
            s.intro_done.append(part_id)
            self.bus.publish("session", **self.snapshot())

    def note_utterance(self) -> None:
        if self.current is not None:
            self.current.last_utterance_mono = time.monotonic()

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
            return {"active": False}
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
