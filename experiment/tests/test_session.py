import csv
import json

import pytest

from app.config import Settings, default_phrases
from app.events import EventBus
from app.session import CSV_COLUMNS, SessionError, SessionManager


@pytest.fixture
def mgr(tmp_path):
    settings = Settings()
    phrases = default_phrases()
    return SessionManager(tmp_path / "logs", EventBus(), lambda: settings, lambda: phrases)


def test_validation(mgr):
    with pytest.raises(SessionError):
        mgr.start("Hana", "ちゃん", "robot_first", "empathy")
    with pytest.raises(SessionError):
        mgr.start("はな", "さん", "robot_first", "empathy")
    with pytest.raises(SessionError):
        mgr.start("はな", "ちゃん", "bogus", "empathy")
    with pytest.raises(SessionError):
        mgr.start("はな", "ちゃん", "robot_first", "bogus")
    with pytest.raises(SessionError):
        mgr.set_phase("main")


def test_start_log_and_snapshot(mgr, tmp_path):
    s = mgr.start("はな", "ちゃん", "robot_first", "empathy")
    assert s.child_display == "はなちゃん"
    snap = mgr.snapshot()
    assert snap["active"] and snap["phase"] == "intro" and snap["main_remaining_s"] is None
    assert snap["intro_sequence"] == ["A1", "A2", "A3", "A4", "A5-1", "A5-2"]
    mgr.row("button", item_id="A1", text="初めまして", gesture="greet_wiggle")
    mgr.set_phase("main")
    snap = mgr.snapshot()
    assert 0 < snap["main_remaining_s"] <= 8 * 60
    mgr.note_utterance()
    assert mgr.snapshot()["since_last_utterance_s"] < 1.0
    mgr.mark_intro_done("A1")
    assert mgr.snapshot()["intro_done"] == ["A1"]

    # ログは行ごとに flush されている(閉じる前に読める)
    csv_path = tmp_path / "logs" / (s.log_stem.rsplit("/", 1)[-1] + ".csv")
    raw = csv_path.read_bytes()
    assert raw.startswith("﻿".encode("utf-8"))
    rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
    assert rows[0] == CSV_COLUMNS
    kinds = [r[3] for r in rows[1:]]
    assert kinds == ["session", "button", "phase"]
    assert rows[2][4] == "A1" and rows[2][5] == "初めまして"

    meta = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["child"] == "はなちゃん" and meta["condition"] == "empathy"
    assert meta["settings"]["names"]["robot"] == "ドラちゃん"
    assert "phrases" in meta

    mgr.end()
    assert mgr.current is None and not mgr.snapshot()["active"]
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8-sig").splitlines()))
    assert rows[-1][3] == "session" and rows[-1][8] == "end"


def test_starting_new_session_ends_previous(mgr):
    a = mgr.start("はな", "ちゃん", "robot_first", "empathy")
    b = mgr.start("たろう", "くん", "experimenter_first", "logical")
    assert mgr.current is b and b.log_stem != a.log_stem
    assert mgr.snapshot()["intro_sequence"] == ["A1", "A2", "A6-1", "A6-2", "A6-3"]


def test_row_without_session_only_publishes(mgr):
    q = mgr.bus.subscribe()
    mgr.row("system", detail="x")
    ev = q.get_nowait()
    assert ev["type"] == "log" and ev["phase"] == "-"
