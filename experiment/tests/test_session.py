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


def _mgr(logs, bus, settings, phrases):
    return SessionManager(logs, bus, lambda: settings, lambda: phrases)


def test_suspend_persists_and_resume_continues_same_log(tmp_path):
    import time
    from pathlib import Path

    from app.session import STATE_FILE

    settings, phrases, bus = Settings(), default_phrases(), EventBus()
    logs = tmp_path / "logs"
    m1 = _mgr(logs, bus, settings, phrases)
    s = m1.start("はな", "ちゃん", "robot_first", "empathy")
    m1.mark_intro_done("A1")
    m1.set_phase("main")
    m1.note_utterance()
    time.sleep(0.3)
    m1.suspend()  # Ctrl+C 相当
    assert m1.current is None and (logs / STATE_FILE).exists()
    csv_path = Path(s.log_stem).with_suffix(".csv")
    rows = list(csv.reader(open(csv_path, encoding="utf-8-sig")))
    assert rows[-1][3] == "session" and rows[-1][8].startswith("suspended")

    # 再起動を模す: 新しいマネージャが中断セッションを見つける
    m2 = _mgr(logs, bus, settings, phrases)
    pend = m2.load_pending()
    total = settings.session.main_minutes * 60
    assert pend and pend["child"] == "はなちゃん" and pend["phase"] == "main" and pend["log_stem"] == s.log_stem
    assert 0 < pend["main_remaining_s"] < total - 0.25
    snap = m2.snapshot()
    assert snap["active"] is False and snap["pending"]["child"] == "はなちゃん"

    m2.resume()
    snap = m2.snapshot()
    assert snap["active"] and snap["phase"] == "main" and snap["intro_done"] == ["A1"] and snap["log_stem"] == s.log_stem
    assert snap["main_remaining_s"] < total - 0.25 and snap["since_last_utterance_s"] >= 0.3
    assert snap["pending"] is None if "pending" in snap else True
    rows = list(csv.reader(open(csv_path, encoding="utf-8-sig")))
    assert rows[-1][8].startswith("resumed after restart")
    assert rows[0] == CSV_COLUMNS and sum(1 for r in rows if r == CSV_COLUMNS) == 1  # ヘッダは 1 回だけ
    meta = json.loads(Path(s.log_stem).with_suffix(".json").read_text(encoding="utf-8"))
    assert "settings" in meta and meta["child"] == "はなちゃん"  # 開始時の写しはそのまま
    assert (logs / STATE_FILE).exists()
    m2.end()
    assert not (logs / STATE_FILE).exists()


def test_discard_expiry_and_new_session_supersedes(tmp_path):
    import time
    from pathlib import Path

    from app.session import RESUME_WINDOW_S, STATE_FILE

    settings, phrases, bus = Settings(), default_phrases(), EventBus()
    logs = tmp_path / "logs"

    def suspended_session(name):
        m = _mgr(logs, bus, settings, phrases)
        s = m.start(name, "ちゃん", "robot_first", "empathy")
        m.suspend()
        return Path(s.log_stem).with_suffix(".csv")

    def last_detail(csv_path):
        return list(csv.reader(open(csv_path, encoding="utf-8-sig")))[-1][8]

    # 破棄: 状態ファイルが消え、古い CSV に end 行が付く
    csv1 = suspended_session("はな")
    m = _mgr(logs, bus, settings, phrases)
    assert m.load_pending()
    m.discard("discarded by the experimenter")
    assert m.pending is None and not (logs / STATE_FILE).exists()
    assert last_detail(csv1) == "end (discarded by the experimenter)"
    assert m.snapshot()["pending"] is None

    # 30 分を超えた中断は候補にしない
    csv2 = suspended_session("ゆい")
    m = _mgr(logs, bus, settings, phrases)
    assert m.load_pending(now=time.time() + RESUME_WINDOW_S + 10) is None
    assert not (logs / STATE_FILE).exists() and last_detail(csv2) == "end (too old to resume)"

    # 再開せずに新しいセッションを始めたら、古い方は自動で閉じる
    csv3 = suspended_session("りく")
    m = _mgr(logs, bus, settings, phrases)
    assert m.load_pending()
    m.start("そら", "くん", "experimenter_first", "logical")
    assert m.pending is None and m.current is not None and m.current.child_display == "そらくん"
    assert last_detail(csv3) == "end (superseded by a new session)"
    with pytest.raises(SessionError):
        m.resume()
