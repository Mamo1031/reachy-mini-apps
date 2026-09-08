import json

from app.config import (
    Gestures,
    Phrases,
    Settings,
    default_gestures,
    default_phrases,
    expand,
    load_or_create,
    save_atomic,
)


def test_expand_placeholders():
    t = "{child}っていうんだね！ わたしは{robot}。{experimenter}と{unknown}"
    out = expand(t, robot="ドラちゃん", child="はなちゃん", experimenter="はるかお姉さん")
    assert out == "はなちゃんっていうんだね！ わたしはドラちゃん。はるかお姉さんと{unknown}"
    assert expand("波括弧なし") == "波括弧なし"


def test_default_phrases_leaves_and_sequences():
    ph = default_phrases()
    leaves = ph.leaves()
    assert set(leaves) >= {"A1", "A2", "A3", "A4", "A5-1", "A5-2", "A6-1", "A6-2", "A6-3"}
    assert [k for k in leaves if leaves[k].category == "empathy"] == [f"B{i}" for i in range(1, 11)]
    assert [k for k in leaves if leaves[k].category == "logical"] == [f"C{i}" for i in range(1, 11)]
    assert [leaves[k].text for k in ("BC1", "BC2", "BC3", "BC4")] == ["うん", "ちがうよ", "そうだね", "できたね"]
    assert ph.intro_sequence("robot_first") == ["A1", "A2", "A3", "A4", "A5-1", "A5-2"]
    assert ph.intro_sequence("experimenter_first") == ["A1", "A2", "A6-1", "A6-2", "A6-3"]
    # 名前は A2 だけで使う
    assert [k for k, v in leaves.items() if "{child}" in v.text] == ["A2"]


def test_every_phrase_gesture_exists():
    gestures = default_gestures()
    names = set(gestures.names())
    missing = {k: v.gesture for k, v in default_phrases().leaves().items() if v.gesture not in names}
    assert missing == {}
    for g in gestures.root.values():
        for step in g.steps:
            assert step in names


def test_settings_default_fill_and_unknown_keys(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"names": {"robot": "ポチ"}, "bogus": 1, "motion": {"idle": {"interval_s": 20}}}), encoding="utf-8")
    s, warn = load_or_create(p, Settings, Settings())
    assert warn is None
    assert s.names.robot == "ポチ" and s.names.experimenter == "はるかお姉さん"
    assert s.motion.idle.interval_s == 20 and s.motion.idle.enabled is True
    assert s.positions["sample"].label == "お手本"


def test_load_or_create_writes_defaults_and_atomic_save(tmp_path):
    p = tmp_path / "phrases.json"
    ph, warn = load_or_create(p, Phrases, default_phrases())
    assert p.exists() and warn is None
    assert not list(tmp_path.glob("*.tmp"))
    ph2 = Phrases.model_validate(json.loads(p.read_text(encoding="utf-8")))
    assert ph2.leaves().keys() == ph.leaves().keys()
    save_atomic(p, ph2)
    assert not list(tmp_path.glob("*.tmp"))


def test_broken_json_is_quarantined(tmp_path):
    p = tmp_path / "gestures.json"
    p.write_text("{not json", encoding="utf-8")
    g, warn = load_or_create(p, Gestures, default_gestures())
    assert warn and "デフォルト" in warn
    assert list(tmp_path.glob("gestures.json.broken-*"))
    assert "nod_small" in g.root
    assert Gestures.model_validate(json.loads(p.read_text(encoding="utf-8"))).get("nod_small")
