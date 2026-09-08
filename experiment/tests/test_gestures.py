from pathlib import Path

import pytest

from app.config import DATA_DIR, Keyframe, Settings, default_gestures
from app.gestures import GestureError, GestureLibrary, Trajectory, from_keyframes, from_recorded
from app.pose import DEG, NEUTRAL, Envelope

MOVES = DATA_DIR / "moves"


@pytest.fixture
def lib():
    settings = Settings()
    gestures = default_gestures()
    return GestureLibrary(lambda: gestures, lambda: settings, MOVES)


def _max_step(frames):
    """隣接フレーム間の頭部角度の最大変化量(rad)。アンテナは公式データが速いので別扱い。"""
    worst = 0.0
    for (_, a), (_, b) in zip(frames, frames[1:]):
        worst = max(worst, abs(a.pitch - b.pitch), abs(a.yaw - b.yaw), abs(a.roll - b.roll))
    return worst


def _max_antenna_step(frames):
    worst = 0.0
    for (_, a), (_, b) in zip(frames, frames[1:]):
        worst = max(worst, abs(a.ant_r - b.ant_r), abs(a.ant_l - b.ant_l))
    return worst


def test_keyframes_timing_continuity_and_hold():
    env = Envelope()
    frames = from_keyframes([Keyframe(head={"pitch": 8}, d=0.2), Keyframe(hold=0.5), Keyframe(head={"pitch": 0}, d=0.3)], hz=50, env=env)
    assert frames[0][1] == NEUTRAL
    assert abs(frames[-1][0] - 1.0) < 1e-6
    # ピークで 8°、hold 中は不変
    peak = max(p.pitch for _, p in frames)
    assert abs(peak - 8 * DEG) < 1e-9
    hold = [p.pitch for t, p in frames if 0.21 <= t <= 0.69]
    assert all(abs(v - peak) < 1e-12 for v in hold)
    # 端点で速度ゼロ(最初と最後の変化量が小さい)
    assert abs(frames[1][1].pitch - frames[0][1].pitch) < 0.1 * DEG
    assert _max_step(frames) < 2.0 * DEG  # 8° を 0.2 秒で動かす最小躍度のピーク速度 ≒ 1.5°/フレーム


def test_keyframes_antennas_physical_and_unknown_key():
    env = Envelope()
    frames = from_keyframes([Keyframe(antennas=[25, 25], d=0.2)], hz=50, env=env)
    last = frames[-1][1]
    assert abs(last.ant_r + 25 * DEG) < 1e-9 and abs(last.ant_l - 25 * DEG) < 1e-9
    assert last.pitch == 0.0  # 頭は前値保持
    with pytest.raises(GestureError):
        from_keyframes([Keyframe(head={"bogus": 1})], hz=50, env=env)


def test_keyframes_clipped_by_envelope():
    env = Envelope.from_degrees(30, 25, 45, 30, 170)
    frames = from_keyframes([Keyframe(head={"pitch": 60}, d=0.3)], hz=50, env=env)
    assert abs(frames[-1][1].pitch - 25 * DEG) < 1e-9


def test_recorded_move_normalization_trim_and_speed():
    import json

    doc = json.loads((MOVES / "cheerful1.json").read_text())
    env = Envelope()
    frames = from_recorded(doc, env)
    assert len(frames) == len(doc["time"])
    assert abs(frames[-1][0] - doc["time"][-1]) < 1e-9
    # cheerful1 は z=20mm 一定 → 正規化で 0 になる
    assert all(abs(p.z) < 1e-9 for _, p in frames)
    # アンテナは生の値をそのまま
    assert frames[0][1].ant_r == pytest.approx(doc["set_target_data"][0]["antennas"][0])
    trimmed = from_recorded(doc, env, start=0.5, end=1.5, speed=2.0)
    assert trimmed[0][0] == 0.0 and abs(trimmed[-1][0] - 0.5) < 0.02
    raw = from_recorded(doc, env, normalize_xyz=False)
    assert abs(raw[0][1].z - 0.02) < 1e-6


def test_library_builds_every_default_gesture_without_warnings(lib):
    warnings = lib.validate()
    assert warnings == []
    for name in lib.names():
        traj = lib.build(name)
        assert isinstance(traj, Trajectory) and traj.duration > 0
        assert _max_step(traj.frames) < 6 * DEG, name  # 頭は 1 フレームで 6° 以上飛ばない
        assert _max_antenna_step(traj.frames) < 15 * DEG, name


def test_point_uses_settings_position(lib):
    traj = lib.build("point_sample")
    assert traj.moves_head
    yaws = [p.yaw for _, p in traj.frames]
    assert min(yaws) == pytest.approx(-35 * DEG, abs=1e-6)  # 右 = 負
    assert max(p.pitch for _, p in traj.frames) == pytest.approx((10 + 8) * DEG, abs=1e-6)


def test_sequence_concatenates(lib):
    a = lib.build("happy_lean")
    b = lib.build("antenna_clap")
    seq = lib.build("happy_lean_clap")
    assert seq.duration == pytest.approx(a.duration + 0.2 + b.duration, abs=0.05)
    assert _max_step(seq.frames) < 6 * DEG


def test_antenna_only_gesture_does_not_move_head(lib):
    traj = lib.build("antenna_twitch")
    assert not traj.moves_head


def test_missing_file_and_target_are_reported():
    from app.config import GestureDef, Gestures

    settings = Settings()
    gestures = Gestures({"bad_file": GestureDef(kind="recorded", file="nope"), "bad_target": GestureDef(kind="point", target="moon")})
    lib = GestureLibrary(lambda: gestures, lambda: settings, MOVES)
    warns = lib.validate()
    assert any("bad_file" in w for w in warns) and any("bad_target" in w for w in warns)


def test_pose_at_interpolates():
    traj = Trajectory("t", [(0.0, NEUTRAL), (1.0, NEUTRAL.with_(pitch=1.0))], True)
    assert traj.pose_at(-1).pitch == 0.0
    assert traj.pose_at(0.5).pitch == pytest.approx(0.5)
    assert traj.pose_at(5).pitch == 1.0
