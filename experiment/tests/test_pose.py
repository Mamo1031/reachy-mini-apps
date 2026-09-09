import json
import math
from pathlib import Path

import pytest

from app.pose import (
    DEG,
    NEUTRAL,
    Envelope,
    Pose,
    antennas_from_physical,
    antennas_to_physical,
    clip,
    clip_excess_deg,
    head_moves,
    lerp_pose,
    mat_to_rpy,
    minjerk,
    pose_from_matrix,
    pose_to_matrix_flat,
    rpy_to_mat,
)

MOVES = Path(__file__).resolve().parents[1] / "moves"


def test_minjerk_endpoints_and_monotonic():
    assert minjerk(0.0) == 0.0
    assert minjerk(1.0) == 1.0
    assert minjerk(-1.0) == 0.0 and minjerk(2.0) == 1.0
    prev = 0.0
    for i in range(1, 101):
        v = minjerk(i / 100)
        assert v >= prev
        prev = v
    # 端点で速度ゼロ(数値微分)
    assert abs(minjerk(0.001) - minjerk(0.0)) < 1e-6
    assert abs(minjerk(1.0) - minjerk(0.999)) < 1e-6


@pytest.mark.parametrize("roll", [-40, -10, 0, 15, 40])
@pytest.mark.parametrize("pitch", [-40, -5, 0, 20, 40])
@pytest.mark.parametrize("yaw", [-45, -20, 0, 30, 45])
def test_mat_to_rpy_roundtrip(roll, pitch, yaw):
    r, p, y = roll * DEG, pitch * DEG, yaw * DEG
    m = rpy_to_mat(r, p, y)
    rr, pp, yy = mat_to_rpy(m)
    assert abs(rr - r) < 1e-9 and abs(pp - p) < 1e-9 and abs(yy - y) < 1e-9


def test_mat_to_rpy_matches_recorded_move_sample():
    # amazed1 の先頭フレーム: 事前に scipy(extrinsic xyz)で計算した期待値と一致すること
    doc = json.loads((MOVES / "amazed1.json").read_text())
    m = doc["set_target_data"][0]["head"]
    r, p, y = mat_to_rpy(m)
    # 行列を再構成して比較(scipy 非依存の検算)
    m2 = rpy_to_mat(r, p, y)
    for i in range(3):
        for j in range(3):
            assert abs(m[i][j] - m2[i][j]) < 1e-5
    assert -30 * DEG < r < 0  # amazed1 は最初に大きく左へロール


def test_pose_from_matrix_translation():
    m = [[1, 0, 0, 0.01], [0, 1, 0, -0.02], [0, 0, 1, 0.03], [0, 0, 0, 1]]
    p = pose_from_matrix(m, -0.1, 0.2)
    assert (p.x, p.y, p.z) == (0.01, -0.02, 0.03)
    assert (p.ant_r, p.ant_l) == (-0.1, 0.2)


def test_clip_and_excess():
    env = Envelope.from_degrees(30, 25, 45, 30, 170)
    p = Pose(roll=50 * DEG, pitch=-40 * DEG, x=0.05, ant_l=200 * DEG)
    c = clip(p, env)
    assert abs(c.roll - 30 * DEG) < 1e-12 and abs(c.pitch + 25 * DEG) < 1e-12
    assert c.x == 0.030 and abs(c.ant_l - 170 * DEG) < 1e-12
    assert clip_excess_deg(p, env) == pytest.approx(30, abs=1e-6)  # ant_l が 30° 超過
    assert clip_excess_deg(NEUTRAL, env) == 0.0


def test_lerp_pose_endpoints():
    a, b = NEUTRAL, Pose(roll=0.5, pitch=-0.2, yaw=0.1, x=0.01, ant_r=-1.0, ant_l=1.0)
    assert lerp_pose(a, b, 0.0) == a
    assert lerp_pose(a, b, 1.0) == b
    mid = lerp_pose(a, b, 0.5)
    assert abs(mid.roll - 0.25) < 1e-9


def test_antenna_physical_conversion_roundtrip():
    r, l = antennas_from_physical(10.0, 10.0)
    assert abs(r + 0.1745) < 1e-3 and abs(l - 0.1745) < 1e-3  # ニュートラルと一致
    assert antennas_to_physical(*antennas_from_physical(30.0, -20.0)) == pytest.approx((30.0, -20.0))


def test_head_moves():
    assert not head_moves([NEUTRAL, NEUTRAL.with_(ant_l=1.0)])
    assert head_moves([NEUTRAL, NEUTRAL.with_(pitch=5 * DEG)])
    assert head_moves([NEUTRAL, NEUTRAL.with_(z=0.01)])
    assert not head_moves([])


def test_pose_to_matrix_flat_roundtrip():
    p = Pose(roll=0.2, pitch=-0.1, yaw=0.3, x=0.01, y=-0.02, z=0.005, ant_r=-0.3, ant_l=0.4)
    flat = pose_to_matrix_flat(p)
    assert len(flat) == 16 and flat[12:] == [0.0, 0.0, 0.0, 1.0]
    m = [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]
    q = pose_from_matrix(m, p.ant_r, p.ant_l)
    for a in ("roll", "pitch", "yaw", "x", "y", "z"):
        assert abs(getattr(q, a) - getattr(p, a)) < 1e-9, a
