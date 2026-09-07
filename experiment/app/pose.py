"""姿勢の表現・単位変換・補間・安全範囲。

内部と通信はラジアン / メートル。人が編集するファイル(settings.json, gestures.json)は
度 / mm で、変換はこのモジュールに集約する。

アンテナはロボットへ送る生の値(`[右, 左]`、左右で符号が鏡像)を `ant_r` / `ant_l` に保持する。
ジェスチャー定義では「物理角度」(左右共通の符号、ニュートラル = +10°、折りたたみ = +175°)で
書き、`antennas_from_physical()` で生の値に変換する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Sequence

ANT_NEUTRAL_PHYSICAL = 0.1745  # rad(≒10°)。デーモンの INIT_ANTENNAS_JOINT_POSITIONS に一致
DEG = math.pi / 180.0


@dataclass(frozen=True)
class Pose:
    """頭部 6 自由度 + アンテナ 2 本。rad / m。"""

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    ant_r: float = -ANT_NEUTRAL_PHYSICAL
    ant_l: float = ANT_NEUTRAL_PHYSICAL

    def with_(self, **kw: float) -> "Pose":
        return replace(self, **kw)


NEUTRAL = Pose()


@dataclass(frozen=True)
class Envelope:
    """安全範囲(絶対値の上限)。rad / m。"""

    roll: float = 30 * DEG
    pitch: float = 25 * DEG
    yaw: float = 45 * DEG
    xyz: float = 0.030
    antenna: float = 170 * DEG

    @classmethod
    def from_degrees(
        cls, roll_deg: float, pitch_deg: float, yaw_deg: float, xyz_mm: float, antenna_deg: float
    ) -> "Envelope":
        return cls(
            roll=roll_deg * DEG,
            pitch=pitch_deg * DEG,
            yaw=yaw_deg * DEG,
            xyz=xyz_mm / 1000.0,
            antenna=antenna_deg * DEG,
        )


def _clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def clip(p: Pose, env: Envelope) -> Pose:
    return Pose(
        roll=_clamp(p.roll, env.roll),
        pitch=_clamp(p.pitch, env.pitch),
        yaw=_clamp(p.yaw, env.yaw),
        x=_clamp(p.x, env.xyz),
        y=_clamp(p.y, env.xyz),
        z=_clamp(p.z, env.xyz),
        ant_r=_clamp(p.ant_r, env.antenna),
        ant_l=_clamp(p.ant_l, env.antenna),
    )


def clip_excess_deg(p: Pose, env: Envelope) -> float:
    """安全範囲を超えている最大量(度)。並進は 1 mm = 1° 相当として扱う(警告用)。"""
    c = clip(p, env)
    ang = max(abs(p.roll - c.roll), abs(p.pitch - c.pitch), abs(p.yaw - c.yaw), abs(p.ant_r - c.ant_r), abs(p.ant_l - c.ant_l)) / DEG
    lin = max(abs(p.x - c.x), abs(p.y - c.y), abs(p.z - c.z)) * 1000.0
    return max(ang, lin)


def minjerk(s: float) -> float:
    """最小躍度の S 字(0→1、端点で速度・加速度 0)。"""
    s = 0.0 if s < 0.0 else 1.0 if s > 1.0 else s
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def lerp_pose(a: Pose, b: Pose, s: float, ease=minjerk) -> Pose:
    t = ease(s)
    return Pose(
        roll=a.roll + (b.roll - a.roll) * t,
        pitch=a.pitch + (b.pitch - a.pitch) * t,
        yaw=a.yaw + (b.yaw - a.yaw) * t,
        x=a.x + (b.x - a.x) * t,
        y=a.y + (b.y - a.y) * t,
        z=a.z + (b.z - a.z) * t,
        ant_r=a.ant_r + (b.ant_r - a.ant_r) * t,
        ant_l=a.ant_l + (b.ant_l - a.ant_l) * t,
    )


def mat_to_rpy(m: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """4x4(または 3x3)回転行列 → (roll, pitch, yaw)。

    デーモンと同じ extrinsic xyz(R = Rz(yaw)·Ry(pitch)·Rx(roll))。
    """
    r00, r10, r20 = m[0][0], m[1][0], m[2][0]
    r21, r22 = m[2][1], m[2][2]
    pitch = math.atan2(-r20, math.hypot(r00, r10))
    if abs(math.cos(pitch)) < 1e-9:  # ジンバルロック(実データでは起きない)
        roll = math.atan2(-m[1][2], m[1][1])
        yaw = 0.0
    else:
        roll = math.atan2(r21, r22)
        yaw = math.atan2(r10, r00)
    return roll, pitch, yaw


def rpy_to_mat(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    """(roll, pitch, yaw) → 3x3 回転行列(テストと検算用)。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def pose_from_matrix(m: Sequence[Sequence[float]], ant_r: float, ant_l: float) -> Pose:
    roll, pitch, yaw = mat_to_rpy(m)
    return Pose(roll=roll, pitch=pitch, yaw=yaw, x=m[0][3], y=m[1][3], z=m[2][3], ant_r=ant_r, ant_l=ant_l)


def antennas_from_physical(right_deg: float, left_deg: float) -> tuple[float, float]:
    """物理角度(度、左右共通の符号)→ ロボットへ送る生の値 [右, 左](rad)。"""
    return -right_deg * DEG, left_deg * DEG


def antennas_to_physical(ant_r: float, ant_l: float) -> tuple[float, float]:
    return -ant_r / DEG, ant_l / DEG


def head_payload(p: Pose) -> dict[str, float]:
    return {"x": p.x, "y": p.y, "z": p.z, "roll": p.roll, "pitch": p.pitch, "yaw": p.yaw}


def antennas_payload(p: Pose) -> list[float]:
    return [p.ant_r, p.ant_l]


def head_distance(a: Pose, b: Pose) -> float:
    """頭部の姿勢差(rad 換算の最大値。並進は 1 mm = 1° として合算)。"""
    ang = max(abs(a.roll - b.roll), abs(a.pitch - b.pitch), abs(a.yaw - b.yaw))
    lin = max(abs(a.x - b.x), abs(a.y - b.y), abs(a.z - b.z)) * 1000.0 * DEG
    return max(ang, lin)


def antenna_distance(a: Pose, b: Pose) -> float:
    return max(abs(a.ant_r - b.ant_r), abs(a.ant_l - b.ant_l))


def head_moves(poses: Iterable[Pose], eps_rad: float = 0.5 * DEG, eps_m: float = 0.002) -> bool:
    """頭部が動く軌道か(最初の姿勢からの最大偏差で判定)。"""
    it = iter(poses)
    first = next(it, None)
    if first is None:
        return False
    for p in it:
        if (
            abs(p.roll - first.roll) > eps_rad
            or abs(p.pitch - first.pitch) > eps_rad
            or abs(p.yaw - first.yaw) > eps_rad
            or abs(p.x - first.x) > eps_m
            or abs(p.y - first.y) > eps_m
            or abs(p.z - first.z) > eps_m
        ):
            return True
    return False
