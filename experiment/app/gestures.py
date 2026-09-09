"""ジェスチャー定義 → 軌道(時刻付き Pose 列)の生成。

3 種類の元データを同じ Trajectory に変換する:
- keyframes: パラメトリックなキーフレーム(度 / mm / 物理アンテナ角)を最小躍度で補間
- recorded : 公式モーション集の JSON(50 Hz の 4x4 行列)をそのまま(区間切り出し・速度変更可)
- point    : 設定の位置(yaw/pitch)へ頭を向けて頷く
- sequence : 複数のジェスチャーを連結
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import GestureDef, Gestures, Keyframe, Position, Settings
from .pose import DEG, NEUTRAL, Envelope, Pose, antennas_from_physical, clip, clip_excess_deg, head_moves, lerp_pose, pose_from_matrix

log = logging.getLogger(__name__)


class GestureError(Exception):
    pass


@dataclass
class Trajectory:
    name: str
    frames: list[tuple[float, Pose]]  # (秒, Pose)。時刻は 0 から単調増加
    moves_head: bool

    @property
    def duration(self) -> float:
        return self.frames[-1][0] if self.frames else 0.0

    @property
    def first(self) -> Pose:
        return self.frames[0][1] if self.frames else NEUTRAL

    @property
    def last(self) -> Pose:
        return self.frames[-1][1] if self.frames else NEUTRAL

    def pose_at(self, t: float) -> Pose:
        """時刻 t の姿勢(フレーム間は線形補間、範囲外は端の姿勢)。"""
        if not self.frames:
            return NEUTRAL
        if t <= self.frames[0][0]:
            return self.frames[0][1]
        if t >= self.frames[-1][0]:
            return self.frames[-1][1]
        times = self._times()
        i = bisect.bisect_right(times, t)
        t0, p0 = self.frames[i - 1]
        t1, p1 = self.frames[i]
        if t1 <= t0:
            return p1
        return lerp_pose(p0, p1, (t - t0) / (t1 - t0), ease=lambda s: s)

    def _times(self) -> list[float]:
        cache = getattr(self, "_times_cache", None)
        if cache is None or len(cache) != len(self.frames):
            cache = [t for t, _ in self.frames]
            self._times_cache = cache  # type: ignore[attr-defined]
        return cache


# ---------------------------------------------------------------- builders


def _apply_head(base: Pose, head: dict[str, float] | None) -> Pose:
    if not head:
        return base
    kw: dict[str, float] = {}
    for k, v in head.items():
        if k in ("roll", "pitch", "yaw"):
            kw[k] = float(v) * DEG
        elif k in ("x", "y", "z"):
            kw[k] = float(v) / 1000.0
        else:
            raise GestureError(f"キーフレームの head に未知のキー: {k}")
    return base.with_(**kw)


def _apply_antennas(base: Pose, antennas: list[float] | None) -> Pose:
    if antennas is None:
        return base
    if len(antennas) != 2:
        raise GestureError("antennas は [右, 左] の 2 要素で指定してください")
    r, l = antennas_from_physical(float(antennas[0]), float(antennas[1]))
    return base.with_(ant_r=r, ant_l=l)


def from_keyframes(frames: list[Keyframe], hz: float, env: Envelope, start: Pose = NEUTRAL) -> list[tuple[float, Pose]]:
    dt = 1.0 / hz
    out: list[tuple[float, Pose]] = [(0.0, clip(start, env))]
    t = 0.0
    cur = start
    for kf in frames:
        if kf.hold is not None:
            n = max(1, int(round(kf.hold * hz)))
            for _ in range(n):
                t += dt
                out.append((t, clip(cur, env)))
            continue
        target = _apply_antennas(_apply_head(cur, kf.head), kf.antennas)
        d = max(kf.d, dt)
        n = max(1, int(round(d * hz)))
        for i in range(1, n + 1):
            t += dt
            out.append((t, clip(lerp_pose(cur, target, i / n), env)))
        cur = target
    return out


def from_recorded(
    doc: dict,
    env: Envelope,
    *,
    start: float | None = None,
    end: float | None = None,
    speed: float = 1.0,
    normalize_xyz: bool = True,
) -> list[tuple[float, Pose]]:
    times: list[float] = doc["time"]
    data: list[dict] = doc["set_target_data"]
    if not times or len(times) != len(data):
        raise GestureError("モーション JSON の time と set_target_data の長さが一致しません")
    speed = speed if speed > 0 else 1.0
    lo = start if start is not None else times[0]
    hi = end if end is not None else times[-1]
    idx = [i for i, t in enumerate(times) if lo - 1e-9 <= t <= hi + 1e-9]
    if not idx:
        raise GestureError("指定区間にフレームがありません")
    t_base = times[idx[0]]
    out: list[tuple[float, Pose]] = []
    ox = oy = oz = 0.0
    for n, i in enumerate(idx):
        fr = data[i]
        ant = fr.get("antennas") or [NEUTRAL.ant_r, NEUTRAL.ant_l]
        p = pose_from_matrix(fr["head"], float(ant[0]), float(ant[1]))
        if n == 0 and normalize_xyz:
            ox, oy, oz = p.x, p.y, p.z
        p = p.with_(x=p.x - ox, y=p.y - oy, z=p.z - oz)
        out.append(((times[i] - t_base) / speed, clip(p, env)))
    return out


def point_frames(target: Position, g: GestureDef, hz: float, env: Envelope) -> list[tuple[float, Pose]]:
    yaw, pitch = target.yaw_deg, target.pitch_deg
    kfs = [
        Keyframe(head={"yaw": yaw, "pitch": pitch}, d=g.turn_s),
        Keyframe(hold=g.hold_s),
        Keyframe(head={"pitch": pitch + g.nod_deg}, d=0.3),
        Keyframe(head={"pitch": pitch}, d=0.3),
        Keyframe(hold=0.4),
    ]
    return from_keyframes(kfs, hz, env)


# ---------------------------------------------------------------- library


class GestureLibrary:
    def __init__(self, gestures_ref: Callable[[], Gestures], settings_ref: Callable[[], Settings], moves_dir: Path) -> None:
        self.gestures_ref = gestures_ref
        self.settings_ref = settings_ref
        self.moves_dir = moves_dir
        self._docs: dict[str, dict] = {}

    # ------------------------------------------------------------ helpers
    def _env(self) -> Envelope:
        e = self.settings_ref().motion.envelope
        return Envelope.from_degrees(e.roll_deg, e.pitch_deg, e.yaw_deg, e.xyz_mm, e.antenna_deg)

    def _hz(self) -> float:
        return max(10.0, float(self.settings_ref().motion.stream_hz))

    def load_doc(self, file: str) -> dict:
        if file not in self._docs:
            p = self.moves_dir / f"{file}.json"
            if not p.exists():
                raise GestureError(f"モーションファイルがありません: {p.name}")
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError) as e:
                raise GestureError(f"モーションファイルを読めません: {p.name}: {e}") from e
            if not isinstance(doc, dict) or "time" not in doc or "set_target_data" not in doc:
                raise GestureError(f"モーションファイルの形式が不正です: {p.name}")
            self._docs[file] = doc
        return self._docs[file]

    def names(self) -> list[str]:
        return self.gestures_ref().names()

    # ------------------------------------------------------------ build
    def build(self, name: str, _depth: int = 0) -> Trajectory:
        g = self.gestures_ref().get(name)
        if g is None:
            raise GestureError(f"ジェスチャーがありません: {name}")
        if _depth > 3:
            raise GestureError(f"ジェスチャーの入れ子が深すぎます: {name}")
        env, hz = self._env(), self._hz()
        if g.kind == "keyframes":
            frames = from_keyframes(g.frames, hz, env)
        elif g.kind == "recorded":
            if not g.file:
                raise GestureError(f"{name}: file が未指定です")
            try:
                frames = from_recorded(self.load_doc(g.file), env, start=g.start, end=g.end, speed=g.speed, normalize_xyz=g.normalize_xyz)
            except (KeyError, IndexError, TypeError, ValueError) as e:  # データ欠損・型違い
                raise GestureError(f"{name}: モーション {g.file} のデータが不正です({type(e).__name__}: {e})") from e
        elif g.kind == "point":
            positions = self.settings_ref().positions
            if not g.target or g.target not in positions:
                raise GestureError(f"{name}: 位置 '{g.target}' が設定にありません")
            frames = point_frames(positions[g.target], g, hz, env)
        elif g.kind == "sequence":
            frames = []
            t_off = 0.0
            for step in g.steps:
                sub = self.build(step, _depth + 1)
                if frames:
                    # 前のジェスチャーの終端から次の始端へ 0.35 秒(最小躍度)でつなぐ
                    prev = frames[-1][1]
                    n = max(1, int(round(0.35 * hz)))
                    for i in range(1, n + 1):
                        t_off += 1.0 / hz
                        frames.append((t_off, lerp_pose(prev, sub.first, i / n)))
                base = t_off
                for t, p in sub.frames:
                    frames.append((base + t, p))
                t_off = frames[-1][0]
        else:
            raise GestureError(f"{name}: 未知の kind {g.kind}")
        if not frames:
            raise GestureError(f"{name}: フレームが空です")
        return Trajectory(name=name, frames=frames, moves_head=head_moves(p for _, p in frames))

    # ------------------------------------------------------------ validation
    def validate(self) -> list[str]:
        """全ジェスチャーを組み立てて問題を列挙する(起動時と設定保存時に呼ぶ)。"""
        warnings: list[str] = []
        env = self._env()
        gestures = self.gestures_ref()
        for name in gestures.names():
            g = gestures.get(name)
            assert g is not None
            try:
                traj = self.build(name)
            except GestureError as e:
                warnings.append(f"{name}: {e}")
                continue
            # クリップ量は raw データで評価する(build 済みは既にクリップされている)
            if g.kind == "recorded" and g.file:
                raw = from_recorded(self.load_doc(g.file), Envelope(roll=math.inf, pitch=math.inf, yaw=math.inf, xyz=math.inf, antenna=math.inf), start=g.start, end=g.end, speed=g.speed, normalize_xyz=g.normalize_xyz)
                excess = max(clip_excess_deg(p, env) for _, p in raw)
                if excess > 2.0:
                    warnings.append(f"{name}: 安全範囲で {excess:.0f}° 分クリップされます(設定 > 動作 > 可動域)")
            if traj.duration > 8.0:
                warnings.append(f"{name}: {traj.duration:.1f} 秒と長めです")
        return warnings
