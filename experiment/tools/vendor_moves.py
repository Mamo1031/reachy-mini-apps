"""公式モーション集(Hugging Face キャッシュ)から使うものだけを moves/ にコピーする。

使い方: cd experiment && uv run python tools/vendor_moves.py
ロボット(またはこの Mac の公式アプリ)が一度でもデータセットを取得していれば
~/.cache/huggingface/hub/ にある。無ければ `huggingface-cli download` で取得すること。
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.pose import DEG, mat_to_rpy  # noqa: E402

HF_HUB = Path.home() / ".cache" / "huggingface" / "hub"
DATASETS = {
    "emotions": "pollen-robotics/reachy-mini-emotions-library",
    "dances": "pollen-robotics/reachy-mini-dances-library",
}
# ジェスチャーに使うモーション(dataset, name)
MOVES = [
    ("emotions", "welcoming1"),
    ("emotions", "welcoming2"),
    ("emotions", "cheerful1"),
    ("emotions", "success1"),
    ("emotions", "enthusiastic1"),
    ("emotions", "serenity1"),
    ("emotions", "downcast1"),
    ("emotions", "understanding1"),
    ("emotions", "understanding2"),
    ("emotions", "amazed1"),
    ("emotions", "come1"),
    ("emotions", "no_excited1"),
    ("emotions", "inquiring2"),
    ("emotions", "inquiring3"),
    ("emotions", "thoughtful1"),
]
DEST = Path(__file__).resolve().parents[1] / "moves"


def snapshot_dir(repo: str) -> tuple[Path, str]:
    base = HF_HUB / ("datasets--" + repo.replace("/", "--"))
    ref = (base / "refs" / "main").read_text().strip()
    return base / "snapshots" / ref, ref


def stats(doc: dict) -> str:
    frames = doc["set_target_data"]
    rpy = [mat_to_rpy(f["head"]) for f in frames]
    xyz = [(f["head"][0][3], f["head"][1][3], f["head"][2][3]) for f in frames]
    ants = [f["antennas"] for f in frames]
    yaws = [f.get("body_yaw", 0.0) or 0.0 for f in frames]

    def rng(vals):
        return f"{min(vals) / DEG:6.1f}..{max(vals) / DEG:6.1f}"

    first, last = rpy[0], rpy[-1]
    return (
        f"dur={doc['time'][-1]:5.2f}s n={len(frames):3d} "
        f"roll[{rng([r for r, _, _ in rpy])}] pitch[{rng([p for _, p, _ in rpy])}] yaw[{rng([y for _, _, y in rpy])}] "
        f"z0={xyz[0][2] * 1000:5.1f}mm z[{min(v[2] for v in xyz) * 1000:5.1f}..{max(v[2] for v in xyz) * 1000:5.1f}] "
        f"ant0=({ants[0][0] / DEG:6.1f},{ants[0][1] / DEG:6.1f}) antN=({ants[-1][0] / DEG:6.1f},{ants[-1][1] / DEG:6.1f}) "
        f"body_yaw[{rng(yaws)}] first_rpy=({first[0] / DEG:.1f},{first[1] / DEG:.1f},{first[2] / DEG:.1f}) "
        f"last_rpy=({last[0] / DEG:.1f},{last[1] / DEG:.1f},{last[2] / DEG:.1f})"
    )


def main() -> int:
    DEST.mkdir(exist_ok=True)
    credits = ["# 公式モーション集の出典\n", "以下のファイルは Pollen Robotics 公開のデータセット(Apache-2.0)からコピーしたものです。\n"]
    ok = True
    for key, repo in DATASETS.items():
        try:
            snap, ref = snapshot_dir(repo)
        except FileNotFoundError:
            print(f"見つかりません: {repo}(~/.cache/huggingface/hub にキャッシュがありません)")
            ok = False
            continue
        credits.append(f"\n- https://huggingface.co/datasets/{repo} (revision {ref})\n")
        for ds, name in MOVES:
            if ds != key:
                continue
            src = snap / f"{name}.json"
            if not src.exists():
                print(f"無い: {src}")
                ok = False
                continue
            shutil.copyfile(src, DEST / f"{name}.json")
            doc = json.loads(src.read_text())
            credits.append(f"  - `{name}.json` — {doc.get('description', '').strip()}\n")
            print(f"{name:16s} {stats(doc)}")
    (DEST / "CREDITS.md").write_text("".join(credits), encoding="utf-8")
    print(f"→ {DEST}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
