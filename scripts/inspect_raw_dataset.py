from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def video_frames(path: Path) -> int:
    reader = imageio.get_reader(path)
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit raw VLA episode alignment and quality")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--max-idle-ratio", type=float, default=0.65)
    args = parser.parse_args()
    paths = sorted(args.dataset_root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not paths:
        raise FileNotFoundError(f"No episodes under {args.dataset_root}")
    report = []
    invalid = 0
    for path in paths:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        arrays = np.load(path / "data.npz")
        lengths = {
            key: len(arrays[key])
            for key in (
                "timestamp",
                "observation_state",
                "observation_velocity",
                "observation_ee_pose",
                "action",
                "phase",
            )
        }
        lengths["overview_video"] = video_frames(path / "overview.mp4")
        lengths["wrist_video"] = video_frames(path / "wrist_rgbd.mp4")
        aligned = len(set(lengths.values())) == 1
        finite = all(
            np.isfinite(arrays[key]).all()
            for key in (
                "timestamp",
                "observation_state",
                "observation_velocity",
                "observation_ee_pose",
                "action",
            )
        )
        action_delta = np.linalg.norm(np.diff(arrays["action"][:, :7], axis=0), axis=1)
        idle_ratio = float(np.mean(action_delta < 1.0e-4)) if len(action_delta) else 1.0
        timestamps_ok = bool(
            np.all(np.diff(arrays["timestamp"]) > 0)
            and np.allclose(
                np.diff(arrays["timestamp"]),
                1.0 / float(metadata["fps"]),
                atol=1.0e-6,
            )
        )
        valid = aligned and finite and timestamps_ok and idle_ratio <= args.max_idle_ratio
        invalid += int(not valid)
        report.append(
            {
                "episode": path.name,
                "success": metadata["success"],
                "valid": valid,
                "aligned": aligned,
                "finite": finite,
                "timestamps_ok": timestamps_ok,
                "idle_ratio": idle_ratio,
                "lengths": lengths,
            }
        )
    result = {
        "episodes": len(paths),
        "valid_episodes": len(paths) - invalid,
        "invalid_episodes": invalid,
        "details": report,
    }
    output = args.dataset_root / "audit.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if invalid == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


