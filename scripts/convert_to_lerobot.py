from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw MuJoCo episodes to LeRobot")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset id, e.g. user/panda-cup")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument(
        "--video-codec",
        default="h264",
        help="LeRobot video encoder; h264 is much faster than the AV1 default",
    )
    return parser.parse_args()


def read_video(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(path)
    try:
        return [np.asarray(frame) for frame in reader]
    finally:
        reader.close()


def main() -> int:
    args = parse_args()
    try:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is optional. Install this project with `pip install -e '.[dataset]'`."
        ) from exc

    episodes = sorted(args.raw_dir.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"))
    if not episodes:
        raise FileNotFoundError(f"No episodes found under {args.raw_dir}")
    first_metadata = json.loads((episodes[0] / "metadata.json").read_text(encoding="utf-8"))
    height, width, channels = first_metadata["image_shape_hwc"]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": first_metadata["state_names"],
        },
        "observation.velocity": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"joint_{i}_velocity" for i in range(1, 8)],
        },
        "observation.ee_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": first_metadata["action_names"],
        },
    }
    for camera in ("overview", "wrist_rgbd"):
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        }

    create_kwargs = {
        "repo_id": args.repo_id,
        "fps": int(first_metadata["fps"]),
        "robot_type": "franka_panda_mujoco",
        "features": features,
        "use_videos": True,
        "image_writer_threads": 4,
    }
    try:
        from lerobot.configs.video import RGBEncoderConfig

        create_kwargs["rgb_encoder"] = RGBEncoderConfig(
            vcodec=args.video_codec,
            crf=23,
            preset="veryfast" if args.video_codec == "h264" else None,
            video_backend="pyav",
        )
    except ImportError:
        pass
    if args.output_root is not None:
        create_kwargs["root"] = args.output_root
    dataset = LeRobotDataset.create(**create_kwargs)

    exported = 0
    for episode in episodes:
        metadata = json.loads((episode / "metadata.json").read_text(encoding="utf-8"))
        if not metadata["success"] and not args.include_failures:
            continue
        arrays = np.load(episode / "data.npz")
        videos = {
            camera: read_video(episode / f"{camera}.mp4")
            for camera in ("overview", "wrist_rgbd")
        }
        frame_count = len(arrays["action"])
        if any(len(frames) != frame_count for frames in videos.values()):
            raise RuntimeError(f"Video/action length mismatch in {episode}")
        for index in range(frame_count):
            dataset.add_frame(
                {
                    "observation.state": arrays["observation_state"][index],
                    "observation.velocity": arrays["observation_velocity"][index],
                    "observation.ee_pose": arrays["observation_ee_pose"][index],
                    "observation.images.overview": videos["overview"][index],
                    "observation.images.wrist_rgbd": videos["wrist_rgbd"][index],
                    "action": arrays["action"][index],
                    "task": metadata["task"],
                }
            )
        if "task" in inspect.signature(dataset.save_episode).parameters:
            dataset.save_episode(task=metadata["task"])
        else:
            dataset.save_episode()
        exported += 1
    if exported == 0:
        raise RuntimeError("No eligible successful episodes were found")
    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()
    if args.push_to_hub:
        dataset.push_to_hub()
    print(f"Exported {exported} episodes to {args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

