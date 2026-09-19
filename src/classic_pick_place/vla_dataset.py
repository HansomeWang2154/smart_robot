from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
import tempfile

import imageio.v2 as imageio
import mujoco
import numpy as np


STATE_NAMES = [f"joint_{index}" for index in range(1, 8)] + ["gripper"]


@dataclass(frozen=True)
class EpisodeSummary:
    path: Path
    frames: int
    duration_s: float
    success: bool


class RawVlaEpisodeRecorder:
    """Record aligned Panda sensorimotor episodes before LeRobot conversion.

    The raw layer intentionally has no LeRobot dependency. This keeps MuJoCo
    collection lightweight and makes interrupted episodes easy to inspect.
    `scripts/convert_to_lerobot.py` performs the standardized export.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        root: Path,
        *,
        fps: int = 20,
        width: int = 320,
        height: int = 240,
        task: str = "pick up the blue cup and place it on the green target",
        seed: int = 0,
        source: str = "teleoperation",
    ):
        self.model = model
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        self.width = int(width)
        self.height = int(height)
        self.task = task
        self.seed = int(seed)
        self.source = source
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.temp_path = Path(tempfile.mkdtemp(prefix=".recording_", dir=self.root))
        self.video_writers = {
            camera: imageio.get_writer(
                self.temp_path / f"{camera}.mp4",
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            for camera in ("overview", "wrist_rgbd")
        }
        self.timestamps: list[float] = []
        self.sim_times: list[float] = []
        self.states: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.ee_poses: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.phases: list[str] = []
        self.closed = False

    def add_frame(
        self,
        data: mujoco.MjData,
        *,
        arm_qpos: np.ndarray,
        arm_dofs: np.ndarray,
        finger_qpos: np.ndarray,
        ee_position: np.ndarray,
        ee_rotation: np.ndarray,
        q_target: np.ndarray,
        gripper_open: bool,
        phase: str,
    ) -> None:
        if self.closed:
            raise RuntimeError("Cannot append to a closed episode")
        for camera, writer in self.video_writers.items():
            self.renderer.update_scene(data, camera=camera)
            writer.append_data(self.renderer.render())

        gripper_position = float(
            np.clip(np.mean(data.qpos[finger_qpos]) / 0.04, 0.0, 1.0)
        )
        state = np.concatenate(
            (data.qpos[arm_qpos].copy(), np.array([gripper_position]))
        ).astype(np.float32)
        velocity = data.qvel[arm_dofs].copy().astype(np.float32)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, np.asarray(ee_rotation).ravel())
        ee_pose = np.concatenate((ee_position, quaternion)).astype(np.float32)
        action = np.concatenate(
            (np.asarray(q_target), np.array([1.0 if gripper_open else 0.0]))
        ).astype(np.float32)
        frame_index = len(self.timestamps)
        self.timestamps.append(frame_index / self.fps)
        self.sim_times.append(float(data.time))
        self.states.append(state)
        self.velocities.append(velocity)
        self.ee_poses.append(ee_pose)
        self.actions.append(action)
        self.phases.append(phase)

    def _close_streams(self) -> None:
        if self.closed:
            return
        for writer in self.video_writers.values():
            writer.close()
        self.renderer.close()
        self.closed = True

    def save(
        self,
        *,
        success: bool,
        metrics: dict[str, object] | None = None,
    ) -> EpisodeSummary:
        if not self.states:
            raise RuntimeError("Refusing to save an empty episode")
        self._close_streams()
        np.savez_compressed(
            self.temp_path / "data.npz",
            timestamp=np.asarray(self.timestamps, dtype=np.float64),
            sim_time=np.asarray(self.sim_times, dtype=np.float64),
            observation_state=np.stack(self.states),
            observation_velocity=np.stack(self.velocities),
            observation_ee_pose=np.stack(self.ee_poses),
            action=np.stack(self.actions),
            phase=np.asarray(self.phases),
        )
        episode_index = self._next_episode_index()
        final_path = self.root / f"episode_{episode_index:06d}"
        metadata = {
            "schema_version": "classic-pick-place-raw-v1",
            "episode_index": episode_index,
            "task": self.task,
            "success": bool(success),
            "fps": self.fps,
            "frames": len(self.states),
            "duration_s": len(self.states) / self.fps,
            "seed": self.seed,
            "source": self.source,
            "robot_type": "franka_panda_mujoco",
            "state_names": STATE_NAMES,
            "action_names": [f"joint_{i}_target" for i in range(1, 8)]
            + ["gripper_open"],
            "action_space": "absolute_joint_position_rad_plus_binary_gripper",
            "cameras": ["overview", "wrist_rgbd"],
            "image_shape_hwc": [self.height, self.width, 3],
            "metrics": metrics or {},
        }
        (self.temp_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self.temp_path.rename(final_path)
        return EpisodeSummary(
            final_path,
            len(self.states),
            len(self.states) / self.fps,
            bool(success),
        )

    def discard(self) -> None:
        self._close_streams()
        resolved_root = self.root.resolve()
        resolved_temp = self.temp_path.resolve()
        if resolved_temp.parent != resolved_root or not resolved_temp.name.startswith(
            ".recording_"
        ):
            raise RuntimeError(f"Unsafe temporary episode path: {resolved_temp}")
        shutil.rmtree(resolved_temp)

    def _next_episode_index(self) -> int:
        indices = []
        for path in self.root.glob("episode_[0-9][0-9][0-9][0-9][0-9][0-9]"):
            try:
                indices.append(int(path.name.split("_")[-1]))
            except ValueError:
                continue
        return max(indices, default=-1) + 1


