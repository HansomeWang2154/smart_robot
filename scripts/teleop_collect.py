from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classic_pick_place.simulation import PandaPickPlace
from classic_pick_place.teleop import PygameTeleopDevice
from classic_pick_place.vla_dataset import RawVlaEpisodeRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gamepad teleoperation and VLA data collection")
    parser.add_argument("--device", choices=("gamepad", "keyboard"), default="gamepad")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "datasets" / "raw")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--task", default="pick up the blue cup and place it on the green target")
    parser.add_argument("--translation-speed", type=float, default=0.12, help="m/s at full stick")
    parser.add_argument("--rotation-speed", type=float, default=0.9, help="rad/s at full stick")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "pick_place.xml"
    if not xml.exists():
        raise FileNotFoundError("Run `python scripts/setup_assets.py` first")
    sim = PandaPickPlace(xml)
    device = PygameTeleopDevice(args.device, headless=args.headless)
    display_renderer = mujoco.Renderer(sim.model, width=960, height=544)
    episode_number = 0
    seed = args.seed
    recorder: RawVlaEpisodeRecorder | None = None

    def reset_episode() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        nonlocal recorder, seed
        sim.reset()
        sim.randomize_object(seed)
        position, rotation = sim.kin.pose(sim.data)
        q_target = sim.data.qpos[sim.arm_qpos].copy()
        recorder = RawVlaEpisodeRecorder(
            sim.model,
            args.dataset_root,
            fps=args.fps,
            task=args.task,
            seed=seed,
            source=f"{args.device}_teleoperation",
        )
        seed += 1
        return position, rotation, q_target, True

    target_position, target_rotation, q_target, gripper_open = reset_episode()
    period = 1.0 / args.fps
    physics_steps = max(1, int(round(period / sim.dt)))
    try:
        while episode_number < args.episodes:
            loop_start = time.perf_counter()
            command = device.poll()
            if command.quit:
                break
            if command.toggle_gripper:
                gripper_open = not gripper_open
                if gripper_open and bool(sim.data.eq_active[sim.grasp_equality]):
                    sim.set_grasp_constraint(False)
            if command.reset or command.discard_episode:
                assert recorder is not None
                recorder.discard()
                target_position, target_rotation, q_target, gripper_open = reset_episode()
                continue
            if command.active:
                target_position = target_position + command.twist[:3] * (
                    args.translation_speed * period
                )
                target_position = np.clip(
                    target_position,
                    np.array([0.25, -0.47, 0.39]),
                    np.array([0.80, 0.47, 0.85]),
                )
                delta_rotation = Rotation.from_rotvec(
                    command.twist[3:] * args.rotation_speed * period
                ).as_matrix()
                target_rotation = delta_rotation @ target_rotation
                result = sim.kin.solve_ik(
                    sim.data.qpos[sim.arm_qpos].copy(),
                    target_position,
                    target_rotation,
                    max_iterations=80,
                )
                if result.success:
                    if bool(sim.data.eq_active[sim.grasp_equality]):
                        checker = sim.collision_checker(
                            payload=True,
                            payload_obstacle_clearance=0.012,
                            payload_table_clearance=0.0,
                        )
                    else:
                        checker = sim.collision_checker(allow_object_contact=True)
                    if checker(result.q):
                        q_target = result.q
                    else:
                        target_position, target_rotation = sim.kin.pose(sim.data)

            for _ in range(physics_steps):
                sim.step(q_target, np.zeros(7), gripper_open)
                sim.update_collision_stats(allow_object=not gripper_open)
            if (
                not gripper_open
                and not bool(sim.data.eq_active[sim.grasp_equality])
                and all(sim.grasp_contact_state())
            ):
                sim.set_grasp_constraint(True, require_bilateral_contact=True)

            ee_position, ee_rotation = sim.kin.pose(sim.data)
            assert recorder is not None
            recorder.add_frame(
                sim.data,
                arm_qpos=sim.arm_qpos,
                arm_dofs=sim.arm_dofs,
                finger_qpos=sim.finger_qpos,
                ee_position=ee_position,
                ee_rotation=ee_rotation,
                q_target=q_target,
                gripper_open=gripper_open,
                phase="teleoperation",
            )
            display_renderer.update_scene(sim.data, camera="overview")
            device.show(
                display_renderer.render(),
                f"episode={episode_number}  deadman={'ON' if command.active else 'off'}  "
                f"gripper={'open' if gripper_open else 'closed'}  Enter/Start=save  Back=discard",
            )

            if command.save_episode:
                goal = np.array([0.53, 0.22])
                error = float(np.linalg.norm(sim.object_position()[:2] - goal))
                success = error < 0.065 and sim.stats.payload_obstacle_collision_steps == 0
                summary = recorder.save(
                    success=success,
                    metrics={
                        "placement_xy_error_m": error,
                        "forbidden_contact_count": sim.stats.forbidden_contacts,
                        "payload_obstacle_collision_steps": sim.stats.payload_obstacle_collision_steps,
                    },
                )
                print(f"Saved {summary.path} ({summary.frames} frames, success={success})")
                episode_number += 1
                if episode_number < args.episodes:
                    target_position, target_rotation, q_target, gripper_open = reset_episode()

            remaining = period - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        if recorder is not None and not recorder.closed:
            recorder.discard()
        display_renderer.close()
        device.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


