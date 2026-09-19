from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classic_pick_place.perception import CupDetection, RgbdCupDetector
from classic_pick_place.planning import RRTConnect, smooth_joint_path
from classic_pick_place.simulation import PandaPickPlace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual collision-free Panda pick and place")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--fixed-object",
        action="store_true",
        help="Keep the XML cup pose instead of randomizing it from the seed",
    )
    return parser.parse_args()


def detection_metrics(detection: CupDetection, truth: np.ndarray) -> dict[str, object]:
    return {
        "estimated_object_position": detection.position_world.tolist(),
        "true_object_position_evaluation_only": truth.tolist(),
        "position_error_m": float(np.linalg.norm(detection.position_world - truth)),
        "pixel_uv": list(detection.pixel_uv),
        "confidence": detection.confidence,
        "visible_pixels": detection.visible_pixels,
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xml = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "pick_place.xml"
    if not xml.exists():
        raise FileNotFoundError(f"{xml} is missing. Run `python scripts/setup_assets.py` first.")

    sim = PandaPickPlace(xml)
    if not args.fixed_object:
        sim.randomize_object(args.seed)

    initial_ee, target_rotation = sim.kin.pose(sim.data)
    initial_object_truth = sim.object_position()

    fixed_detector = RgbdCupDetector(sim.model, camera_name="perception")
    try:
        coarse_detection = fixed_detector.detect(
            sim.data,
            output_dir=args.output_dir,
            artifact_prefix="fixed_camera",
        )
    finally:
        fixed_detector.close()

    ik: dict[str, np.ndarray] = {}
    ik_diagnostics: dict[str, dict[str, float | int | bool]] = {}

    def solve_ik(name: str, seed_q: np.ndarray, position: np.ndarray) -> np.ndarray:
        result = sim.kin.solve_ik(seed_q, position, target_rotation)
        ik_diagnostics[name] = {
            "success": result.success,
            "iterations": result.iterations,
            "position_error_m": result.position_error,
            "orientation_error_rad": result.orientation_error,
        }
        if not result.success:
            raise RuntimeError(f"IK failed for {name}: {ik_diagnostics[name]}")
        ik[name] = result.q
        return result.q

    coarse_pregrasp_position = coarse_detection.position_world + np.array([0.0, 0.0, 0.18])
    coarse_pregrasp_q = solve_ik("coarse_pregrasp", sim.home, coarse_pregrasp_position)

    plans = {}
    approach_planner = RRTConnect(
        sim.kin.lower + 0.02,
        sim.kin.upper - 0.02,
        sim.collision_checker(),
        seed=args.seed,
    )
    plans["home_to_observation"] = approach_planner.plan(sim.home, coarse_pregrasp_q)
    if not plans["home_to_observation"].success:
        raise RuntimeError("RRT-Connect failed for home_to_observation")

    renderer = (
        mujoco.Renderer(sim.model, height=args.height, width=args.width)
        if args.render
        else None
    )
    writer = (
        imageio.get_writer(
            args.output_dir / "pick_place.mp4",
            fps=30,
            codec="libx264",
            quality=8,
        )
        if args.render
        else None
    )
    render_stride = max(1, int(round(1.0 / (30.0 * sim.dt))))
    log = {"time": [], "q": [], "q_ref": [], "ee": [], "object": [], "phase": []}
    global_step = 0
    wrist_detection: CupDetection | None = None
    wrist_truth: np.ndarray | None = None
    object_position = coarse_detection.position_world.copy()
    goal_position = np.array([0.53, 0.22, object_position[2]])
    grasp_contacts_at_lock = (False, False)
    object_jump_at_lock = np.inf

    def execute_phase(
        phase_name: str,
        waypoints: list[np.ndarray],
        *,
        gripper_open: bool,
        allow_object_contact: bool,
    ) -> None:
        nonlocal global_step
        min_duration = 1.15 if phase_name == "close" else (1.0 if phase_name == "open" else 0.5)
        if phase_name == "descend":
            max_speed = 0.16
        elif phase_name == "refine":
            max_speed = 0.22
        elif phase_name == "transfer":
            max_speed = 0.25
        else:
            max_speed = 0.42
        q_ref, dq_ref = smooth_joint_path(
            waypoints,
            sim.dt,
            max_speed=max_speed,
            minimum_duration=min_duration,
        )
        if phase_name == "descend":
            hold_duration = 0.50
        elif phase_name == "close":
            hold_duration = 0.35
        else:
            hold_duration = 0.18
        hold_steps = int(round(hold_duration / sim.dt))
        q_ref = np.vstack((q_ref, np.repeat(q_ref[-1][None, :], hold_steps, axis=0)))
        dq_ref = np.vstack((dq_ref, np.zeros((hold_steps, 7))))
        for qr, dqr in zip(q_ref, dq_ref):
            sim.step(qr, dqr, gripper_open)
            sim.update_collision_stats(allow_object=allow_object_contact)
            ee, _ = sim.kin.pose(sim.data)
            log["time"].append(sim.data.time)
            log["q"].append(sim.data.qpos[sim.arm_qpos].copy())
            log["q_ref"].append(qr.copy())
            log["ee"].append(ee)
            log["object"].append(sim.object_position())
            log["phase"].append(phase_name)
            if renderer is not None and global_step % render_stride == 0:
                renderer.update_scene(sim.data, camera="overview")
                assert writer is not None
                writer.append_data(renderer.render())
            global_step += 1

    try:
        execute_phase(
            "approach_observation",
            plans["home_to_observation"].path,
            gripper_open=True,
            allow_object_contact=False,
        )

        if renderer is not None:
            renderer.close()
            renderer = None
        wrist_truth = sim.object_position()
        wrist_detector = RgbdCupDetector(sim.model, camera_name="wrist_rgbd")
        try:
            wrist_detection = wrist_detector.detect(
                sim.data,
                output_dir=args.output_dir,
                artifact_prefix="wrist_camera",
            )
        finally:
            wrist_detector.close()
        object_position = wrist_detection.position_world
        goal_position = np.array([0.53, 0.22, object_position[2]])
        (args.output_dir / "perception.json").write_text(
            json.dumps(
                {
                    "fixed_camera_coarse": detection_metrics(
                        coarse_detection, initial_object_truth
                    ),
                    "wrist_camera_refined": detection_metrics(
                        wrist_detection, wrist_truth
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if args.render:
            renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width)

        targets = {
            "refined_pregrasp": object_position + np.array([0.0, 0.0, 0.18]),
            "grasp": object_position + np.array([0.0, 0.0, 0.035]),
            "lift": object_position + np.array([0.0, 0.0, 0.17]),
            "preplace": goal_position + np.array([0.0, 0.0, 0.23]),
            "place": goal_position + np.array([0.0, 0.0, 0.035]),
            "retreat": goal_position + np.array([0.0, 0.0, 0.22]),
        }
        seed_q = sim.data.qpos[sim.arm_qpos].copy()
        for name, position in targets.items():
            seed_q = solve_ik(name, seed_q, position)

        refine_planner = RRTConnect(
            sim.kin.lower + 0.02,
            sim.kin.upper - 0.02,
            sim.collision_checker(),
            seed=args.seed + 1,
        )
        plans["observation_to_refined_pregrasp"] = refine_planner.plan(
            sim.data.qpos[sim.arm_qpos].copy(),
            ik["refined_pregrasp"],
        )
        return_planner = RRTConnect(
            sim.kin.lower + 0.02,
            sim.kin.upper - 0.02,
            sim.collision_checker(),
            seed=args.seed + 3,
        )
        plans["retreat_to_home"] = return_planner.plan(ik["retreat"], sim.home)
        failed = [name for name, result in plans.items() if not result.success]
        if failed:
            raise RuntimeError(f"RRT-Connect failed for: {failed}")

        execute_phase(
            "refine",
            plans["observation_to_refined_pregrasp"].path,
            gripper_open=True,
            allow_object_contact=False,
        )
        execute_phase(
            "descend",
            [ik["refined_pregrasp"], ik["grasp"]],
            gripper_open=True,
            allow_object_contact=True,
        )
        execute_phase(
            "close",
            [ik["grasp"], ik["grasp"]],
            gripper_open=False,
            allow_object_contact=True,
        )

        grasp_contacts_at_lock = sim.grasp_contact_state()
        object_before_lock = sim.object_position()
        sim.set_grasp_constraint(True, require_bilateral_contact=True)
        object_jump_at_lock = float(
            np.linalg.norm(sim.object_position() - object_before_lock)
        )
        execute_phase(
            "lift",
            [ik["grasp"], ik["lift"]],
            gripper_open=False,
            allow_object_contact=True,
        )
        # The physical grasp can be slightly off-centre. Plan only after the
        # lift so every RRT state carries the full cup at its measured pose.
        transfer_planner = RRTConnect(
            sim.kin.lower + 0.02,
            sim.kin.upper - 0.02,
            sim.collision_checker(
                payload=True,
                payload_obstacle_clearance=0.025,
                payload_table_clearance=0.010,
            ),
            seed=args.seed + 2,
            max_iterations=10000,
        )
        plans["lift_to_preplace"] = transfer_planner.plan(
            sim.data.qpos[sim.arm_qpos].copy(), ik["preplace"]
        )
        if not plans["lift_to_preplace"].success:
            raise RuntimeError("RRT-Connect failed for lift_to_preplace")
        execute_phase(
            "transfer",
            plans["lift_to_preplace"].path,
            gripper_open=False,
            allow_object_contact=True,
        )
        execute_phase(
            "place",
            [ik["preplace"], ik["place"]],
            gripper_open=False,
            allow_object_contact=True,
        )
        sim.set_grasp_constraint(False)
        execute_phase(
            "open",
            [ik["place"], ik["place"]],
            gripper_open=True,
            allow_object_contact=True,
        )
        execute_phase(
            "retreat",
            [ik["place"], ik["retreat"]],
            gripper_open=True,
            allow_object_contact=True,
        )
        execute_phase(
            "return",
            plans["retreat_to_home"].path,
            gripper_open=True,
            allow_object_contact=False,
        )
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    assert wrist_detection is not None and wrist_truth is not None
    arrays = {key: np.asarray(value) for key, value in log.items() if key != "phase"}
    np.savez_compressed(args.output_dir / "trajectory.npz", **arrays)
    q_error = arrays["q_ref"] - arrays["q"]
    final_object = sim.object_position()
    xy_error = float(np.linalg.norm(final_object[:2] - goal_position[:2]))
    success = (
        xy_error < 0.065
        and final_object[2] > 0.37
        and sim.stats.forbidden_contacts == 0
        and sim.stats.payload_obstacle_collision_steps == 0
        and all(grasp_contacts_at_lock)
    )

    metrics = {
        "success": bool(success),
        "final_object_position": final_object.tolist(),
        "goal_position": goal_position.tolist(),
        "perception": {
            "control_source": "eye_in_hand_rgbd_refinement",
            "fixed_camera_coarse": detection_metrics(coarse_detection, initial_object_truth),
            "wrist_camera_refined": detection_metrics(wrist_detection, wrist_truth),
        },
        "grasp": {
            "bilateral_contact_at_lock": list(grasp_contacts_at_lock),
            "alignment_error_before_zero_residual_lock_m": float(sim.grasp_alignment_error),
            "object_jump_at_lock_m": object_jump_at_lock,
            "constraint_ramp_duration_s": 0.30,
            "stabilization": "contact_gated_zero_residual_soft_to_stiff_site_weld",
        },
        "placement_xy_error_m": xy_error,
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean(q_error**2))),
        "max_joint_tracking_error_rad": float(np.max(np.abs(q_error))),
        "forbidden_contact_count": sim.stats.forbidden_contacts,
        "safety_margin_contact_count": sim.stats.safety_margin_contacts,
        "forbidden_contact_pairs": sorted([list(pair) for pair in sim.stats.contact_pairs]),
        "payload_obstacle_collision_steps": sim.stats.payload_obstacle_collision_steps,
        "payload_obstacle_contact_pairs": sorted(
            [list(pair) for pair in sim.stats.payload_contact_pairs]
        ),
        "minimum_payload_obstacle_signed_distance_m": (
            float(sim.stats.minimum_payload_obstacle_signed_distance)
            if np.isfinite(sim.stats.minimum_payload_obstacle_signed_distance)
            else None
        ),
        "minimum_robot_obstacle_signed_distance_m": (
            float(sim.stats.minimum_robot_obstacle_signed_distance)
            if np.isfinite(sim.stats.minimum_robot_obstacle_signed_distance)
            else None
        ),
        "minimum_ee_obstacle_clearance_m": float(sim.stats.minimum_obstacle_clearance),
        "duration_s": float(sim.data.time),
        "initial_ee_position": initial_ee.tolist(),
        "ik": ik_diagnostics,
        "planner": {
            name: {
                "success": result.success,
                "iterations": result.iterations,
                "collision_checks": result.collision_checks,
                "waypoints": len(result.path),
            }
            for name, result in plans.items()
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(arrays["time"], q_error)
    axes[0].set_ylabel("joint tracking error [rad]")
    axes[0].grid(alpha=0.25)
    axes[1].plot(arrays["time"], arrays["object"][:, 0], label="object x")
    axes[1].plot(arrays["time"], arrays["object"][:, 1], label="object y")
    axes[1].plot(arrays["time"], arrays["object"][:, 2], label="object z")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("object position [m]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3)
    fig.savefig(args.output_dir / "trajectory.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())

