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

from classic_pick_place.planning import RRTConnect, smooth_joint_path
from classic_pick_place.perception import RgbdCupDetector
from classic_pick_place.simulation import PandaPickPlace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical collision-free Panda pick and place")
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


def concatenate_paths(parts: list[list[np.ndarray]]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for part in parts:
        if result and np.allclose(result[-1], part[0]):
            result.extend(part[1:])
        else:
            result.extend(part)
    return result


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xml = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "pick_place.xml"
    if not xml.exists():
        raise FileNotFoundError(
            f"{xml} is missing. Run `python scripts/setup_assets.py` first."
        )

    sim = PandaPickPlace(xml)
    if not args.fixed_object:
        sim.randomize_object(args.seed)

    detector = RgbdCupDetector(sim.model)
    try:
        detection = detector.detect(sim.data, output_dir=args.output_dir)
    finally:
        detector.close()

    initial_ee, target_rotation = sim.kin.pose(sim.data)
    true_object_position = sim.object_position()
    object_position = detection.position_world
    goal_position = np.array([0.53, 0.22, object_position[2]])

    targets = {
        "pregrasp": object_position + np.array([0.0, 0.0, 0.18]),
        "grasp": object_position + np.array([0.0, 0.0, 0.035]),
        # A vertical tool orientation constrains the Panda workspace. 0.245 m
        # is reachable on both sides but lower than the obstacle top, forcing
        # the configuration-space planner to find a genuine route around it.
        # A lower initial lift stays reachable across the randomized pick
        # region; the payload planner then raises toward the fixed shelf side.
        "lift": object_position + np.array([0.0, 0.0, 0.20]),
        "preplace": goal_position + np.array([0.0, 0.0, 0.23]),
        "place": goal_position + np.array([0.0, 0.0, 0.035]),
        "retreat": goal_position + np.array([0.0, 0.0, 0.22]),
    }

    ik: dict[str, np.ndarray] = {}
    seed_q = sim.home.copy()
    ik_diagnostics: dict[str, dict[str, float | int | bool]] = {}
    for name, position in targets.items():
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
        seed_q = result.q

    plans = {}
    planner1 = RRTConnect(sim.kin.lower + 0.02, sim.kin.upper - 0.02, sim.collision_checker(), seed=args.seed)
    plans["home_to_pregrasp"] = planner1.plan(sim.home, ik["pregrasp"])
    planner2 = RRTConnect(
        sim.kin.lower + 0.02,
        sim.kin.upper - 0.02,
        sim.collision_checker(payload=True),
        seed=args.seed + 1,
        max_iterations=6000,
    )
    plans["lift_to_preplace"] = planner2.plan(ik["lift"], ik["preplace"])
    planner3 = RRTConnect(sim.kin.lower + 0.02, sim.kin.upper - 0.02, sim.collision_checker(), seed=args.seed + 2)
    plans["retreat_to_home"] = planner3.plan(ik["retreat"], sim.home)
    failed = [name for name, result in plans.items() if not result.success]
    if failed:
        raise RuntimeError(f"RRT-Connect failed for: {failed}")

    phases: list[tuple[str, list[np.ndarray], bool, bool]] = [
        ("approach", plans["home_to_pregrasp"].path, True, False),
        ("descend", [ik["pregrasp"], ik["grasp"]], True, True),
        ("close", [ik["grasp"], ik["grasp"]], False, True),
        ("lift", [ik["grasp"], ik["lift"]], False, True),
        ("transfer", plans["lift_to_preplace"].path, False, True),
        ("place", [ik["preplace"], ik["place"]], False, True),
        ("open", [ik["place"], ik["place"]], True, True),
        # Contact with the manipulated cube is still intentional while the
        # opened fingers withdraw; obstacle/table/self contacts remain banned.
        ("retreat", [ik["place"], ik["retreat"]], True, True),
        ("return", plans["retreat_to_home"].path, True, False),
    ]

    renderer = mujoco.Renderer(sim.model, height=args.height, width=args.width) if args.render else None
    writer = imageio.get_writer(args.output_dir / "pick_place.mp4", fps=30, codec="libx264", quality=8) if args.render else None
    render_stride = max(1, int(round(1.0 / (30.0 * sim.dt))))
    log = {"time": [], "q": [], "q_ref": [], "ee": [], "object": [], "phase": []}
    global_step = 0

    try:
        for phase_name, waypoints, gripper_open, allow_object_contact in phases:
            if phase_name == "lift":
                sim.set_grasp_constraint(True)
            elif phase_name == "open":
                sim.set_grasp_constraint(False)
            min_duration = 1.0 if phase_name in {"close", "open"} else 0.5
            max_speed = 0.25 if phase_name == "transfer" else 0.42
            q_ref, dq_ref = smooth_joint_path(
                waypoints, sim.dt, max_speed=max_speed, minimum_duration=min_duration
            )
            # The arm is torque-limited. A short terminal hold prevents state
            # transitions from accumulating tracking lag at RRT waypoints.
            hold_steps = int(round(0.18 / sim.dt))
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
                    writer.append_data(renderer.render())
                global_step += 1
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    arrays = {key: np.asarray(value) for key, value in log.items() if key != "phase"}
    np.savez_compressed(args.output_dir / "trajectory.npz", **arrays)
    q_error = arrays["q_ref"] - arrays["q"]
    final_object = sim.object_position()
    xy_error = float(np.linalg.norm(final_object[:2] - goal_position[:2]))
    success = xy_error < 0.065 and final_object[2] > 0.37 and sim.stats.forbidden_contacts == 0

    metrics = {
        "success": bool(success),
        "final_object_position": final_object.tolist(),
        "goal_position": goal_position.tolist(),
        "perception": {
            "source": "fixed_overhead_rgbd_camera",
            "estimated_object_position": object_position.tolist(),
            "true_object_position_evaluation_only": true_object_position.tolist(),
            "position_error_m": float(np.linalg.norm(object_position - true_object_position)),
            "pixel_uv": list(detection.pixel_uv),
            "confidence": detection.confidence,
            "visible_pixels": detection.visible_pixels,
        },
        "placement_xy_error_m": xy_error,
        "joint_tracking_rmse_rad": float(np.sqrt(np.mean(q_error**2))),
        "max_joint_tracking_error_rad": float(np.max(np.abs(q_error))),
        "forbidden_contact_count": sim.stats.forbidden_contacts,
        "safety_margin_contact_count": sim.stats.safety_margin_contacts,
        "forbidden_contact_pairs": sorted([list(pair) for pair in sim.stats.contact_pairs]),
        "minimum_robot_obstacle_signed_distance_m": (
            float(sim.stats.minimum_robot_obstacle_signed_distance)
            if np.isfinite(sim.stats.minimum_robot_obstacle_signed_distance)
            else None
        ),
        "minimum_ee_obstacle_clearance_m": float(sim.stats.minimum_obstacle_clearance),
        "duration_s": float(sim.data.time),
        "grasp_stabilization": "site_weld_enabled_after_closure_and_disabled_before_release",
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
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

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

