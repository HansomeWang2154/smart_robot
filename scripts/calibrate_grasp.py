from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from classic_pick_place.planning import smooth_joint_path
from classic_pick_place.simulation import PandaPickPlace


def evaluate(xml: Path, grasp_offset: float) -> dict[str, object]:
    sim = PandaPickPlace(xml)
    sim.randomize_object(7)
    object_start = sim.object_position()
    # Reproduce the measured seed-7 wrist-camera error instead of calibrating
    # only at a perfect ground-truth center.
    perceived_object = object_start + np.array([0.00423, 0.00026, -0.00009])
    _, rotation = sim.kin.pose(sim.data)
    pregrasp = sim.kin.solve_ik(
        sim.home,
        perceived_object + np.array([0.0, 0.0, 0.18]),
        rotation,
    )
    grasp = sim.kin.solve_ik(
        pregrasp.q,
        perceived_object + np.array([0.0, 0.0, grasp_offset]),
        rotation,
    )
    if not (pregrasp.success and grasp.success):
        return {"offset_m": grasp_offset, "ik_success": False}

    sim.data.qpos[sim.arm_qpos] = pregrasp.q
    sim.data.qvel[:] = 0.0
    mujoco.mj_forward(sim.model, sim.data)
    q_ref, dq_ref = smooth_joint_path(
        [pregrasp.q, grasp.q],
        sim.dt,
        max_speed=0.20,
        minimum_duration=1.0,
    )
    for qr, dqr in zip(q_ref, dq_ref):
        sim.step(qr, dqr, gripper_open=True)
    for _ in range(int(round(1.5 / sim.dt))):
        sim.step(grasp.q, np.zeros(7), gripper_open=False)

    contact = sim.grasp_contact_state()
    site_error_vector = (
        sim.data.site_xpos[sim.grasp_site] - sim.data.site_xpos[sim.object_site]
    )
    return {
        "offset_m": grasp_offset,
        "ik_success": True,
        "bilateral_contact": list(contact),
        "finger_qpos": sim.data.qpos[sim.finger_qpos].tolist(),
        "object_displacement_m": float(np.linalg.norm(sim.object_position() - object_start)),
        "site_error_vector_m": site_error_vector.tolist(),
        "site_error_m": float(np.linalg.norm(site_error_vector)),
    }


def main() -> None:
    xml = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "pick_place.xml"
    offsets = [0.02, 0.035, 0.05, 0.065, 0.08, 0.095]
    print(json.dumps([evaluate(xml, offset) for offset in offsets], indent=2))


if __name__ == "__main__":
    main()

