from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .kinematics import PandaKinematics


ROBOT_BODY_NAMES = {
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
}


@dataclass
class CollisionStats:
    forbidden_contacts: int = 0
    safety_margin_contacts: int = 0
    minimum_robot_obstacle_signed_distance: float = np.inf
    minimum_obstacle_clearance: float = np.inf
    contact_pairs: set[tuple[str, str]] = field(default_factory=set)


class PandaPickPlace:
    def __init__(self, xml_path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.kin = PandaKinematics(self.model)
        self.dt = float(self.model.opt.timestep)
        # Collision-free ready pose on the pick side of the central barrier.
        # It preserves a vertical, downward-facing gripper orientation.
        self.home = np.array(
            [-0.250264, -0.056538, -0.450890, -1.532424, -0.018676, 1.489726, 0.077331]
        )
        self.arm_qpos = self.kin.arm_qpos
        self.arm_dofs = self.kin.arm_dofs
        self.object_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.obstacle_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom")
        self.object_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
        self.object_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
        self.object_qpos_adr = self.model.jnt_qposadr[self.object_joint]
        self.grasp_equality = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld"
        )
        self.robot_bodies = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in ROBOT_BODY_NAMES
        }
        self.robot_geoms = {
            geom for geom in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom]) in self.robot_bodies
        }
        self.object_geoms = {
            geom for geom in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom]) == self.object_body
        }
        self.table_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.stats = CollisionStats()
        self._configure_torque_actuators()
        self.reset()

    def set_grasp_constraint(self, active: bool) -> None:
        """Enable the post-closure rigid-grasp approximation."""
        self.data.eq_active[self.grasp_equality] = bool(active)
        mujoco.mj_forward(self.model, self.data)

    def _configure_torque_actuators(self) -> None:
        # The Menagerie model uses general position servos. Setting unit gain and
        # zero bias turns the first seven transmissions into direct torque motors.
        self.model.actuator_gainprm[:7, :] = 0.0
        self.model.actuator_gainprm[:7, 0] = 1.0
        self.model.actuator_biasprm[:7, :] = 0.0
        torque_limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
        self.model.actuator_ctrlrange[:7, 0] = -torque_limits
        self.model.actuator_ctrlrange[:7, 1] = torque_limits
        self.torque_limits = torque_limits

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.arm_qpos] = self.home
        finger1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        finger2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
        self.data.qpos[self.model.jnt_qposadr[finger1]] = 0.04
        self.data.qpos[self.model.jnt_qposadr[finger2]] = 0.04
        self.data.ctrl[7] = 255.0
        mujoco.mj_forward(self.model, self.data)
        self.stats = CollisionStats()

    def object_position(self, data: mujoco.MjData | None = None) -> np.ndarray:
        data = self.data if data is None else data
        return data.xpos[self.object_body].copy()

    def randomize_object(self, seed: int) -> np.ndarray:
        """Randomize the cup on the reachable pick side of the table."""
        rng = np.random.default_rng(seed)
        self.data.qpos[self.object_qpos_adr] = rng.uniform(0.43, 0.63)
        self.data.qpos[self.object_qpos_adr + 1] = rng.uniform(-0.31, -0.17)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self.object_position()

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"

    def forbidden_contacts(
        self,
        data: mujoco.MjData,
        *,
        allow_object: bool,
        penetration_only: bool = False,
    ) -> list[tuple[str, str]]:
        result: list[tuple[str, str]] = []
        for contact in data.contact[: data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            r1, r2 = g1 in self.robot_geoms, g2 in self.robot_geoms
            if not (r1 or r2):
                continue
            # MuJoCo generates contacts inside a geom's positive margin before
            # surfaces touch. Keep those for conservative planning, but do not
            # report them as physical collisions during execution.
            if penetration_only and float(contact.dist) >= 0.0:
                continue
            if allow_object and (g1 in self.object_geoms or g2 in self.object_geoms):
                continue
            # The fixed base touching the floor is benign; Menagerie exclusions
            # already handle adjacent-link contacts.
            names = {self._geom_name(g1), self._geom_name(g2)}
            body_names = {
                mujoco.mj_id2name(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(self.model.geom_bodyid[geom]),
                )
                for geom in (g1, g2)
            }
            # The two fingertips are intentionally allowed to meet after a
            # close command when the perceived grasp is slightly off-center.
            if body_names <= {"left_finger", "right_finger"}:
                continue
            if "floor" in names and any("link0" in n for n in names):
                continue
            result.append(tuple(sorted(names)))
        return result

    def collision_checker(self, *, payload: bool = False):
        scratch = mujoco.MjData(self.model)
        object_qpos = self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 7].copy()
        obstacle_center = self.model.geom_pos[self.obstacle_geom].copy()
        obstacle_body = int(self.model.geom_bodyid[self.obstacle_geom])
        obstacle_center = self.model.body_pos[obstacle_body] + obstacle_center
        obstacle_half = self.model.geom_size[self.obstacle_geom].copy()

        def is_free(q: np.ndarray) -> bool:
            scratch.qpos[:] = self.data.qpos
            scratch.qpos[self.object_qpos_adr : self.object_qpos_adr + 7] = object_qpos
            scratch.qpos[self.arm_qpos] = q
            scratch.qvel[:] = 0.0
            mujoco.mj_forward(self.model, scratch)
            if self.forbidden_contacts(scratch, allow_object=payload):
                return False
            if payload:
                point, _ = self.kin.pose(scratch)
                closest = np.maximum(np.abs(point - obstacle_center) - obstacle_half, 0.0)
                # Conservative swept-volume proxy for the grasped cube. The
                # 85 mm radius absorbs grasp offset and closed-loop tracking
                # error, not just the cube's geometric half-width.
                if np.linalg.norm(closest) < 0.085:
                    return False
                if point[2] < 0.395:
                    return False
            return True

        return is_free

    def computed_torque(self, q_ref: np.ndarray, dq_ref: np.ndarray) -> np.ndarray:
        q = self.data.qpos[self.arm_qpos]
        dq = self.data.qvel[self.arm_dofs]
        kp = np.array([95.0, 95.0, 82.0, 82.0, 55.0, 42.0, 32.0])
        kd = 2.0 * np.sqrt(kp) * 0.92
        acceleration = kp * (q_ref - q) + kd * (dq_ref - dq)
        full_mass = np.zeros((self.model.nv, self.model.nv))
        # MuJoCo 3.13 accepts MjData directly; older releases accept qM.
        try:
            mujoco.mj_fullM(self.model, self.data, full_mass)
        except TypeError:
            mujoco.mj_fullM(self.model, full_mass, self.data.qM)
        mass = full_mass[np.ix_(self.arm_dofs, self.arm_dofs)]
        bias = self.data.qfrc_bias[self.arm_dofs]
        tau = mass @ acceleration + bias
        return np.clip(tau, -self.torque_limits, self.torque_limits)

    def step(self, q_ref: np.ndarray, dq_ref: np.ndarray, gripper_open: bool) -> None:
        self.data.ctrl[:7] = self.computed_torque(q_ref, dq_ref)
        self.data.ctrl[7] = 255.0 if gripper_open else 0.0
        mujoco.mj_step(self.model, self.data)

    def update_collision_stats(self, allow_object: bool) -> None:
        margin_contacts = self.forbidden_contacts(self.data, allow_object=allow_object)
        contacts = self.forbidden_contacts(
            self.data, allow_object=allow_object, penetration_only=True
        )
        self.stats.safety_margin_contacts += max(0, len(margin_contacts) - len(contacts))
        if contacts:
            self.stats.forbidden_contacts += len(contacts)
            self.stats.contact_pairs.update(contacts)

        for contact in self.data.contact[: self.data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if self.obstacle_geom in (g1, g2) and (g1 in self.robot_geoms or g2 in self.robot_geoms):
                self.stats.minimum_robot_obstacle_signed_distance = min(
                    self.stats.minimum_robot_obstacle_signed_distance, float(contact.dist)
                )

        obstacle_center = self.data.geom_xpos[self.obstacle_geom]
        obstacle_half = self.model.geom_size[self.obstacle_geom]
        ee_position, _ = self.kin.pose(self.data)
        closest = np.maximum(np.abs(ee_position - obstacle_center) - obstacle_half, 0.0)
        self.stats.minimum_obstacle_clearance = min(
            self.stats.minimum_obstacle_clearance, float(np.linalg.norm(closest))
        )

