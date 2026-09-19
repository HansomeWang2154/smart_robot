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
    payload_obstacle_collision_steps: int = 0
    minimum_payload_obstacle_signed_distance: float = np.inf
    contact_pairs: set[tuple[str, str]] = field(default_factory=set)
    payload_contact_pairs: set[tuple[str, str]] = field(default_factory=set)


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
        self.grasp_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )
        self.object_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "object_site"
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
        # Ignore purely visual cup geometry, but include every physical part
        # (the body and handle) when checking a carried payload.
        self.payload_collision_geoms = {
            geom for geom in self.object_geoms
            if int(self.model.geom_contype[geom]) != 0
            or int(self.model.geom_conaffinity[geom]) != 0
        }
        self.left_finger_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger"
        )
        self.right_finger_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger"
        )
        self.finger_qpos = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
            ],
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")
            ],
        ])
        self.left_finger_geoms = {
            geom for geom in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom]) == self.left_finger_body
        }
        self.right_finger_geoms = {
            geom for geom in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom]) == self.right_finger_body
        }
        self.table_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.stats = CollisionStats()
        self.grasp_ramp_step = 0
        self.grasp_ramp_steps = max(1, int(round(0.30 / self.dt)))
        self.grasp_soft_timeconst = 0.08
        self.grasp_stiff_timeconst = 0.015
        self.grasp_alignment_error = np.inf
        self._configure_torque_actuators()
        self.reset()

    def grasp_contact_state(self) -> tuple[bool, bool]:
        """Return whether the physical cup touches each fingertip group."""
        left = False
        right = False
        for contact in self.data.contact[: self.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair & self.object_geoms:
                left |= bool(pair & self.left_finger_geoms)
                right |= bool(pair & self.right_finger_geoms)
        return left, right

    def _align_object_site_to_grasp(self) -> float:
        """Move the cup's weld site so activation starts with zero residual."""
        mujoco.mj_forward(self.model, self.data)
        grasp_position = self.data.site_xpos[self.grasp_site].copy()
        grasp_rotation = self.data.site_xmat[self.grasp_site].reshape(3, 3).copy()
        object_position = self.data.xpos[self.object_body].copy()
        object_rotation = self.data.xmat[self.object_body].reshape(3, 3).copy()
        before = float(
            np.linalg.norm(self.data.site_xpos[self.grasp_site] - self.data.site_xpos[self.object_site])
        )
        self.model.site_pos[self.object_site] = object_rotation.T @ (
            grasp_position - object_position
        )
        local_rotation = object_rotation.T @ grasp_rotation
        local_quaternion = np.empty(4)
        mujoco.mju_mat2Quat(local_quaternion, local_rotation.ravel())
        self.model.site_quat[self.object_site] = local_quaternion
        mujoco.mj_forward(self.model, self.data)
        return before

    def set_grasp_constraint(self, active: bool, *, require_bilateral_contact: bool = True) -> None:
        """Enable a contact-gated, zero-residual, gradually stiffened weld."""
        if active:
            left, right = self.grasp_contact_state()
            if require_bilateral_contact and not (left and right):
                site_error_vector = (
                    self.data.site_xpos[self.grasp_site]
                    - self.data.site_xpos[self.object_site]
                )
                site_error = float(np.linalg.norm(site_error_vector))
                raise RuntimeError(
                    "Grasp rejected: bilateral contact required "
                    f"(left={left}, right={right}, "
                    f"finger_qpos={self.data.qpos[self.finger_qpos].tolist()}, "
                    f"site_error_vector_m={site_error_vector.tolist()}, "
                    f"site_error_m={site_error:.6f})"
                )
            self.grasp_alignment_error = self._align_object_site_to_grasp()
            self.model.eq_solref[self.grasp_equality, 0] = self.grasp_soft_timeconst
            self.grasp_ramp_step = 0
        self.data.eq_active[self.grasp_equality] = bool(active)
        if not active:
            self.grasp_ramp_step = 0
        mujoco.mj_forward(self.model, self.data)

    def _update_grasp_softness(self) -> None:
        if not bool(self.data.eq_active[self.grasp_equality]):
            return
        alpha = min(1.0, self.grasp_ramp_step / self.grasp_ramp_steps)
        self.model.eq_solref[self.grasp_equality, 0] = (
            (1.0 - alpha) * self.grasp_soft_timeconst
            + alpha * self.grasp_stiff_timeconst
        )
        self.grasp_ramp_step += 1

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
        # Keep the cup in the physically graspable half of the pick region.
        # Positions closer than roughly 0.23 m to the centerline leave too
        # little room between the Panda hand and the central barrier.
        self.data.qpos[self.object_qpos_adr + 1] = rng.uniform(-0.32, -0.24)
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

    def _geom_signed_distance(
        self,
        data: mujoco.MjData,
        geom1: int,
        geom2: int,
    ) -> float:
        from_to = np.empty(6)
        return float(
            mujoco.mj_geomDistance(
                self.model, data, geom1, geom2, 10.0, from_to
            )
        )

    def collision_checker(
        self,
        *,
        payload: bool = False,
        allow_object_contact: bool = False,
        payload_obstacle_clearance: float = 0.025,
        payload_table_clearance: float = 0.010,
    ):
        scratch = mujoco.MjData(self.model)
        object_qpos = self.data.qpos[self.object_qpos_adr : self.object_qpos_adr + 7].copy()
        payload_position_in_grasp = None
        payload_rotation_in_grasp = None
        if payload:
            if not bool(self.data.eq_active[self.grasp_equality]):
                raise RuntimeError(
                    "Payload collision checking must be created after the cup is grasped."
                )
            mujoco.mj_forward(self.model, self.data)
            grasp_position = self.data.site_xpos[self.grasp_site].copy()
            grasp_rotation = self.data.site_xmat[self.grasp_site].reshape(3, 3).copy()
            object_position = self.data.xpos[self.object_body].copy()
            object_rotation = self.data.xmat[self.object_body].reshape(3, 3).copy()
            # Preserve the measured hand-to-cup transform rather than assuming
            # that the cup is centred on the end effector.
            payload_position_in_grasp = grasp_rotation.T @ (
                object_position - grasp_position
            )
            payload_rotation_in_grasp = grasp_rotation.T @ object_rotation

        def is_free(q: np.ndarray) -> bool:
            scratch.qpos[:] = self.data.qpos
            scratch.qpos[self.object_qpos_adr : self.object_qpos_adr + 7] = object_qpos
            scratch.qpos[self.arm_qpos] = q
            scratch.qvel[:] = 0.0
            mujoco.mj_forward(self.model, scratch)
            if payload:
                assert payload_position_in_grasp is not None
                assert payload_rotation_in_grasp is not None
                grasp_position = scratch.site_xpos[self.grasp_site].copy()
                grasp_rotation = scratch.site_xmat[self.grasp_site].reshape(3, 3).copy()
                object_position = (
                    grasp_position + grasp_rotation @ payload_position_in_grasp
                )
                object_rotation = grasp_rotation @ payload_rotation_in_grasp
                object_quaternion = np.empty(4)
                mujoco.mju_mat2Quat(object_quaternion, object_rotation.ravel())
                scratch.qpos[
                    self.object_qpos_adr : self.object_qpos_adr + 3
                ] = object_position
                scratch.qpos[
                    self.object_qpos_adr + 3 : self.object_qpos_adr + 7
                ] = object_quaternion
                mujoco.mj_forward(self.model, scratch)
            if self.forbidden_contacts(
                scratch, allow_object=(payload or allow_object_contact)
            ):
                return False
            if payload:
                obstacle_distance = min(
                    self._geom_signed_distance(scratch, geom, self.obstacle_geom)
                    for geom in self.payload_collision_geoms
                )
                if obstacle_distance < payload_obstacle_clearance:
                    return False
                table_distance = min(
                    self._geom_signed_distance(scratch, geom, self.table_geom)
                    for geom in self.payload_collision_geoms
                )
                if table_distance < payload_table_clearance:
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
        self._update_grasp_softness()
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

        payload_distances = {
            geom: self._geom_signed_distance(self.data, geom, self.obstacle_geom)
            for geom in self.payload_collision_geoms
        }
        minimum_payload_distance = min(payload_distances.values())
        self.stats.minimum_payload_obstacle_signed_distance = min(
            self.stats.minimum_payload_obstacle_signed_distance,
            minimum_payload_distance,
        )
        colliding_payload_geoms = [
            geom for geom, distance in payload_distances.items() if distance < 0.0
        ]
        if colliding_payload_geoms:
            self.stats.payload_obstacle_collision_steps += 1
            obstacle_name = self._geom_name(self.obstacle_geom)
            self.stats.payload_contact_pairs.update(
                tuple(sorted((self._geom_name(geom), obstacle_name)))
                for geom in colliding_payload_geoms
            )

