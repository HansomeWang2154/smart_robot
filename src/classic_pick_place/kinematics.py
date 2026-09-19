from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    position_error: float
    orientation_error: float
    iterations: int
    success: bool


class PandaKinematics:
    """Forward kinematics and damped least-squares IK for the Panda hand."""

    def __init__(self, model: mujoco.MjModel, hand_offset: float = 0.103):
        self.model = model
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.hand_id < 0:
            raise ValueError("Panda body 'hand' was not found")
        self.hand_offset = np.array([0.0, 0.0, hand_offset], dtype=float)
        self.arm_qpos = np.array(
            [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 8)]
        )
        self.arm_dofs = np.array(
            [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")] for i in range(1, 8)]
        )
        self.lower = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"), 0] for i in range(1, 8)])
        self.upper = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}"), 1] for i in range(1, 8)])

    def pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        rotation = data.xmat[self.hand_id].reshape(3, 3).copy()
        position = data.xpos[self.hand_id].copy() + rotation @ self.hand_offset
        return position, rotation

    def jacobian(self, data: mujoco.MjData) -> np.ndarray:
        position, _ = self.pose(data)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, data, jacp, jacr, position, self.hand_id)
        return np.vstack((jacp[:, self.arm_dofs], jacr[:, self.arm_dofs]))

    @staticmethod
    def orientation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
        return 0.5 * sum(np.cross(current[:, i], target[:, i]) for i in range(3))

    def solve_ik(
        self,
        q_seed: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        *,
        max_iterations: int = 500,
        damping: float = 0.035,
        position_tolerance: float = 2.5e-3,
        orientation_tolerance: float = 1.5e-2,
    ) -> IKResult:
        data = mujoco.MjData(self.model)
        q = np.clip(np.asarray(q_seed, dtype=float), self.lower + 0.01, self.upper - 0.01)
        center = 0.5 * (self.lower + self.upper)

        for iteration in range(1, max_iterations + 1):
            data.qpos[self.arm_qpos] = q
            data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, data)
            position, rotation = self.pose(data)
            ep = np.asarray(target_position) - position
            er = self.orientation_error(rotation, np.asarray(target_rotation))
            p_norm = float(np.linalg.norm(ep))
            r_norm = float(np.linalg.norm(er))
            if p_norm < position_tolerance and r_norm < orientation_tolerance:
                return IKResult(q.copy(), p_norm, r_norm, iteration, True)

            error = np.concatenate((ep, 0.55 * er))
            jac = self.jacobian(data)
            weighted_jac = jac.copy()
            weighted_jac[3:] *= 0.55
            lhs = weighted_jac @ weighted_jac.T + damping**2 * np.eye(6)
            dq = weighted_jac.T @ np.linalg.solve(lhs, error)

            # Gentle null-space centering improves clearance from joint limits.
            pinv = weighted_jac.T @ np.linalg.solve(lhs, np.eye(6))
            null = np.eye(7) - pinv @ weighted_jac
            dq += null @ (0.025 * (center - q))
            max_step = 0.12
            scale = min(1.0, max_step / max(float(np.max(np.abs(dq))), 1e-9))
            q = np.clip(q + scale * dq, self.lower + 0.008, self.upper - 0.008)

        data.qpos[self.arm_qpos] = q
        mujoco.mj_forward(self.model, data)
        position, rotation = self.pose(data)
        ep = float(np.linalg.norm(np.asarray(target_position) - position))
        er = float(np.linalg.norm(self.orientation_error(rotation, np.asarray(target_rotation))))
        return IKResult(q.copy(), ep, er, max_iterations, False)

