from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class PlanResult:
    path: list[np.ndarray]
    iterations: int
    collision_checks: int
    success: bool


class _Tree:
    def __init__(self, root: np.ndarray, from_start: bool):
        self.vertices = [np.asarray(root, dtype=float).copy()]
        self.parents = [-1]
        self.from_start = from_start

    def nearest(self, q: np.ndarray) -> int:
        distance = np.linalg.norm(np.asarray(self.vertices) - q, axis=1)
        return int(np.argmin(distance))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.vertices.append(np.asarray(q, dtype=float).copy())
        self.parents.append(parent)
        return len(self.vertices) - 1

    def branch(self, index: int) -> list[np.ndarray]:
        result: list[np.ndarray] = []
        while index >= 0:
            result.append(self.vertices[index])
            index = self.parents[index]
        return result[::-1]


class RRTConnect:
    """Small deterministic RRT-Connect implementation for 7-DoF joint space."""

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        collision_free: Callable[[np.ndarray], bool],
        *,
        step_size: float = 0.16,
        edge_resolution: float = 0.045,
        max_iterations: int = 3500,
        seed: int = 7,
    ):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        self.is_free = collision_free
        self.step_size = step_size
        self.edge_resolution = edge_resolution
        self.max_iterations = max_iterations
        self.rng = np.random.default_rng(seed)
        self.checks = 0

    def _free(self, q: np.ndarray) -> bool:
        self.checks += 1
        return bool(self.is_free(q))

    def _edge_free(self, qa: np.ndarray, qb: np.ndarray) -> bool:
        distance = float(np.linalg.norm(qb - qa))
        count = max(2, int(np.ceil(distance / self.edge_resolution)) + 1)
        return all(self._free((1.0 - s) * qa + s * qb) for s in np.linspace(0.0, 1.0, count)[1:])

    def _extend(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        parent = tree.nearest(target)
        q_near = tree.vertices[parent]
        delta = target - q_near
        distance = float(np.linalg.norm(delta))
        if distance < 1e-10:
            return "reached", parent
        q_new = q_near + delta * min(1.0, self.step_size / distance)
        if not self._edge_free(q_near, q_new):
            return "trapped", parent
        index = tree.add(q_new, parent)
        return ("reached" if distance <= self.step_size else "advanced"), index

    def _connect(self, tree: _Tree, target: np.ndarray) -> tuple[str, int]:
        while True:
            status, index = self._extend(tree, target)
            if status != "advanced":
                return status, index

    def _assemble(self, a: _Tree, ia: int, b: _Tree, ib: int) -> list[np.ndarray]:
        branch_a = a.branch(ia)
        branch_b = b.branch(ib)
        if a.from_start:
            return branch_a + branch_b[-2::-1]
        return branch_b + branch_a[-2::-1]

    def plan(self, start: np.ndarray, goal: np.ndarray) -> PlanResult:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        self.checks = 0
        if not self._free(start) or not self._free(goal):
            return PlanResult([], 0, self.checks, False)
        if self._edge_free(start, goal):
            return PlanResult([start.copy(), goal.copy()], 0, self.checks, True)

        a, b = _Tree(start, True), _Tree(goal, False)
        for iteration in range(1, self.max_iterations + 1):
            sample = goal if self.rng.random() < 0.12 else self.rng.uniform(self.lower, self.upper)
            status, ia = self._extend(a, sample)
            if status != "trapped":
                status_b, ib = self._connect(b, a.vertices[ia])
                if status_b == "reached":
                    path = self._assemble(a, ia, b, ib)
                    return PlanResult(self.shortcut(path), iteration, self.checks, True)
            a, b = b, a
        return PlanResult([], self.max_iterations, self.checks, False)

    def shortcut(self, path: list[np.ndarray], attempts: int = 160) -> list[np.ndarray]:
        path = [q.copy() for q in path]
        for _ in range(attempts):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.integers(0, len(path), size=2))
            if j <= i + 1:
                continue
            if self._edge_free(path[i], path[j]):
                path = path[: i + 1] + path[j:]
        return path


def smooth_joint_path(
    waypoints: list[np.ndarray],
    dt: float,
    max_speed: float = 0.72,
    minimum_duration: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Cubic smoothstep interpolation with zero velocity at every waypoint."""
    positions: list[np.ndarray] = []
    velocities: list[np.ndarray] = []
    for qa, qb in zip(waypoints[:-1], waypoints[1:]):
        delta = np.asarray(qb) - np.asarray(qa)
        duration = max(minimum_duration, float(np.max(np.abs(delta))) / max_speed * 1.5)
        steps = max(2, int(np.ceil(duration / dt)))
        phase = np.linspace(0.0, 1.0, steps, endpoint=False)
        blend = 3.0 * phase**2 - 2.0 * phase**3
        dblend = (6.0 * phase - 6.0 * phase**2) / duration
        positions.extend(np.asarray(qa) + blend[:, None] * delta)
        velocities.extend(dblend[:, None] * delta)
    positions.append(np.asarray(waypoints[-1]).copy())
    velocities.append(np.zeros_like(waypoints[-1]))
    return np.asarray(positions), np.asarray(velocities)


