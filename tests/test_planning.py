from __future__ import annotations

import unittest

import numpy as np

from classic_pick_place.planning import RRTConnect, smooth_joint_path


class PlanningTests(unittest.TestCase):
    def test_rrt_routes_around_circle(self) -> None:
        def collision_free(q: np.ndarray) -> bool:
            return np.linalg.norm(q) > 0.32

        planner = RRTConnect(
            lower=np.array([-1.0, -1.0]),
            upper=np.array([1.0, 1.0]),
            collision_free=collision_free,
            step_size=0.12,
            edge_resolution=0.02,
            seed=4,
        )
        result = planner.plan(np.array([-0.9, 0.0]), np.array([0.9, 0.0]))
        self.assertTrue(result.success)
        self.assertGreaterEqual(len(result.path), 3)
        for qa, qb in zip(result.path[:-1], result.path[1:]):
            for alpha in np.linspace(0.0, 1.0, 80):
                self.assertTrue(collision_free((1.0 - alpha) * qa + alpha * qb))

    def test_smooth_path_endpoints(self) -> None:
        waypoints = [np.zeros(3), np.array([0.4, -0.2, 0.1])]
        q, dq = smooth_joint_path(waypoints, dt=0.01, max_speed=0.5)
        np.testing.assert_allclose(q[0], waypoints[0])
        np.testing.assert_allclose(q[-1], waypoints[-1])
        np.testing.assert_allclose(dq[0], 0.0)
        np.testing.assert_allclose(dq[-1], 0.0)


if __name__ == "__main__":
    unittest.main()


