# Classical Collision-Free Pick and Place in MuJoCo

This project demonstrates a fully classical manipulation stack for a Franka
Emika Panda:

- damped-least-squares inverse kinematics;
- joint-space RRT-Connect with MuJoCo collision checking;
- time-scaled joint trajectories;
- computed-torque tracking with gravity/Coriolis compensation;
- a deterministic pick/transfer/place state machine;
- physical finger closure followed by a disclosed, switchable rigid-grasp
  constraint for path-tracking experiments;
- headless video, trajectory plots, and machine-readable metrics.

The robot description is the Franka Panda model distributed with the Google
DeepMind MuJoCo Menagerie. The setup script downloads the model from the
official repository into an ignored local directory, so third-party meshes are
not duplicated in this repository.

## Run

```bash
python scripts/setup_assets.py
python -m pip install -e .
export MUJOCO_GL=egl
python scripts/run_demo.py --render --seed 7
```

`setup_assets.py` performs a shallow sparse checkout containing only the Panda
model, then installs this project's `assets/pick_place.xml` beside the upstream
MJCF. Running it again is safe.

Artifacts are written to `outputs/`:

- `pick_place.mp4`: overview video;
- `trajectory.png`: tracking and task-space plots;
- `metrics.json`: success, collision, tracking, and planner statistics;
- `trajectory.npz`: raw time-series data.

## Verified baseline

Four independent RRT seeds (`0, 1, 2, 7`) were evaluated in the supplied
scene. All four completed the task. The final seed-7 artifact reports:

| Metric | Result |
|---|---:|
| placement XY error | 5.18 mm |
| physical forbidden contacts | 0 |
| minimum robot-obstacle signed distance | 14.73 mm |
| joint tracking RMSE | 0.0242 rad |
| simulated task duration | 28.46 s |

`safety_margin_contact_count` is intentionally reported separately: it counts
MuJoCo contacts generated inside the positive 15 mm planning margin, before
physical surfaces touch. A physical collision requires negative signed
distance and is counted in `forbidden_contact_count`.

Run a fast, non-rendered validation with:

```bash
python scripts/run_demo.py --seed 7
```

## Method

The arm path is planned in configuration space. Every candidate edge is
interpolated and checked through MuJoCo's collision pipeline. During object
transfer, the carried cube is approximated by an inflated payload sphere so
the planner also keeps the payload clear of the obstacle. The low-level
controller applies

```text
tau = M(q) [Kp (q_ref - q) + Kd (dq_ref - dq)] + h(q, dq),
```

where `M` is the joint-space inertia matrix and `h` is MuJoCo's bias force
(gravity plus Coriolis/centrifugal terms). Joint torques are clipped to the
Panda limits.

The fingers first close through the original Panda contact model. Once closure
has completed, a disabled-by-default site weld is switched on to represent a
stable force-closure grasp during aggressive RRT motions; it is switched off
immediately before opening. This deliberately isolates motion-planning and
tracking quality from gripper-contact tuning. A pure-contact grasp can be
tested by removing the two `set_grasp_constraint` calls in `run_demo.py`.

