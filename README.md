# Classical Collision-Free Pick and Place in MuJoCo

This project demonstrates a fully classical manipulation stack for a Franka
Emika Panda:

- randomized cup poses observed through a fixed overhead RGB-D camera;
- classical color segmentation and calibrated depth back-projection;
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

- `camera_rgb.png`: raw perception-camera observation;
- `camera_detection.png`: detected cup pixels and estimated image center;
- `pick_place.mp4`: overview video;
- `trajectory.png`: tracking and task-space plots;
- `metrics.json`: success, collision, tracking, and planner statistics;
- `trajectory.npz`: raw time-series data.

## Verified baseline

Four independently randomized cup poses and RRT seeds (`0, 1, 2, 7`) were
evaluated in the supplied scene. All four completed the task using only the
RGB-D position estimate for grasp planning. Simulator ground truth is retained
only to report perception error. The final seed-7 artifact reports:

| Metric | Result |
|---|---:|
| RGB-D cup-position error | 7.14 mm |
| placement XY error | 16.29 mm |
| physical forbidden contacts | 0 |
| simulated task duration | 35.37 s |

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

## Perception boundary

The current detector is a transparent classical baseline for the synthetic
blue cup: it segments blue pixels, selects the largest connected component,
uses the component bounding-box center, and combines it with the depth image
and calibrated MuJoCo camera pose to recover a world-frame 3-D position. It is
not yet a category-level cup detector. The `RgbdCupDetector` interface is the
intended replacement point for YOLO-World/Grounding DINO plus depth, while the
planner and controller can remain unchanged.

