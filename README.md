# Classical Collision-Free Pick and Place in MuJoCo

This project demonstrates a fully classical manipulation stack for a Franka
Emika Panda:

- randomized cup poses observed through fixed and eye-in-hand RGB-D cameras;
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
model, injects the `wrist_rgbd` camera into the Panda hand, then installs this
project's `assets/pick_place.xml` beside the upstream MJCF. Running it again is
safe.

Artifacts are written to `outputs/`:

- `fixed_camera_rgb.png` / `fixed_camera_detection.png`: coarse observation;
- `wrist_camera_rgb.png` / `wrist_camera_detection.png`: eye-in-hand refinement;
- `perception.json`: coarse/refined estimates and evaluation-only ground truth;
- `pick_place.mp4`: overview video;
- `trajectory.png`: tracking and task-space plots;
- `metrics.json`: success, collision, tracking, and planner statistics;
- `trajectory.npz`: raw time-series data.

## Verified baseline

Four independently randomized reachable cup poses and RRT seeds (`0, 1, 2, 7`)
were evaluated in the supplied scene. All four completed the task using the
eye-in-hand-refined position for grasp planning. Simulator ground truth is
retained only to report perception error. The final seed-7 artifact reports:

| Metric | Result |
|---|---:|
| fixed-camera cup-position error | 11.26 mm |
| wrist-camera cup-position error | 4.26 mm |
| bilateral fingertip contact before lock | true / true |
| object displacement when lock is enabled | 0.11 µm |
| placement XY error | 12.89 mm |
| physical forbidden contacts | 0 |
| cup/handle–obstacle collision steps | 0 |
| minimum cup/handle–obstacle signed distance | 37.09 mm |
| joint tracking RMSE | 0.0203 rad |

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
interpolated and checked through MuJoCo's collision pipeline. After the real
finger closure and lift, the transfer path is planned online using the
measured hand-to-cup transform. At every candidate configuration, the complete
physical cup body and handle are moved with the hand and checked with
`mj_geomDistance`. The planner requires 25 mm nominal obstacle clearance and
10 mm table clearance; execution-time signed distance is recorded separately
so tracking error cannot silently turn a safe reference path into a payload
collision. The low-level controller applies

```text
tau = M(q) [Kp (q_ref - q) + Kd (dq_ref - dq)] + h(q, dq),
```

where `M` is the joint-space inertia matrix and `h` is MuJoCo's bias force
(gravity plus Coriolis/centrifugal terms). Joint torques are clipped to the
Panda limits.

The fingers first close through the original Panda contact model. A grasp is
accepted only when both fingertip groups contact the cup. The cup-side weld
site is then re-expressed at the current hand pose, so enabling the constraint
does not move the physical cup. The constraint starts soft and ramps to its
tracking stiffness over 0.30 s. This retains a stable grasp during aggressive
RRT motions without the previous snap-to-site artifact. A pure-contact grasp
can still be tested by omitting the stabilization call.

## Perception boundary

The fixed camera supplies a coarse position used to reach an observation pose.
The hand-mounted camera then observes the cup again and its estimate is used to
recompute IK and collision-free plans online. Both currently use a transparent
classical detector for the synthetic blue cup: color segmentation, connected
components, metric depth, and calibrated camera extrinsics. This is not yet a
category-level cup detector. `RgbdCupDetector` is the replacement point for
YOLO-World/Grounding DINO plus depth, while the planner and controller remain
unchanged.

## Teleoperation and VLA data

Install the optional gamepad dependency and start Cartesian teleoperation:

```bash
python -m pip install -e '.[teleop]'
python scripts/teleop_collect.py --device gamepad --episodes 10
```

The input layer also supports `--device keyboard`. Motion uses a deadman
button, DLS IK, joint-limit clipping, MuJoCo collision rejection, and the same
payload-aware cup/handle checker as the autonomous demo. Episodes can be saved
or discarded from the controller.

Every accepted episode synchronizes two RGB views, robot state, velocity,
end-effector pose, the actual commanded joint target, gripper command,
timestamp, language task, and success/collision metadata at 20 Hz. Convert the
auditable raw format to LeRobot for openpi/π0.5:

```bash
python scripts/inspect_raw_dataset.py datasets/raw
python -m pip install -e '.[dataset]'
python scripts/convert_to_lerobot.py \
  --raw-dir datasets/raw \
  --repo-id <hf-user>/panda-cup-mujoco
```

See [docs/VLA_DATA_COLLECTION.md](docs/VLA_DATA_COLLECTION.md) for controller
mapping, schema, QA, train/validation splitting, and the recommended
scripted-expert → human teleoperation → HIL/DAgger collection progression.

The cloud smoke dataset contains four successful randomized episodes: 3,309
aligned steps and 6,618 RGB frames. It reloads with LeRobot 0.6.1 as four
episodes at 20 Hz with two `(3, 240, 320)` observations, 8-D state and 8-D
absolute joint-position/gripper actions.

