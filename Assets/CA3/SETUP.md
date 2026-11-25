# CA3 setup — Kinova Gen3

Arm: **Kinova Gen3** (`mujoco_menagerie/kinova_gen3/gen3.xml`). It has 7 hinge joints and 7 position actuators, so STATE = 14 and ACTION = 7.

| MJCF item | Value used |
|---|---|
| Joints | `joint_1` … `joint_7` |
| Actuators | `joint_1` … `joint_7` (`<position>`; large kp=2000 kv=100, small kp=500 kv=50) |
| End effector | body `bracelet_link` (site `pinch_site` sits below it) |
| Home keyframe | `0 0.2618 3.1416 -2.2689 0 0.9599 1.5708` |
| Ranges | joint_2 ±2.24, joint_4 ±2.57, joint_6 ±2.09; joints 1/3/5/7 are continuous, so they use home ± π |
| Gripper | none (leave `ArmAgent.gripper` empty) |

The same values appear in `ArmAgent.cs`, `ArmJointController.cs` and `train_controller.py` (`HOME`, `JOINT_LO`, `JOINT_HI`). If you change one, change all three.

## 1. Unity
1. Install the MuJoCo Unity plugin (version from the assignment) and the Inference Engine package (`com.unity.ai.inference`).
2. Copy everything in this folder, `.meta` files included, to `Assets/CA3/`. The recorder writes to `Assets/CA3/Data`. If you use another folder, change `RolloutRecorder.dataFolder`.
3. The project includes the Menagerie Kinova Gen3 files and MuJoCo 3.17.0. On first editor compile, `CA3/Import and Configure Gen3 Scene` imports `scene.xml`, creates `Assets/CA3/Scenes/Gen3Scene.unity`, and adds ArmAgent, ArmJointController, RolloutRecorder (disabled), WorldModelController (disabled), and IkGoalProvider with a collider-free TrackingTarget. The same menu item can be run again in an empty scene.
   - The Unity package's **Assets > Import MuJoCo Scene** is still available for manual imports.
   - ArmAgent finds joints, actuators and `bracelet_link` by name in `Awake`. Check the Console for "not found" errors.
4. Leave Fixed Timestep at 0.02. Each FixedUpdate is one MuJoCo step, and the defaults (hold 3 s = 150 steps, `MAX_DELTA` 0.05 rad/step, horizon 200) assume this.

## 2. Collect data
Follow "Collect rollout" in `SCENE_MODES.txt`. Press Play. The arm teleports to home, then tracks a new random joint goal every 3 s. The Console logs progress every 5000 steps, and `Data/rollout.csv` is written at 70000 steps (also on Stop, if you end early). For faster collection, raise **Time Scale** in Project Settings > Time.

A 14,151-transition rollout is included at `Data/rollout.csv`. It has 35 columns: `q[7], qd[7], q*[7], q'[7], qd'[7]`. The saved scene is configured for learned control and IK; to recollect data, enable RolloutRecorder, disable IkGoalProvider and WorldModelController, and enable ArmJointController goal resampling.

CSV columns (35): `q[7], qd[7], q*[7], q'[7], qd'[7]`.

## 3. Train (Python 3.12 venv, already created at `CA3/.venv`)
```bash
cd Assets/CA3
/Users/dishabenpate/Documents/PhysicalAI/CA3/.venv/bin/python train.py             # ~1–3 min
/Users/dishabenpate/Documents/PhysicalAI/CA3/.venv/bin/python train_controller.py  # ~20 min on this Mac
```
These write `Data/dynamics.pt`, `dynamics.onnx`, `norm.json`, `norm.npz` and `controller.json`. Trained artifacts from the included rollout are present. The report records the measured validation errors and held-out imagined returns.

(This is an Intel Mac. torch 2.2.2 is the last build for it and needs Python ≤ 3.12, which is why the venv uses 3.12.)

## 4. Hook up the trained files
- WorldModelController: Controller Json = `Data/controller.json`. Keep `actionSmoothing` 0.2 and `maxDeltaPerStep` 0.05 so they match training.
- WorldModelDriver: Model Asset = `Data/dynamics.onnx`, Norm Json = `Data/norm.json`. Ghost Arm = a copy of the imported arm prefab (visual only; the script strips the MuJoCo components). Ghost Material = a transparent material.
- **Step 3 check:** enable WorldModelController with IkGoalProvider disabled. The controller takes over `emitActions`, and the random goals keep coming from ArmJointController.

WorldModelDriver is attached to the saved scene and references `dynamics.onnx` and `norm.json`. If Ghost Arm is unassigned, it makes a visual-only clone of the imported arm.

The imported `wrist` sensor Camera must stay disabled for the Game view: it has render depth 0 and would cover the scene camera. The saved scene disables it, and `ArmAgent` disables non-MainCamera cameras at runtime. Its startup framing places the Gen3 arm in view.

## 5. IK target
Create an empty GameObject or a Sphere named `TrackingTarget` near the arm, about 0.5 m in front and 0.4 m up. **Remove its collider.** The saved scene already has `TrackingTarget` assigned to `IkGoalProvider.target`, with IkGoalProvider + WorldModelController enabled and RolloutRecorder disabled. While playing, drag the target in the Scene view. The gizmo turns green when the target is reachable (residual < 1 cm) and red when it isn't.

How the pieces connect: IkGoalProvider (order −7) runs damped least squares on `mj_jacBody` for `bracelet_link`. It uses a null-space pull toward home and warm-starts from the previous solution. Each step it writes joint goals with `ArmJointController.SetTarget`. WorldModelController (order −5) reads those goals and outputs smoothed joint targets. ArmAgent (order −1) writes them to `data->ctrl` before MjScene steps.
