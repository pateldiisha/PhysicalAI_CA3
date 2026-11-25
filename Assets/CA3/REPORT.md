# CA3 Report

**Name:**  Disha Patel
**Unity version:** Unity 6.3 LTS (6000.3.24f1)  
**MuJoCo plugin version:** 3.17.0  
**Chosen arm (Menagerie):** Kinova Gen3 (`kinova_gen3/gen3.xml`)  
**Gallery / repo link:** [MuJoCo Menagerie Kinova Gen3](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kinova_gen3)

## Why this arm
The Kinova Gen3 is a seven-axis robot arm with a useful redundant degree of freedom for reaching around obstacles or selecting comfortable joint postures. It is a robot arm from the MuJoCo Menagerie and is distinct from the Panda lecture example.

## Pipeline summary
The Unity scene imports the Menagerie MJCF and uses MuJoCo simulation for the arm. A random joint-goal rollout records 14,151 transitions in `Data/rollout.csv`; a normalized MLP predicts state deltas, and a CMA-ES search trains a linear joint-goal policy in the learned model. The saved scene attaches the learned controller, model driver, and damped-least-squares IK goal provider to a collider-free movable target.

## World model results
- DOF (`ACTION`): 7 (STATE = 14: q + qd)
- Training transitions: 14,151 (12,735 train / 1,416 held-out)
- One-step joint RMSE: 0.06083 rad
- Open-loop 10 / 50 / 100-step joint RMSE: 0.13274 / 0.18947 / 0.23343 rad

## Controller results
- CMA-ES iterations: 60; policy selected by held-out imagined return
- Baseline imagined return (direct goal command): -0.20219
- Selected policy imagined return: -0.16471
- Qualitative tracking: verified in Unity Play mode. I temporarily shifted `TrackingTarget` by 0.2 m on Unity X; the arm moved toward the shifted target, then adjusted back after the target was restored. The probe was removed afterward. The Game view shows the Gen3 arm, collider-free target marker, and forecast trace; `WorldModelController` reports ready.

## WorldModelDriver
The driver loads `dynamics.onnx` and `norm.json`, predicts joint-state deltas, and periodically reanchors its imagined state when drift exceeds the threshold. It draws a forecast and deviation line and uses an auto-generated visual-only clone of the arm when no ghost prefab is assigned. The imported wrist sensor camera was enabled above the scene camera and covered the Game view; it is now disabled in the scene and by `ArmAgent` at runtime. Play mode now shows the robot and forecast with the scene camera framed on the arm.

## IK + live target
- IK method: damped least squares on the MuJoCo body Jacobian of `bracelet_link`, with a null-space pull toward home
- `TrackingTarget` has no collider and is wired to `IkGoalProvider`
- Target reachability is exposed via `reachable` and `residual`; the gizmo is green below 1 cm residual and red otherwise

## Design question
Rollout collection can stay in joint space because the learned dynamics map joint state and joint target to the next joint state. Cartesian deployment first converts the end-effector position goal into a joint target with IK; the learned controller then tracks that joint target using the same state/action interface learned from the rollout.
