import json
from pathlib import Path
import time
import cma
import numpy as np
import torch
from torch import nn

from train import build_model

ROOT = Path(__file__).resolve().parent / "Data"
STATE = 14
ACTION = 7
FEAT = STATE
N_PARAMS = ACTION * FEAT + ACTION

# Kinova Gen3 "home" keyframe and joint ranges; must match ArmJointController in Unity.
# joint_2/4/6 ranges come from the MJCF; continuous joints 1/3/5/7 use home +/- pi.
HOME = np.array([0.0, 0.26179939, 3.14159265, -2.26892803, 0.0, 0.95993109, 1.57079633], dtype=np.float32)
JOINT_LO = np.array([-np.pi, -2.24, 0.0, -2.57, -np.pi, -2.09, -np.pi / 2], dtype=np.float32)
JOINT_HI = np.array([np.pi, 2.24, 2 * np.pi, 2.57, np.pi, 2.09, 3 * np.pi / 2], dtype=np.float32)
RANDOM_SCALE = 0.35

VEL_PENALTY = 0.05
ACTION_SMOOTH_PENALTY = 0.1
ACTION_SMOOTH = 0.2
MAX_DELTA = 0.05
HORIZON = 200
EPISODES = 16
CMA_ITERS = 60
CMA_SIGMA = 0.05
MAX_QD = 10.0

# Start states for imagined rollouts, drawn from the recorded data (set in main)
START_STATES = None


def build_dynamics():
    model = build_model()
    path = ROOT / "dynamics.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_norm():
    path = ROOT / "norm.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run train.py first.")
    data = np.load(path)
    return data["x_mean"], data["x_std"], data["y_mean"], data["y_std"]


def unpack(theta):
    w = theta[: ACTION * FEAT].reshape(ACTION, FEAT)
    b = theta[ACTION * FEAT :]
    return w, b


def policy(state, goal, w, b):
    # Linear policy on [q - goal, qd] that outputs a joint-target action around the goal.
    # theta = 0 means "command the goal directly", so CMA-ES learns corrections on top of that.
    features = np.concatenate([state[..., 0:ACTION] - goal, state[..., ACTION:STATE]], axis=-1)
    return (goal + features @ w.T + b).astype(np.float32)


@torch.no_grad()
def step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std):
    x = np.concatenate([state, action], axis=-1)
    x_n = (x - x_mean) / x_std
    delta = model(torch.as_tensor(x_n, dtype=torch.float32)).numpy() * y_std + y_mean
    nxt = (state + delta).astype(np.float32)
    # Keep imagination inside the region the model was trained on
    nxt[..., 0:ACTION] = np.clip(nxt[..., 0:ACTION], JOINT_LO - 0.5, JOINT_HI + 0.5)
    nxt[..., ACTION:STATE] = np.clip(nxt[..., ACTION:STATE], -MAX_QD, MAX_QD)
    return nxt


def sample_goal(rng):
    span = (JOINT_HI - JOINT_LO) * 0.5 * RANDOM_SCALE
    goal = HOME + rng.uniform(-span, span)
    return np.clip(goal, JOINT_LO, JOINT_HI).astype(np.float32)


def smooth_action(prev, raw):
    blended = (1.0 - ACTION_SMOOTH) * prev + ACTION_SMOOTH * raw
    delta = np.clip(blended - prev, -MAX_DELTA, MAX_DELTA)
    return np.clip(prev + delta, JOINT_LO, JOINT_HI).astype(np.float32)


def start_state(rng):
    if START_STATES is not None and len(START_STATES) > 0:
        return START_STATES[rng.integers(len(START_STATES))].copy()
    state = np.zeros(STATE, dtype=np.float32)
    state[0:ACTION] = sample_goal(rng)
    return state


def rollout(model, theta, x_mean, x_std, y_mean, y_std, rng):
    # One imagined episode: random start, random joint goal, the policy acts through the same
    # smoothing as WorldModelController, and the world model supplies the next state.
    w, b = unpack(theta)
    state = start_state(rng)
    goal = sample_goal(rng)
    prev = state[0:ACTION].copy()
    total = 0.0
    for _ in range(HORIZON):
        action = smooth_action(prev, policy(state, goal, w, b))
        state = step_dynamics(model, state, action, x_mean, x_std, y_mean, y_std)
        q_err = state[0:ACTION] - goal
        qd = state[ACTION:STATE]
        cost = (np.mean(q_err ** 2)
                + VEL_PENALTY * np.mean(qd ** 2)
                + ACTION_SMOOTH_PENALTY * np.mean((action - prev) ** 2))
        total -= cost
        prev = action
    return total / HORIZON


def fitness(theta, model, x_mean, x_std, y_mean, y_std, seed):
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(EPISODES):
        returns.append(rollout(model, theta, x_mean, x_std, y_mean, y_std, rng))

    return -float(np.mean(returns))


def save_controller(theta):
    w, b = unpack(theta)
    payload = {
        "feature": "q_minus_goal_and_qd",
        "state_size": STATE,
        "action_size": ACTION,
        "W": w.astype(float).reshape(-1).tolist(),
        "b": b.astype(float).tolist(),
        "joint_lo": JOINT_LO.astype(float).tolist(),
        "joint_hi": JOINT_HI.astype(float).tolist(),
        "home": HOME.astype(float).tolist(),
        "vel_penalty": VEL_PENALTY,
        "horizon": HORIZON,
    }
    path = ROOT / "controller.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {path}")


def load_start_states():
    path = ROOT / "rollout.csv"
    if not path.exists():
        return None
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return data[:, 0:STATE]


def main():
    global START_STATES
    torch.set_num_threads(1)
    model = build_dynamics()
    x_mean, x_std, y_mean, y_std = load_norm()
    START_STATES = load_start_states()

    def imagined_return(theta, seed):
        return -fitness(theta, model, x_mean, x_std, y_mean, y_std, seed)

    baseline = np.zeros(N_PARAMS)
    eval_seeds = range(1000, 1004)
    base_ret = np.mean([imagined_return(baseline, s) for s in eval_seeds])
    print(f"baseline (command goal directly) imagined return: {base_ret:.5f}")

    es = cma.CMAEvolutionStrategy(baseline, CMA_SIGMA, {"maxiter": CMA_ITERS, "seed": 1, "verbose": -9})
    best_theta, best_fit = baseline.copy(), np.inf
    for it in range(CMA_ITERS):
        candidates = es.ask()
        # Same seed for every candidate in a generation -> fair comparison (common random numbers)
        fits = [fitness(np.asarray(c), model, x_mean, x_std, y_mean, y_std, seed=it) for c in candidates]
        es.tell(candidates, fits)
        i = int(np.argmin(fits))
        if fits[i] < best_fit:
            best_fit, best_theta = fits[i], np.asarray(candidates[i]).copy()
        print(f"iter {it:3d}  best {-min(fits):.5f}  mean {-np.mean(fits):.5f}  sigma {es.sigma:.4f}")

    # Pick the final controller on held-out seeds
    options = {"baseline": baseline, "best": best_theta, "mean": np.asarray(es.result.xfavorite)}
    scores = {k: np.mean([imagined_return(v, s) for s in eval_seeds]) for k, v in options.items()}
    for k, v in scores.items():
        print(f"held-out imagined return [{k}]: {v:.5f}")
    choice = max(scores, key=scores.get)
    print(f"saving '{choice}' controller")
    save_controller(options[choice])


if __name__ == "__main__":
    current_time = time.time()
    main()
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
