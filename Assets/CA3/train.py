import numpy as np
import torch
import time
from pathlib import Path
from torch import nn

ROOT = Path(__file__).resolve().parent / "Data"
STATE = 14
ACTION = 7

HIDDEN = 256
EPOCHS = 40
BATCH = 512
LR = 1e-3
VAL_FRACTION = 0.1
MULTI_HORIZON = 10
MULTI_BATCH = 128
MULTI_WEIGHT = 0.5


def mean_std(z):
    return z.mean(0), z.std(0) + 1e-6


def norm(z, mean, std):
    return (z - mean) / std


def save_norm_json(x_mean, x_std, y_mean, y_std):
    arrays = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    lines = ["{"]
    keys = list(arrays)
    for index, key in enumerate(keys):
        values = ", ".join(f"{float(value):.8f}" for value in arrays[key])
        comma = "," if index < len(keys) - 1 else ""
        lines.append(f'  "{key}": [{values}]{comma}')
    lines.append("}")
    (ROOT / "norm.json").write_text("\n".join(lines) + "\n")


def load_rollout():
    path = ROOT / "rollout.csv"
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"No rollout at {path}. Enable RolloutRecorder and Play.")
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != STATE + ACTION + STATE:
        raise ValueError(f"Expected {STATE + ACTION + STATE} columns, found {data.shape[1]}.")
    return data


def predict_delta(model, state, action, x_mean, x_std, y_mean, y_std):
    x = np.concatenate([state, action], axis=-1)
    x_n = (x - x_mean) / x_std
    with torch.no_grad():
        delta_n = model(torch.as_tensor(x_n)).numpy()
    return delta_n * y_std + y_mean


def joint_rmse(pred, true):
    err = pred[..., 0:ACTION] - true[..., 0:ACTION]
    return float(np.sqrt((err ** 2).sum(axis=-1).mean()))


def open_loop_rmse(model, data, start, x_mean, x_std, y_mean, y_std, horizon):
    end = len(data) - horizon
    if end <= start:
        return float("nan")
    idx = np.arange(start, end)
    if len(idx) > 2000:
        idx = idx[:: max(1, len(idx) // 2000)]
    state = data[idx, 0:STATE].copy()
    for t in range(horizon):
        action = data[idx + t, STATE:STATE + ACTION]
        state = state + predict_delta(model, state, action, x_mean, x_std, y_mean, y_std)
    return joint_rmse(state, data[idx + horizon, 0:STATE])


def build_model():
    # MLP: normalized [q, qd, q*] -> normalized state delta [dq, dqd]
    return nn.Sequential(
        nn.Linear(STATE + ACTION, HIDDEN),
        nn.SiLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.SiLU(),
        nn.Linear(HIDDEN, HIDDEN),
        nn.SiLU(),
        nn.Linear(HIDDEN, STATE),
    )


def multi_step_loss(model, data, split, x_mean, x_std, y_mean, y_std, horizon, batch, loss_fn):
    # Roll the model forward on its own predictions for `horizon` steps from random training
    # starts, feeding the recorded actions, and penalize the state error at every step.
    # data, x_mean, ... are torch tensors here.
    starts = torch.randint(0, split - horizon, (batch,))
    state = data[starts, 0:STATE]
    s_std = x_std[0:STATE]
    loss = 0.0
    for t in range(horizon):
        action = data[starts + t, STATE:STATE + ACTION]
        x_n = (torch.cat([state, action], dim=-1) - x_mean) / x_std
        state = state + model(x_n) * y_std + y_mean
        true = data[starts + t, STATE + ACTION:]
        loss = loss + loss_fn(state / s_std, true / s_std)
    return loss / horizon


def train_world_model(data):
    torch.manual_seed(0)
    np.random.seed(0)

    # Inputs [q, qd, q*], targets are deltas s' - s
    x = data[:, 0:STATE + ACTION]
    y = data[:, STATE + ACTION:] - data[:, 0:STATE]

    # Time-ordered split so validation is a contiguous, unseen stretch of the rollout
    split = int(len(data) * (1.0 - VAL_FRACTION))
    x_mean, x_std = mean_std(x[:split])
    y_mean, y_std = mean_std(y[:split])
    x_n = norm(x, x_mean, x_std)
    y_n = norm(y, y_mean, y_std)
    print(f"{len(data)} transitions, train {split}, val {len(data) - split}")

    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    steps_per_epoch = max(1, split // BATCH)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * steps_per_epoch)
    loss_fn = nn.MSELoss()

    data_t = torch.as_tensor(data)
    x_tr = torch.as_tensor(x_n[:split])
    y_tr = torch.as_tensor(y_n[:split])
    x_va = torch.as_tensor(x_n[split:])
    y_va = torch.as_tensor(y_n[split:])
    stats = [torch.as_tensor(v) for v in (x_mean, x_std, y_mean, y_std)]

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(split)
        total_one = total_multi = 0.0
        # Ramp in the multi-step term once one-step predictions are sensible
        multi_weight = MULTI_WEIGHT * min(1.0, epoch / max(1, EPOCHS // 4))
        for i in range(steps_per_epoch):
            idx = perm[i * BATCH:(i + 1) * BATCH]
            one = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss = one
            if multi_weight > 0:
                multi = multi_step_loss(model, data_t, split, *stats, MULTI_HORIZON, MULTI_BATCH, loss_fn)
                loss = loss + multi_weight * multi
                total_multi += float(multi)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total_one += float(one)

        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(x_va), y_va))
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  one-step {total_one / steps_per_epoch:.5f}  "
                  f"multi {total_multi / steps_per_epoch:.5f}  val {val:.5f}")

    model.eval()
    val_states = data[split:, 0:STATE]
    val_actions = data[split:, STATE:STATE + ACTION]
    pred_next = val_states + predict_delta(model, val_states, val_actions, x_mean, x_std, y_mean, y_std)
    print(f"one-step joint RMSE (val): {joint_rmse(pred_next, data[split:, STATE + ACTION:]):.5f} rad")
    for horizon in (10, 50, 100):
        rmse = open_loop_rmse(model, data, split, x_mean, x_std, y_mean, y_std, horizon)
        print(f"open-loop {horizon:3d}-step joint RMSE (val): {rmse:.5f} rad")

    ROOT.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ROOT / "dynamics.pt")
    np.savez(ROOT / "norm.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    save_norm_json(x_mean, x_std, y_mean, y_std)
    export_onnx(model)
    print(f"wrote dynamics.pt, norm.npz, norm.json, dynamics.onnx to {ROOT}")
    return model


def export_onnx(model):
    dummy = torch.zeros(1, STATE + ACTION)
    kwargs = dict(input_names=["input"], output_names=["output"], opset_version=15)
    try:
        torch.onnx.export(model, dummy, ROOT / "dynamics.onnx", dynamo=False, **kwargs)
    except TypeError:
        # Older torch without the dynamo argument
        torch.onnx.export(model, dummy, ROOT / "dynamics.onnx", **kwargs)


if __name__ == "__main__":
    current_time = time.time()
    train_world_model(load_rollout())
    end_time = time.time()
    print(f"Training time: {end_time - current_time} seconds")
