#!/usr/bin/env python
# coding: utf-8

"""
Train ConvAE on selected (T, k) train cases.

Pipeline:
1) Select train cases by --samples-per-axis on the full parameter grid.
2) Split those selected cases into AE-train / AE-val at case level.
3) Use all timesteps from each case and train AE.
4) Save trained AE model/checkpoints/loss plot.
"""

import argparse
import copy
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


CASE_NAME_RE = re.compile(r"^T_([-+0-9.eE]+)_k_([-+0-9.eE]+)$")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvAE for Vlasov snapshots.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="vlasov_twostream_param_grid",
        help="Root directory containing T_<T>_k_<k> case folders.",
    )
    parser.add_argument(
        "--samples-per-axis",
        type=int,
        default=5,
        help="Train-case subsample count per axis (5 -> 25 train cases).",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=1,
        help="Use every Nth saved snapshot from each case.",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="Maximum train snapshots used for AE (<=0 uses all).",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation fraction at case level (e.g. 0.2 -> 20/5 split for 25 cases).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["swish", "sigmoid"],
        default="swish",
        help="Activation function for hidden layers.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=8,
        help="Latent dimension Nz.",
    )
    parser.add_argument(
        "--width-factor",
        type=float,
        default=2.0,
        help="Unused in ConvAE (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--mask-block",
        type=int,
        default=8,
        help="Unused in ConvAE (kept for CLI compatibility).",
    )
    parser.add_argument(
        "--mask-stride",
        type=int,
        default=1,
        help="Unused in ConvAE (kept for CLI compatibility).",
    )
    parser.add_argument("--batch-size", type=int, default=1024, help="AE training batch size.")
    parser.add_argument("--num-epochs", type=int, default=1000, help="AE maximum epochs.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=100,
        help="AE early stopping patience on validation loss.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save AE artifacts.",
    )
    return parser.parse_args()


def _parse_case_dir(case_dir: Path):
    match = CASE_NAME_RE.match(case_dir.name)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def discover_cases(dataset_root: Path):
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    cases = []
    for case_dir in sorted(dataset_root.iterdir()):
        if not case_dir.is_dir():
            continue
        parsed = _parse_case_dir(case_dir)
        if parsed is None:
            continue
        data_path = case_dir / "distribution_full.npz"
        if not data_path.exists():
            continue
        T, k = parsed
        cases.append({"T": T, "k": k, "path": data_path})

    if not cases:
        raise RuntimeError(f"No valid case folders with distribution_full.npz under {dataset_root}")
    return cases


def select_cases(cases, samples_per_axis: int):
    if samples_per_axis <= 0:
        return list(cases)

    T_values = np.array(sorted({c["T"] for c in cases}), dtype=float)
    k_values = np.array(sorted({c["k"] for c in cases}), dtype=float)
    if samples_per_axis > len(T_values) or samples_per_axis > len(k_values):
        raise ValueError("samples_per_axis exceeds available T-k grid size")

    T_idx = np.round(np.linspace(0, len(T_values) - 1, samples_per_axis)).astype(int)
    k_idx = np.round(np.linspace(0, len(k_values) - 1, samples_per_axis)).astype(int)
    T_idx = np.asarray(list(dict.fromkeys(T_idx.tolist())), dtype=int)
    k_idx = np.asarray(list(dict.fromkeys(k_idx.tolist())), dtype=int)

    target_pairs = {(float(T_values[i]), float(k_values[j])) for i in T_idx for j in k_idx}
    selected = [c for c in cases if (c["T"], c["k"]) in target_pairs]
    selected.sort(key=lambda c: (c["T"], c["k"]))

    if not selected:
        raise RuntimeError("No cases selected; check --samples-per-axis")
    return selected


def split_train_val_cases_balanced(cases, val_fraction: float, seed: int):
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("--val-fraction must be between 0 and 1")
    if len(cases) < 2:
        raise ValueError("Need at least 2 cases to split train/val at case level")

    n_cases = len(cases)
    n_val = int(round(val_fraction * n_cases))
    n_val = max(1, min(n_val, n_cases - 1))

    params = np.array([[c["T"], c["k"]] for c in cases], dtype=np.float64)
    mins = params.min(axis=0)
    spans = params.max(axis=0) - mins
    spans[spans == 0.0] = 1.0
    p_norm = (params - mins) / spans

    # Greedy farthest-point sampling: select validation cases spread in (T, k) space.
    centroid = p_norm.mean(axis=0)
    start_idx = int(np.argmin(np.sum((p_norm - centroid[None, :]) ** 2, axis=1)))
    selected = [start_idx]
    selected_set = {start_idx}
    rng = np.random.default_rng(seed)

    while len(selected) < n_val:
        best_idx = None
        best_score = -1.0
        for idx in range(n_cases):
            if idx in selected_set:
                continue
            dists = np.sqrt(np.sum((p_norm[idx] - p_norm[selected]) ** 2, axis=1))
            score = float(np.min(dists))
            if score > best_score + 1.0e-12:
                best_score = score
                best_idx = idx
            elif abs(score - best_score) <= 1.0e-12:
                # Tie-break with seed-driven randomness for reproducibility.
                if rng.random() < 0.5:
                    best_idx = idx
        selected.append(best_idx)
        selected_set.add(best_idx)

    val_idx = sorted(selected)
    train_idx = [i for i in range(n_cases) if i not in selected_set]
    train_cases = [cases[i] for i in train_idx]
    val_cases = [cases[i] for i in val_idx]
    return train_cases, val_cases


def load_case_trajectories(cases, time_stride: int, expected_shape=None):
    if time_stride <= 0:
        raise ValueError("--time-stride must be >= 1")
    if not cases:
        if expected_shape is None:
            raise ValueError("No cases were provided and expected_shape is unknown.")
        return [], expected_shape

    shape_counts = {}
    shape_by_path = {}
    for case in cases:
        with np.load(case["path"]) as data:
            if "f" not in data:
                raise ValueError(f"'f' missing in {case['path']}")
            f_shape = tuple(np.asarray(data["f"]).shape)
        if len(f_shape) != 3:
            raise ValueError(f"Expected f shape (Nt,Nx,Nv), got {f_shape} in {case['path']}")
        spatial_shape = (f_shape[1], f_shape[2])
        shape_by_path[str(case["path"])] = spatial_shape
        shape_counts[spatial_shape] = shape_counts.get(spatial_shape, 0) + 1

    if expected_shape is None:
        dominant_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]
    else:
        dominant_shape = expected_shape

    if dominant_shape != (128, 128):
        raise ValueError(f"ConvAE expects (128,128), got dominant shape {dominant_shape}")

    skipped = [p for p, shp in shape_by_path.items() if shp != dominant_shape]
    if skipped:
        print(
            f"Warning: found mixed spatial shapes {shape_counts}. "
            f"Using {dominant_shape} and skipping {len(skipped)} case(s)."
        )
        for path in skipped:
            print(f"  skipped: {path}")

    loaded = []
    for case in cases:
        if shape_by_path[str(case["path"])] != dominant_shape:
            continue

        with np.load(case["path"]) as data:
            f = np.asarray(data["f"], dtype=np.float32)
            if "t" in data:
                t = np.asarray(data["t"], dtype=np.float64)
            else:
                t = np.arange(f.shape[0], dtype=np.float64)

        f = f[::time_stride]
        t = t[::time_stride]
        loaded.append(
            {
                "T": case["T"],
                "k": case["k"],
                "path": str(case["path"]),
                "f": f,  # (Nt, 128, 128)
                "t": t,  # (Nt,)
            }
        )

    return loaded, dominant_shape


def collect_snapshots(trajectories, max_snapshots: int, seed: int):
    if not trajectories:
        raise RuntimeError("No trajectories available")

    chunks = [tr["f"][:, None, :, :] for tr in trajectories]
    snapshots = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    total_before_cap = snapshots.shape[0]

    if max_snapshots > 0 and snapshots.shape[0] > max_snapshots:
        rng = np.random.default_rng(seed)
        keep = rng.choice(snapshots.shape[0], size=max_snapshots, replace=False)
        snapshots = snapshots[np.sort(keep)]
    return snapshots, total_before_cap


def avoid_singleton_last_batch(data: np.ndarray, batch_size: int, seed: int):
    if data.shape[0] == 1:
        return np.concatenate([data, data.copy()], axis=0)

    if batch_size > 1 and data.shape[0] % batch_size == 1:
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, data.shape[0]))
        data = np.concatenate([data, data[idx : idx + 1]], axis=0)
    return data


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def get_activation(activation_name: str):
    if activation_name == "sigmoid":
        return nn.Sigmoid
    return SiLU


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int, activation_cls):
        super().__init__()
        act = activation_cls

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),   # -> 16x64x64
            act(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 32x32x32
            act(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 64x16x16
            act(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> 128x8x8
            act(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            act(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, activation_cls):
        super().__init__()
        act = activation_cls

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            act(),
            nn.Linear(256, 128 * 8 * 8),
            act(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> 64x16x16
            act(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 32x32x32
            act(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 16x64x64
            act(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # -> 1x128x128
        )

    def forward(self, z):
        y = self.fc(z)
        y = y.view(-1, 128, 8, 8)
        return self.deconv(y)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_losses(loss_hist, out_path):
    plt.figure()
    plt.semilogy(loss_hist["train"])
    plt.semilogy(loss_hist["val"])
    plt.legend(["train", "val"])
    plt.savefig(out_path)
    plt.close()


def _compute_dataset_loss(encoder, decoder, loss_func, data_np, batch_size, device):
    encoder.eval()
    decoder.eval()
    total = 0.0
    n = data_np.shape[0]
    with torch.set_grad_enabled(False):
        for i in range(0, n, batch_size):
            x = torch.tensor(data_np[i : i + batch_size], device=device)
            y = decoder(encoder(x))
            total += loss_func(y, x).item() * x.shape[0]
    return total / max(1, n)


def train_autoencoder(
    encoder,
    decoder,
    train_data,
    val_data,
    batch_size,
    num_epochs,
    num_epochs_print,
    early_stop_patience,
    model_fname,
    chkpt_fname,
    plt_fname,
):
    model_dir = os.path.dirname(model_fname)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    chkpt_dir = os.path.dirname(chkpt_fname)
    if chkpt_dir:
        os.makedirs(chkpt_dir, exist_ok=True)

    plt_dir = os.path.dirname(plt_fname)
    if plt_dir:
        os.makedirs(plt_dir, exist_ok=True)

    dataset = {
        "train": TensorDataset(torch.tensor(train_data)),
        "val": TensorDataset(torch.tensor(val_data)),
    }
    dataset_sizes = {"train": train_data.shape[0], "val": val_data.shape[0]}
    data_loaders = {
        "train": DataLoader(dataset["train"], batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(dataset["val"], batch_size=batch_size, shuffle=False, num_workers=0),
    }

    device = get_device()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1.0e-3)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    loss_func = nn.MSELoss(reduction="mean")

    last_epoch = 0
    loss_hist = {"train": [], "val": []}
    best_loss = float("inf")
    early_stop_counter = 1
    best_encoder_wts = copy.deepcopy(encoder.state_dict())
    best_decoder_wts = copy.deepcopy(decoder.state_dict())

    if os.path.exists(chkpt_fname):
        checkpoint = torch.load(chkpt_fname, map_location=device)
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        last_epoch = int(checkpoint["epoch"])
        loss_hist = checkpoint["loss_hist"]
        best_loss = float(checkpoint["best_loss"])
        early_stop_counter = int(checkpoint["early_stop_counter"])
        best_encoder_wts = checkpoint["best_encoder_wts"]
        best_decoder_wts = checkpoint["best_decoder_wts"]
        print("\n--------checkpoint restored--------\n")
    else:
        print("\n--------checkpoint not restored--------\n")

    print(f"Start/Resume AE training from epoch {last_epoch + 1} to {num_epochs}")
    since = time.time()
    epoch = last_epoch

    if last_epoch >= num_epochs:
        print(
            f"Checkpoint epoch ({last_epoch}) already reached target num_epochs ({num_epochs}). "
            "Skipping additional training."
        )

    epoch_iter = range(last_epoch + 1, num_epochs + 1)
    if tqdm is not None:
        epoch_iter = tqdm(
            epoch_iter,
            total=max(0, num_epochs - last_epoch),
            desc="AE Training",
            leave=True,
        )

    for epoch in epoch_iter:
        if epoch % num_epochs_print == 0:
            print(f"\nEpoch {epoch}/{num_epochs}, Learning rate {optimizer.param_groups[0]['lr']}")
            print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                encoder.train()
                decoder.train()
            else:
                encoder.eval()
                decoder.eval()

            running_loss = 0.0
            for (inputs,) in data_loaders[phase]:
                inputs = inputs.to(device)
                targets = inputs

                if phase == "train":
                    optimizer.zero_grad()
                    outputs = decoder(encoder(inputs))
                    loss = loss_func(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.shape[0]
                else:
                    with torch.set_grad_enabled(False):
                        outputs = decoder(encoder(inputs))
                        running_loss += loss_func(outputs, targets).item() * inputs.shape[0]

            epoch_loss = running_loss / dataset_sizes[phase]
            loss_hist[phase].append(epoch_loss)

            if phase == "train":
                scheduler.step(epoch_loss)

            if epoch % num_epochs_print == 0:
                print(f"{phase} MSELoss: {epoch_loss}")

        if loss_hist["val"][-1] < best_loss:
            best_loss = loss_hist["val"][-1]
            early_stop_counter = 1
            best_encoder_wts = copy.deepcopy(encoder.state_dict())
            best_decoder_wts = copy.deepcopy(decoder.state_dict())
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                break

        if tqdm is not None:
            epoch_iter.set_postfix(
                train=f"{loss_hist['train'][-1]:.3e}",
                val=f"{loss_hist['val'][-1]:.3e}",
                best=f"{best_loss:.3e}",
            )

        if epoch % num_epochs_print == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state_dict": encoder.state_dict(),
                    "decoder_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_hist": loss_hist,
                    "best_loss": best_loss,
                    "early_stop_counter": early_stop_counter,
                    "best_encoder_wts": best_encoder_wts,
                    "best_decoder_wts": best_decoder_wts,
                },
                chkpt_fname,
            )
            torch.save(
                {"encoder_state_dict": encoder.state_dict(), "decoder_state_dict": decoder.state_dict()},
                model_fname,
            )
            plot_losses(loss_hist, plt_fname)

    encoder.load_state_dict(best_encoder_wts)
    decoder.load_state_dict(best_decoder_wts)

    time_elapsed = time.time() - since
    train_loss = _compute_dataset_loss(encoder, decoder, loss_func, train_data, batch_size, device)
    val_loss = _compute_dataset_loss(encoder, decoder, loss_func, val_data, batch_size, device)

    print()
    if epoch < num_epochs:
        print(
            "Early stopping: {} epochs in {:.0f}h {:.0f}m {:.0f}s".format(
                epoch - last_epoch,
                time_elapsed // 3600,
                (time_elapsed % 3600) // 60,
                (time_elapsed % 3600) % 60,
            )
        )
    else:
        print(
            "No early stopping: {} epochs in {:.0f}h {:.0f}m {:.0f}s".format(
                epoch - last_epoch,
                time_elapsed // 3600,
                (time_elapsed % 3600) // 60,
                (time_elapsed % 3600) % 60,
            )
        )
    print("-" * 10)
    print(f"Best train MSELoss: {train_loss}")
    print(f"Best val MSELoss: {val_loss}")

    torch.save(
        {"encoder_state_dict": encoder.state_dict(), "decoder_state_dict": decoder.state_dict()},
        model_fname,
    )
    plot_losses(loss_hist, plt_fname)

    if os.path.exists(chkpt_fname):
        os.remove(chkpt_fname)
        print("checkpoint removed")

    return loss_hist, float(best_loss)


def main():
    args = parse_args()

    if args.batch_size <= 1:
        raise ValueError("--batch-size must be >= 2")
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be positive")
    if args.early_stop_patience <= 0:
        raise ValueError("--early-stop-patience must be positive")
    if args.latent_dim <= 0:
        raise ValueError("--latent-dim must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    all_cases = discover_cases(dataset_root)
    selected_cases = select_cases(all_cases, args.samples_per_axis)
    train_cases, val_cases = split_train_val_cases_balanced(
        selected_cases, val_fraction=args.val_fraction, seed=args.seed
    )
    train_trajs, shape = load_case_trajectories(train_cases, args.time_stride, expected_shape=(128, 128))
    val_trajs, _ = load_case_trajectories(val_cases, args.time_stride, expected_shape=shape)

    train_snapshots, total_train_pre_cap = collect_snapshots(
        train_trajs, max_snapshots=args.max_snapshots, seed=args.seed
    )
    val_snapshots, total_val = collect_snapshots(
        val_trajs, max_snapshots=0, seed=args.seed + 12345
    )
    train_data = train_snapshots
    val_data = val_snapshots
    train_data = avoid_singleton_last_batch(train_data, args.batch_size, args.seed + 1)
    val_data = avoid_singleton_last_batch(val_data, args.batch_size, args.seed + 2)

    print(f"Dataset root: {dataset_root}")
    print(f"Cases found total: {len(all_cases)}")
    print(f"Cases selected by samples-per-axis: {len(selected_cases)}")
    print(f"AE case split -> train: {len(train_cases)} | val: {len(val_cases)}")
    print(f"Input per snapshot: {(1, shape[0], shape[1])} (each timestep as one sample)")
    print(f"Train snapshots before max-snapshots cap: {total_train_pre_cap} | used: {train_snapshots.shape[0]}")
    print(f"Val snapshots used: {total_val}")
    print(f"AE train shape: {train_data.shape} | AE val shape: {val_data.shape}")

    activation = get_activation(args.activation)
    encoder = ConvEncoder(latent_dim=args.latent_dim, activation_cls=activation).to(device)
    decoder = ConvDecoder(latent_dim=args.latent_dim, activation_cls=activation).to(device)

    num_epochs_print = max(1, args.num_epochs // 100)
    model_path = output_dir / "AE_vlasov.tar"
    checkpoint_path = output_dir / "checkpoint_vlasov.tar"
    loss_plot_path = output_dir / "training_loss_vlasov.png"

    train_autoencoder(
        encoder=encoder,
        decoder=decoder,
        train_data=train_data,
        val_data=val_data,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        num_epochs_print=num_epochs_print,
        early_stop_patience=args.early_stop_patience,
        model_fname=str(model_path),
        chkpt_fname=str(checkpoint_path),
        plt_fname=str(loss_plot_path),
    )
    print(f"AE training complete. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
