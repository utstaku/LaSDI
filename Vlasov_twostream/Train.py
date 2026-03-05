#!/usr/bin/env python
# coding: utf-8

"""
Train one masked shallow autoencoder for Vlasov two-stream data.

Pipeline:
1) Load f(t, x, v) from distribution_full.npz in selected case folders.
2) Flatten each time snapshot to f_flat(t) in R^(Nx*Nv).
3) Stack all selected snapshots and train a single AE on these flattened vectors.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import modAutoEncoder as autoencoder
import modLaSDIUtils as utils


CASE_NAME_RE = re.compile(r"^T_([-+0-9.eE]+)_k_([-+0-9.eE]+)$")


def parse_args():
    parser = argparse.ArgumentParser(description="Train one AE on flattened Vlasov snapshots.")
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
        help="Subsample T-k grid per axis (<=0 uses all cases).",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=1,
        help="Use every Nth time snapshot from each case (default 1 keeps Nt+1).",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=0,
        help="Maximum total snapshots used for training (<=0 uses all selected snapshots).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of snapshots used for test split.",
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
        help="Latent dimension f.",
    )
    parser.add_argument(
        "--width-factor",
        type=float,
        default=2.0,
        help="Encoder hidden width factor: M1 = width_factor * m.",
    )
    parser.add_argument(
        "--mask-block",
        type=int,
        default=8,
        help="Mask block size b for create_mask_1d.",
    )
    parser.add_argument(
        "--mask-stride",
        type=int,
        default=1,
        help="Mask stride db for create_mask_1d.",
    )

    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--num-epochs", type=int, default=1000, help="Maximum training epochs.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=100,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model/checkpoint/loss curve.",
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
        dist_path = case_dir / "animation_data.npz"
        if not dist_path.exists():
            continue
        T, k = parsed
        cases.append({"T": T, "k": k, "path": dist_path})

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


def load_flattened_snapshots(cases, time_stride: int, max_snapshots: int, seed: int):
    if time_stride <= 0:
        raise ValueError("--time-stride must be >= 1")

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

    dominant_shape = max(shape_counts.items(), key=lambda kv: kv[1])[0]
    skipped = [p for p, shp in shape_by_path.items() if shp != dominant_shape]
    if skipped:
        print(
            f"Warning: found mixed spatial shapes {shape_counts}. "
            f"Using dominant shape {dominant_shape} and skipping {len(skipped)} case(s)."
        )
        for path in skipped:
            print(f"  skipped: {path}")

    chunks = []
    flat_dim = None
    nx_ref = None
    nv_ref = None
    total_pre_limit = 0

    for case in cases:
        if shape_by_path[str(case["path"])] != dominant_shape:
            continue

        with np.load(case["path"]) as data:
            f = np.asarray(data["f"], dtype=np.float32)

        f = f[::time_stride]
        flat = f.reshape(f.shape[0], -1)
        total_pre_limit += flat.shape[0]

        if nx_ref is None:
            nx_ref, nv_ref = f.shape[1], f.shape[2]
        if flat_dim is None:
            flat_dim = flat.shape[1]

        chunks.append(flat)

    if not chunks:
        raise RuntimeError("No snapshots loaded from selected cases")

    snapshots = np.vstack(chunks).astype(np.float32, copy=False)

    if max_snapshots > 0 and snapshots.shape[0] > max_snapshots:
        rng = np.random.default_rng(seed)
        keep = rng.choice(snapshots.shape[0], size=max_snapshots, replace=False)
        snapshots = snapshots[np.sort(keep)]

    return snapshots, total_pre_limit, nx_ref, nv_ref


def split_train_test(samples: np.ndarray, test_fraction: float, seed: int):
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("--test-fraction must be between 0 and 1")
    if samples.shape[0] < 4:
        raise ValueError("Need at least 4 snapshots to split train/test robustly")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(samples.shape[0])
    n_test = int(test_fraction * samples.shape[0])
    n_test = max(2, min(n_test, samples.shape[0] - 2))

    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    return samples[train_idx], samples[test_idx]


def avoid_singleton_last_batch(data: np.ndarray, batch_size: int, seed: int):
    if data.shape[0] == 1:
        return np.vstack([data, data.copy()])

    if batch_size > 1 and data.shape[0] % batch_size == 1:
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(0, data.shape[0]))
        data = np.vstack([data, data[idx : idx + 1]])
    return data


def get_activation(activation_name: str):
    if activation_name == "sigmoid":
        return nn.Sigmoid
    return autoencoder.SiLU


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
    if args.width_factor <= 0:
        raise ValueError("--width-factor must be positive")
    if args.mask_block <= 0 or args.mask_stride <= 0:
        raise ValueError("--mask-block and --mask-stride must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = autoencoder.getDevice()
    print("Using device:", device)

    all_cases = discover_cases(dataset_root)
    selected_cases = select_cases(all_cases, args.samples_per_axis)

    snapshots, total_pre_limit, nx, nv = load_flattened_snapshots(
        selected_cases,
        time_stride=args.time_stride,
        max_snapshots=args.max_snapshots,
        seed=args.seed,
    )

    trainset, testset = split_train_test(snapshots, args.test_fraction, args.seed)
    trainset = avoid_singleton_last_batch(trainset, args.batch_size, args.seed + 1)
    testset = avoid_singleton_last_batch(testset, args.batch_size, args.seed + 2)

    print(f"Dataset root: {dataset_root}")
    print(f"Cases found: {len(all_cases)} | Cases selected: {len(selected_cases)}")
    print(f"Grid shape per snapshot: Nx={nx}, Nv={nv}")
    print(f"Snapshots before max-snapshots cap: {total_pre_limit} | used: {snapshots.shape[0]}")
    print(f"Train shape: {trainset.shape} | Test shape: {testset.shape}")

    m = snapshots.shape[1]
    f = args.latent_dim
    M1 = max(f + 1, int(round(args.width_factor * m)))
    b = args.mask_block
    db = args.mask_stride
    M2 = b + (m - 1) * db

    print(f"AE dims: m={m}, f={f}, M1={M1}, M2={M2}, b={b}, db={db}")

    activation = get_activation(args.activation)
    mask = utils.create_mask_1d(m, b, db)
    encoder, decoder = autoencoder.createAE(
        autoencoder.Encoder,
        autoencoder.Decoder,
        activation,
        mask,
        m,
        f,
        M1,
        M2,
        device,
    )

    num_epochs_print = max(1, args.num_epochs // 100)
    model_path = output_dir / "AE_vlasov.tar"
    checkpoint_path = output_dir / "checkpoint_vlasov.tar"
    loss_plot_path = output_dir / "training_loss_vlasov.png"

    autoencoder.trainAE(
        encoder,
        decoder,
        trainset,
        testset,
        args.batch_size,
        args.num_epochs,
        num_epochs_print,
        args.early_stop_patience,
        str(model_path),
        str(checkpoint_path),
        plt_fname=str(loss_plot_path),
    )


if __name__ == "__main__":
    main()
