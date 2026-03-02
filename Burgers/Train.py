#!/usr/bin/env python
# coding: utf-8

"""
This code is modified from the code found in https://arxiv.org/abs/2011.07727. This generates an masked shallow auto-encoder from the training snapshot, "./data/snapshot_git.p" with parameters values as printed. 
The auto-encoder is save at './model/AE_git.tar'. This is used in the LaSDI_1DBurgers_NM.ipynb notebook.

Last Modified: Bill Fries 2/2/22

"""


import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data.dataloader import DataLoader

sys.path.append("..")
import modAutoEncoder as autoencoder
import modLaSDIUtils as utils


# In[2]:


#get_ipython().system('nvidia-smi')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AE from snapshot_git.p with optional parameter-grid subsampling."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="",
        help="Directory with index.csv and per-parameter .npz files (preferred over --snapshot-path).",
    )
    parser.add_argument(
        "--snapshot-path",
        type=str,
        default="./data/snapshot_git.p",
        help="Path to snapshot_git.p",
    )
    parser.add_argument(
        "--samples-per-axis",
        type=int,
        default=5,
        help="Number of parameter samples along each axis (<=0 uses all cases).",
    )
    parser.add_argument(
        "--num-a",
        type=int,
        default=0,
        help="Total grid size along a in snapshot ordering (0 = infer).",
    )
    parser.add_argument(
        "--num-w",
        type=int,
        default=0,
        help="Total grid size along w in snapshot ordering (0 = infer).",
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
        help="Random seed for train/test split.",
    )
    return parser.parse_args()


def infer_grid_shape(n_cases, num_a, num_w):
    if num_a <= 0 and num_w <= 0:
        root = int(round(np.sqrt(n_cases)))
        if root * root != n_cases:
            raise ValueError(
                "Cannot infer square grid from n_cases. Provide --num-a and --num-w explicitly."
            )
        return root, root

    if num_a <= 0:
        if n_cases % num_w != 0:
            raise ValueError("n_cases is not divisible by --num-w")
        return n_cases // num_w, num_w

    if num_w <= 0:
        if n_cases % num_a != 0:
            raise ValueError("n_cases is not divisible by --num-a")
        return num_a, n_cases // num_a

    if num_a * num_w != n_cases:
        raise ValueError("num_a * num_w must match number of parameter cases")
    return num_a, num_w


def select_case_indices(num_a, num_w, samples_per_axis):
    if samples_per_axis <= 0:
        return np.arange(num_a * num_w, dtype=int), np.arange(num_a), np.arange(num_w)

    if samples_per_axis > num_a or samples_per_axis > num_w:
        raise ValueError("samples_per_axis cannot exceed num_a or num_w")

    a_idx = np.round(np.linspace(0, num_a - 1, samples_per_axis)).astype(int)
    w_idx = np.round(np.linspace(0, num_w - 1, samples_per_axis)).astype(int)
    a_idx = np.asarray(list(dict.fromkeys(a_idx.tolist())), dtype=int)
    w_idx = np.asarray(list(dict.fromkeys(w_idx.tolist())), dtype=int)
    case_idx = np.asarray([ia * num_w + iw for ia in a_idx for iw in w_idx], dtype=int)
    return case_idx, a_idx, w_idx


def load_index_rows(dataset_dir):
    index_path = dataset_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"index.csv not found in {dataset_dir}")

    rows = []
    with index_path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if "filename" not in header:
            raise ValueError("index.csv header is invalid")
        for line in handle:
            filename, a_str, w_str = line.strip().split(",")
            rows.append((filename, float(a_str), float(w_str)))
    return rows


def select_grid_rows(rows, samples_per_axis):
    if not rows:
        return []

    if samples_per_axis <= 0:
        return list(rows)

    a_values = np.array(sorted({a for _, a, _ in rows}), dtype=float)
    w_values = np.array(sorted({w for _, _, w in rows}), dtype=float)

    if samples_per_axis > len(a_values) or samples_per_axis > len(w_values):
        raise ValueError("samples_per_axis cannot exceed dataset grid size")

    a_idx = np.round(np.linspace(0, len(a_values) - 1, samples_per_axis)).astype(int)
    w_idx = np.round(np.linspace(0, len(w_values) - 1, samples_per_axis)).astype(int)
    a_idx = np.asarray(list(dict.fromkeys(a_idx.tolist())), dtype=int)
    w_idx = np.asarray(list(dict.fromkeys(w_idx.tolist())), dtype=int)

    target_a = a_values[a_idx]
    target_w = w_values[w_idx]
    targets = np.array([(a, w) for a in target_a for w in target_w], dtype=float)

    params = np.asarray([[a, w] for _, a, w in rows], dtype=float)
    selected_idx = []
    selected_set = set()

    for target in targets:
        dists = np.linalg.norm(params - target, axis=1)
        for idx in np.argsort(dists):
            idx_int = int(idx)
            if idx_int not in selected_set:
                selected_idx.append(idx_int)
                selected_set.add(idx_int)
                break

    return [rows[i] for i in selected_idx]


def load_snapshots_from_dataset(dataset_dir, samples_per_axis):
    dataset_dir = Path(dataset_dir)
    rows = load_index_rows(dataset_dir)
    selected_rows = select_grid_rows(rows, samples_per_axis)
    if not selected_rows:
        raise RuntimeError("No rows selected from dataset")

    snapshots = []
    nx_ref = None
    for filename, _, _ in selected_rows:
        path = dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        with np.load(path) as data:
            if "u" not in data:
                raise ValueError(f"'u' array missing in {path}")
            u = np.asarray(data["u"], dtype=float)

        if u.ndim != 2:
            raise ValueError(f"Expected 2D 'u' in {path}, got shape {u.shape}")
        if nx_ref is None:
            nx_ref = u.shape[1]
        elif u.shape[1] != nx_ref:
            raise ValueError("All selected .npz files must have the same spatial size")

        snapshots.append(u)

    snapshot_full = np.vstack(snapshots)
    return snapshot_full, len(rows), len(selected_rows)


def load_snapshots_from_pickle(snapshot_path, samples_per_axis, num_a, num_w):
    nx = 1001
    nt = 1000
    rows_per_case = nt + 1

    with open(snapshot_path, "rb") as handle:
        solution_snapshot_orig = np.asarray(pickle.load(handle))

    if solution_snapshot_orig.ndim != 2 or solution_snapshot_orig.shape[1] != nx:
        raise ValueError(f"Expected snapshot shape (N, {nx}), got {solution_snapshot_orig.shape}")

    ndata = solution_snapshot_orig.shape[0]
    if ndata % rows_per_case != 0:
        raise ValueError("Number of rows in snapshot is not divisible by nt+1")
    n_cases = ndata // rows_per_case

    grid_num_a, grid_num_w = infer_grid_shape(n_cases, num_a, num_w)
    case_idx, a_idx, w_idx = select_case_indices(grid_num_a, grid_num_w, samples_per_axis)

    # Snapshot ordering is assumed as: for a in a_values: for w in w_values.
    by_case = solution_snapshot_orig.reshape(n_cases, rows_per_case, nx)
    snapshot_full = by_case[case_idx].reshape(-1, nx)
    print(f"Loaded {n_cases} parameter cases from {snapshot_path}")
    print(f"Using {case_idx.size} cases (a-index {a_idx.tolist()}, w-index {w_idx.tolist()})")
    return snapshot_full


def main():
    args = parse_args()

    # Choose device that is not being used
    gpu_ids = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
        solution_snapshot, total_cases, selected_cases = load_snapshots_from_dataset(
            dataset_dir, args.samples_per_axis
        )
        print(f"Loaded {total_cases} parameter cases from {dataset_dir}")
        print(f"Using {selected_cases} cases from dataset index")
    else:
        solution_snapshot = load_snapshots_from_pickle(
            args.snapshot_path,
            args.samples_per_axis,
            args.num_a,
            args.num_w,
        )

    # remove periodic duplicate endpoint if included
    if solution_snapshot.shape[1] == 1001:
        solution_snapshot = solution_snapshot[:, :-1]
    elif solution_snapshot.shape[1] != 1000:
        raise ValueError(
            f"Expected spatial dimension 1001(with endpoint) or 1000(no endpoint), got {solution_snapshot.shape[1]}"
        )
    solution_snapshot = solution_snapshot.astype("float32")

    if not (0.0 < args.test_fraction < 1.0):
        raise ValueError("--test-fraction must be between 0 and 1")

    rng = np.random.default_rng(args.seed)
    all_idx = np.arange(solution_snapshot.shape[0])
    perm = rng.permutation(all_idx)
    n_test = int(args.test_fraction * solution_snapshot.shape[0])
    n_test = max(1, min(n_test, solution_snapshot.shape[0] - 1))
    test_ind = perm[:n_test]
    train_ind = perm[n_test:]

    trainset = solution_snapshot[train_ind]
    testset = solution_snapshot[test_ind]

    dataset = {
        "train": data_utils.TensorDataset(torch.tensor(trainset)),
        "test": data_utils.TensorDataset(torch.tensor(testset)),
    }
    print(dataset["train"].tensors[0].shape, dataset["test"].tensors[0].shape)
    print(trainset.shape, testset.shape)

    # set device
    device = autoencoder.getDevice()
    print("Using device:", device, "\n")

    # set encoder and decoder types, activation function, etc.
    encoder_class = autoencoder.Encoder
    decoder_class = autoencoder.Decoder
    f_activation = autoencoder.SiLU

    # set the number of nodes in each layer
    m = 1000
    f = 4
    b = 36
    db = 12
    M2 = b + (m - 1) * db
    M1 = 2 * m
    mask = utils.create_mask_1d(m, b, db)

    # set batch_size, number of epochs, patience for early stop
    batch_size = 20
    num_epochs = 1000
    num_epochs_print = num_epochs // 100
    early_stop_patience = num_epochs // 10

    # autoencoder filename
    AE_fname = "models/AE_git.tar"
    chkpt_fname = "checkpoint.tar"
    Path(AE_fname).parent.mkdir(parents=True, exist_ok=True)

    encoder, decoder = autoencoder.createAE(
        encoder_class,
        decoder_class,
        f_activation,
        mask,
        m, f, M1, M2,
        device,
    )

    # set data loaders
    train_loader = DataLoader(
        dataset=dataset["train"], batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        dataset=dataset["test"], batch_size=batch_size, shuffle=True, num_workers=0
    )
    data_loaders = {"train": train_loader, "test": test_loader}
    _ = data_loaders

    # train
    autoencoder.trainAE(
        encoder,
        decoder,
        trainset,
        testset,
        batch_size,
        num_epochs,
        num_epochs_print,
        early_stop_patience,
        AE_fname,
        chkpt_fname,
    )


if __name__ == "__main__":
    main()
