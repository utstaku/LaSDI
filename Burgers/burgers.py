#!/usr/bin/env python3
"""Plot Burgers prediction and accurate solution from .npz files.

Supports:
1) One file containing both arrays (for example keys `u` and `u_true`)
2) Two files: one prediction file and one accurate/exact file
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PRED_KEYS = ("u_pred", "pred", "prediction", "u")
TRUE_KEYS = ("u_true", "u_exact", "u_acc", "accurate", "exact", "truth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prediction vs accurate Burgers solution from .npz files.")
    parser.add_argument(
        "--pred",
        type=str,
        default="",
        help="Prediction .npz file (required unless --file contains both prediction and accurate arrays)",
    )
    parser.add_argument(
        "--true",
        type=str,
        default="",
        help="Accurate/exact .npz file (required unless --file contains both prediction and accurate arrays)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default="",
        help="Single .npz file containing both prediction and accurate arrays",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Path to save plot image (otherwise show interactively)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/burgers",
        help="Dataset directory containing accurate solutions named a_<a>_w_<w>.npz",
    )
    return parser.parse_args()


def _find_key(data: np.lib.npyio.NpzFile, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in data:
            return key
    return None


def _maybe_get_grid(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    return data[key] if key in data else None


def _load_single(path: Path) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    data = np.load(path)
    pred_key = _find_key(data, PRED_KEYS)
    true_key = _find_key(data, TRUE_KEYS)
    x = _maybe_get_grid(data, "x")
    t = _maybe_get_grid(data, "t")
    u_pred = data[pred_key] if pred_key else None
    u_true = data[true_key] if true_key else None
    return x, t, u_pred, u_true


def _load_array(path: Path, preferred_keys: tuple[str, ...]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    data = np.load(path)
    key = _find_key(data, preferred_keys)
    if key is None:
        available = ", ".join(data.files)
        raise KeyError(f"No matching array keys {preferred_keys} in {path}. Available keys: {available}")
    x = _maybe_get_grid(data, "x")
    t = _maybe_get_grid(data, "t")
    return x, t, data[key]


def _dataset_file_from_params(data_dir: Path, a: float, w: float) -> Path:
    return data_dir / f"a_{a:.6f}_w_{w:.6f}.npz"


def _load_truth_from_dataset(pred_path: Path, data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_data = np.load(pred_path)
    if "a" not in pred_data or "w" not in pred_data:
        raise ValueError(
            f"{pred_path} does not contain `a` and `w`, so accurate solution cannot be looked up in {data_dir}"
        )
    a = float(pred_data["a"])
    w = float(pred_data["w"])
    truth_path = _dataset_file_from_params(data_dir, a, w)
    if not truth_path.exists():
        raise FileNotFoundError(
            f"Accurate solution file not found: {truth_path} (from a={a:.6f}, w={w:.6f})"
        )
    x_true, t_true, u_true = _load_array(truth_path, ("u",))
    if x_true is None or t_true is None:
        raise ValueError(f"{truth_path} must contain `x`, `t`, and `u`")
    return x_true, t_true, u_true


def load_plot_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if args.file:
        x, t, u_pred, u_true = _load_single(Path(args.file))
        if u_pred is None:
            raise ValueError(f"{args.file} must contain a prediction array (keys: {PRED_KEYS})")
        if u_true is None:
            x_true, t_true, u_true = _load_truth_from_dataset(Path(args.file), Path(args.data_dir))
            if x is None:
                x = x_true
            if t is None:
                t = t_true
        if x is None or t is None:
            raise ValueError(f"{args.file} must contain `x` and `t` arrays")
        if u_pred.shape != u_true.shape:
            raise ValueError(f"Shape mismatch: prediction {u_pred.shape} vs accurate {u_true.shape}")
        if u_pred.shape != (t.size, x.size):
            raise ValueError(
                f"Field shape {u_pred.shape} does not match grid sizes (t={t.size}, x={x.size})"
            )
        return x, t, u_pred, u_true

    if not args.pred:
        raise ValueError("Provide --pred <pred.npz>, optionally with --true <true.npz>, or use --file <combined.npz>")

    x_pred, t_pred, u_pred = _load_array(Path(args.pred), PRED_KEYS)
    if args.true:
        x_true, t_true, u_true = _load_array(Path(args.true), TRUE_KEYS + ("u",))
    else:
        x_true, t_true, u_true = _load_truth_from_dataset(Path(args.pred), Path(args.data_dir))

    x = x_pred if x_pred is not None else x_true
    t = t_pred if t_pred is not None else t_true
    if x is None or t is None:
        raise ValueError("At least one input file must contain both `x` and `t` arrays")

    if x_pred is not None and not np.allclose(x_pred, x):
        raise ValueError("Prediction x-grid does not match selected plotting grid")
    if x_true is not None and not np.allclose(x_true, x):
        raise ValueError("Accurate x-grid does not match selected plotting grid")
    if t_pred is not None and not np.allclose(t_pred, t):
        raise ValueError("Prediction t-grid does not match selected plotting grid")
    if t_true is not None and not np.allclose(t_true, t):
        raise ValueError("Accurate t-grid does not match selected plotting grid")

    if u_pred.shape != u_true.shape:
        raise ValueError(f"Shape mismatch: prediction {u_pred.shape} vs accurate {u_true.shape}")
    if u_pred.shape != (t.size, x.size):
        raise ValueError(
            f"Field shape {u_pred.shape} does not match grid sizes (t={t.size}, x={x.size})"
        )

    return x, t, u_pred, u_true


def plot_fields(x: np.ndarray, t: np.ndarray, u_pred: np.ndarray, u_true: np.ndarray, save_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting") from exc

    err = u_pred - u_true
    vmin = float(min(u_pred.min(), u_true.min()))
    vmax = float(max(u_pred.max(), u_true.max()))
    emax = float(np.max(np.abs(err)))
    extent = [float(x[0]), float(x[-1]), float(t[0]), float(t[-1])]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    im0 = axes[0].imshow(u_pred, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("Prediction")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    im1 = axes[1].imshow(u_true, origin="lower", aspect="auto", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("Accurate Solution")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")

    im2 = axes[2].imshow(err, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", vmin=-emax, vmax=emax)
    axes[2].set_title(f"Error (max |e|={emax:.3e})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")

    fig.colorbar(im1, ax=axes[:2], shrink=0.9, label="u")
    fig.colorbar(im2, ax=axes[2], shrink=0.9, label="u_pred - u_true")

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    x, t, u_pred, u_true = load_plot_data(args)
    plot_fields(x, t, u_pred, u_true, args.save)


if __name__ == "__main__":
    main()
