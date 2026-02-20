#!/usr/bin/env python3
"""Run Burgers prediction using a trained LaSDI auto-encoder model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from burgers_simulation import initial_condition
from lasdi import (
    decode_series,
    encode_series,
    integrate_latent,
    interpolate_coeffs,
    load_model,
    require_torch,
)


def predict(
    model_dir: Path,
    a: float,
    w: float,
    knn: int,
    batch_size: int,
    use_cpu: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require_torch()
    import torch

    device = torch.device("cpu" if use_cpu or not torch.cuda.is_available() else "cuda")
    model, mean, std, params, coeffs, terms, x, t, cfg = load_model(model_dir, device)

    target = np.array([a, w], dtype=float)
    dt = t[1] - t[0]
    steps = t.size - 1

    coeffs_interp = interpolate_coeffs(params, coeffs, target, knn)

    u0 = initial_condition(x, a, w)
    if not cfg["drop_endpoint"]:
        u0[-1] = u0[0]

    z0 = encode_series(model, u0[None, :], mean, std, device, batch_size)[0]
    z = integrate_latent(z0, coeffs_interp, terms, dt, steps)
    u_pred = decode_series(model, z, mean, std, device, batch_size)

    if not cfg["drop_endpoint"]:
        u_pred[:, -1] = u_pred[:, 0]

    return x, t, u_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Burgers dynamics using a trained LaSDI model."
    )
    parser.add_argument("--model-dir", required=True, help="Trained model directory")
    parser.add_argument("--a", type=float, required=True, help="Amplitude parameter a")
    parser.add_argument("--w", type=float, required=True, help="Width parameter w")
    parser.add_argument("--knn", type=int, default=1, help="kNN neighbors for coeffs")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--output",
        type=str,
        default="prediction.npz",
        help="Output .npz file",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Path to save plot image",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x, t, u_pred = predict(
        Path(args.model_dir),
        args.a,
        args.w,
        args.knn,
        args.batch_size,
        args.cpu,
    )

    out = Path(args.output)
    np.savez(out, a=args.a, w=args.w, x=x, t=t, u=u_pred)
    print(f"Saved prediction to {out}")

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or use --no-plot"
        ) from exc

    plt.figure(figsize=(8, 4))
    plt.plot(x, u_pred[0], label="t=0")
    plt.plot(x, u_pred[-1], label=f"t={t[-1]:.3f}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("Burgers Prediction (LaSDI)")
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
