#!/usr/bin/env python3
"""1D inviscid Burgers equation simulation.

Governing equation (periodic on [-3, 3]):
    u_t + u u_x = 0
    u(t, x=3) = u(t, x=-3)

Initial condition (parameterized):
    u(0, x | a, w) = a * exp(-x^2 / (2 w^2))

Defaults:
    dx = 6e-3, dt = 1e-3, t_end = 1.0
    a in [0.7, 0.9], w in [0.9, 1.1]

Numerics:
    - Richtmyer (two-step Lax-Wendroff) scheme
    - Second-order accurate in both space and time
    - Stable for hyperbolic conservation laws
    - Conservative form: f(u) = 0.5 * u^2
    - Properly handles shock formation in inviscid Burgers

Dataset generation:
    --dataset-dir writes one .npz per (a, w) with full time evolution.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

#from . import utils  # noqa: F401 - for potential future utilities
@dataclass
class SimConfig:
    x_min: float = -3.0
    x_max: float = 3.0
    dx: float = 6.0e-3
    dt: float = 1.0e-3
    t_end: float = 1.0
    a: float = 0.8
    w: float = 1.0
    include_endpoint: bool = True
    snapshot_interval: float = 0.0
    snapshot_dir: str = ""


def make_grid(cfg: SimConfig) -> np.ndarray:
    length = cfg.x_max - cfg.x_min
    nx_float = length / cfg.dx
    nx = int(round(nx_float))
    if not np.isclose(nx_float, nx, rtol=0.0, atol=1.0e-12):
        raise ValueError("Domain length must be an integer multiple of dx")
    return np.linspace(cfg.x_min, cfg.x_max, nx, endpoint=False)


def append_endpoint(x: np.ndarray, u: np.ndarray, x_max: float) -> Tuple[np.ndarray, np.ndarray]:
    x_out = np.concatenate([x, [x_max]])
    u_out = np.concatenate([u, [u[0]]])
    return x_out, u_out


def append_endpoint_series(u: np.ndarray) -> np.ndarray:
    return np.concatenate([u, u[:, :1]], axis=1)


def initial_condition(x: np.ndarray, a: float, w: float) -> np.ndarray:
    return a * np.exp(-(x * x) / (2.0 * w * w))


def flux(u: np.ndarray) -> np.ndarray:
    """Flux function f(u) = 0.5 * u^2 for Burgers equation."""
    return 0.5 * u * u


def richtmyer_step(
    u: np.ndarray, dx: float, dt: float
) -> Tuple[np.ndarray, int, bool]:
    """Advance one step using Richtmyer (two-step Lax-Wendroff) scheme.

    This is a stable, second-order accurate scheme for hyperbolic conservation laws.
    It handles shocks in the inviscid Burgers equation properly.

    Algorithm:
    1. Predictor (half step): u_{i+1/2}^* = (u_i + u_{i+1})/2 - dt/(2dx)*(f(u_{i+1}) - f(u_i))
    2. Corrector (full step): u_i^{n+1} = u_i^n - dt/dx*(f(u_{i+1/2}^*) - f(u_{i-1/2}^*))
    """
    # Predictor step: compute values at half-grid points (i+1/2)
    f = flux(u)
    f_plus = np.roll(f, -1)  # f at i+1
    u_plus = np.roll(u, -1)  # u at i+1

    # Half-step values at i+1/2
    u_half = 0.5 * (u + u_plus) - (dt / (2.0 * dx)) * (f_plus - f)

    # Handle periodic boundary for half-step values
    # u_half at i-1/2 is obtained by rolling u_half by +1
    u_half_minus = np.roll(u_half, 1)  # u at i-1/2

    # Corrector step: compute flux at half-points
    f_half = flux(u_half)
    f_half_minus = flux(u_half_minus)

    # Full step update
    u_next = u - (dt / dx) * (f_half - f_half_minus)

    return u_next, 1, True


def time_grid(cfg: SimConfig) -> Tuple[np.ndarray, int]:
    steps_float = cfg.t_end / cfg.dt
    steps = int(round(steps_float))
    if not np.isclose(steps_float, steps, rtol=0.0, atol=1.0e-12):
        raise ValueError("t_end must be an integer multiple of dt")
    t = np.linspace(0.0, cfg.t_end, steps + 1)
    return t, steps


def write_snapshot(out_dir: Path, index: int, t: float, x: np.ndarray, u: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"snapshot_{index:05d}.npz"
    np.savez(path, t=t, x=x, u=u)


def simulate_full(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = make_grid(cfg)
    u0 = initial_condition(x, cfg.a, cfg.w)
    u = u0.copy()

    t, steps = time_grid(cfg)
    u_all = np.empty((steps + 1, x.size), dtype=u.dtype)
    u_all[0] = u0

    for n in range(steps):
        u, _, converged = richtmyer_step(u, cfg.dx, cfg.dt)
        if not converged:
            raise RuntimeError(f"Time step failed at step {n}")
        u_all[n + 1] = u

    if cfg.include_endpoint:
        x_out = np.concatenate([x, [cfg.x_max]])
        u_out = append_endpoint_series(u_all)
        return x_out, t, u_out

    return x, t, u_all


def simulate(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = make_grid(cfg)
    u0 = initial_condition(x, cfg.a, cfg.w)
    u = u0.copy()

    _, steps = time_grid(cfg)

    t = 0.0
    snapshot_every = None
    snapshot_index = 0
    snapshot_dir = None

    if cfg.snapshot_interval > 0.0 and cfg.snapshot_dir:
        snap_float = cfg.snapshot_interval / cfg.dt
        snapshot_every = int(round(snap_float))
        if not np.isclose(snap_float, snapshot_every, rtol=0.0, atol=1.0e-12):
            raise ValueError("snapshot_interval must be an integer multiple of dt")
        snapshot_dir = Path(cfg.snapshot_dir)
        x_out, u_out = (x, u)
        if cfg.include_endpoint:
            x_out, u_out = append_endpoint(x, u, cfg.x_max)
        write_snapshot(snapshot_dir, snapshot_index, t, x_out, u_out)
        snapshot_index += 1

    for n in range(steps):
        u, _, converged = richtmyer_step(u, cfg.dx, cfg.dt)
        if not converged:
            raise RuntimeError(f"Time step failed at step {n}")
        t += cfg.dt

        if snapshot_every is not None and (n + 1) % snapshot_every == 0:
            x_out, u_out = (x, u)
            if cfg.include_endpoint:
                x_out, u_out = append_endpoint(x, u, cfg.x_max)
            write_snapshot(snapshot_dir, snapshot_index, t, x_out, u_out)
            snapshot_index += 1

    if cfg.include_endpoint:
        x_out, u0_out = append_endpoint(x, u0, cfg.x_max)
        _, u_out = append_endpoint(x, u, cfg.x_max)
        return x_out, u0_out, u_out

    return x, u0, u


def make_param_values(v_min: float, v_max: float, v_step: float, name: str) -> np.ndarray:
    if v_step <= 0.0:
        raise ValueError(f"{name}_step must be positive")
    span = v_max - v_min
    if span < 0.0:
        raise ValueError(f"{name}_max must be >= {name}_min")
    count_float = span / v_step
    count = int(round(count_float))
    if not np.isclose(count_float, count, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name} range must be an integer multiple of {name}_step")
    values = v_min + v_step * np.arange(count + 1)
    return np.round(values, 12)


def generate_dataset(
    cfg: SimConfig,
    a_values: Iterable[float],
    w_values: Iterable[float],
    out_dir: Path,
) -> None:
    a_values = list(a_values)
    w_values = list(w_values)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    x_ref = None
    t_ref = None

    for a in a_values:
        for w in w_values:
            run_cfg = replace(cfg, a=float(a), w=float(w), snapshot_interval=0.0, snapshot_dir="")
            x, t, u_all = simulate_full(run_cfg)

            if x_ref is None:
                x_ref = x
                t_ref = t

            filename = f"a_{a:.6f}_w_{w:.6f}.npz"
            np.savez(out_dir / filename, a=a, w=w, x=x, t=t, u=u_all)
            index_rows.append((filename, float(a), float(w)))

    if x_ref is None or t_ref is None:
        raise RuntimeError("No dataset samples were generated")

    np.savez(
        out_dir / "dataset_meta.npz",
        a_values=np.array(list(a_values), dtype=float),
        w_values=np.array(list(w_values), dtype=float),
        x=x_ref,
        t=t_ref,
        dx=cfg.dx,
        dt=cfg.dt,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_end=cfg.t_end,
        include_endpoint=cfg.include_endpoint,
    )

    index_path = out_dir / "index.csv"
    with index_path.open("w", encoding="utf-8") as handle:
        handle.write("filename,a,w\n")
        for filename, a, w in index_rows:
            handle.write(f"{filename},{a:.12f},{w:.12f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1D inviscid Burgers equation simulation with Richtmyer scheme."
    )
    parser.add_argument("--a", type=float, default=0.8, help="Amplitude parameter a")
    parser.add_argument("--w", type=float, default=1.0, help="Width parameter w")
    parser.add_argument("--x-min", type=float, default=-3.0, help="Domain minimum")
    parser.add_argument("--x-max", type=float, default=3.0, help="Domain maximum")
    parser.add_argument("--dx", type=float, default=6.0e-3, help="Spatial step")
    parser.add_argument("--dt", type=float, default=1.0e-3, help="Time step")
    parser.add_argument("--t-end", type=float, default=1.0, help="Final time")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="",
        help="If set, generate a dataset over (a, w) grid and save to this directory",
    )
    parser.add_argument("--a-min", type=float, default=0.7, help="Dataset a minimum")
    parser.add_argument("--a-max", type=float, default=0.9, help="Dataset a maximum")
    parser.add_argument("--a-step", type=float, default=0.01, help="Dataset a step")
    parser.add_argument("--w-min", type=float, default=0.9, help="Dataset w minimum")
    parser.add_argument("--w-max", type=float, default=1.1, help="Dataset w maximum")
    parser.add_argument("--w-step", type=float, default=0.01, help="Dataset w step")
    parser.add_argument(
        "--no-endpoint",
        action="store_true",
        help="Do not append x_max to output (keeps periodic grid only)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=0.0,
        help="Time interval between snapshots (0 disables)",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default="",
        help="Directory to write snapshots (requires --snapshot-interval)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Path to save the plot (e.g., burgers.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig(
        x_min=args.x_min,
        x_max=args.x_max,
        dx=args.dx,
        dt=args.dt,
        t_end=args.t_end,
        a=args.a,
        w=args.w,
        include_endpoint=not args.no_endpoint,
        snapshot_interval=args.snapshot_interval,
        snapshot_dir=args.snapshot_dir,
    )

    if args.dataset_dir:
        a_values = make_param_values(args.a_min, args.a_max, args.a_step, "a")
        w_values = make_param_values(args.w_min, args.w_max, args.w_step, "w")
        generate_dataset(cfg, a_values, w_values, Path(args.dataset_dir))
        return

    x, u0, u = simulate(cfg)

    if args.no_plot:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency for plotting
        raise RuntimeError(
            "matplotlib is required for plotting. Install it or use --no-plot"
        ) from exc

    plt.figure(figsize=(8, 4))
    plt.plot(x, u0, label="t=0")
    plt.plot(x, u, label=f"t={cfg.t_end}")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("1D Inviscid Burgers (Periodic)")
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
