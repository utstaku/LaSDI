#!/usr/bin/env python3
"""Inviscid Burgers equation simulation with periodic boundary conditions.

Equation: u_t + (u^2/2)_x = 0
Spatial discretization: Godunov (exact Riemann for Burgers)
Time stepping: forward Euler with CFL control
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class SimConfig:
    n: int = 200
    length: float = 2.0 * np.pi
    t_end: float = 1.0
    cfl: float = 0.5
    ic: str = "gaussian"
    amp: float = 1.0
    snapshot_interval: float = 0.0
    snapshot_dir: str = ""


def initial_condition(x: np.ndarray, ic: str, amp: float) -> np.ndarray:
    if ic == "sine":
        return amp * np.sin(x)
    if ic == "square":
        return amp * np.where(np.sin(x) >= 0.0, 1.0, -1.0)
    if ic == "gaussian":
        return amp * np.exp(-((x - x.mean()) ** 2) / 0.1)
    raise ValueError(f"Unknown initial condition: {ic}")


def godunov_flux(u_left: np.ndarray, u_right: np.ndarray) -> np.ndarray:
    """Exact Godunov flux for inviscid Burgers equation."""
    f_left = 0.5 * u_left * u_left
    f_right = 0.5 * u_right * u_right

    rare = u_left <= u_right

    # Rarefaction fan
    flux_rare = np.where(
        u_left >= 0.0,
        f_left,
        np.where(u_right <= 0.0, f_right, 0.0),
    )

    # Shock
    shock_speed = 0.5 * (u_left + u_right)
    flux_shock = np.where(shock_speed >= 0.0, f_left, f_right)

    return np.where(rare, flux_rare, flux_shock)


def step(u: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """Advance one time step with periodic boundaries."""
    u_left = u
    u_right = np.roll(u, -1)
    flux = godunov_flux(u_left, u_right)  # flux at i+1/2
    return u - (dt / dx) * (flux - np.roll(flux, 1))


def write_snapshot(out_dir: Path, index: int, t: float, x: np.ndarray, u: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"snapshot_{index:05d}.npz"
    np.savez(path, t=t, x=x, u=u)


def simulate(cfg: SimConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, cfg.length, cfg.n, endpoint=False)
    u0 = initial_condition(x, cfg.ic, cfg.amp)
    u = u0.copy()

    t = 0.0
    dx = cfg.length / cfg.n
    next_snapshot = None
    snapshot_index = 0
    snapshot_dir = None
    if cfg.snapshot_interval > 0.0 and cfg.snapshot_dir:
        snapshot_dir = Path(cfg.snapshot_dir)
        write_snapshot(snapshot_dir, snapshot_index, t, x, u)
        snapshot_index += 1
        next_snapshot = cfg.snapshot_interval

    while t < cfg.t_end:
        max_speed = np.max(np.abs(u))
        if max_speed == 0.0:
            dt = cfg.t_end - t
        else:
            dt = cfg.cfl * dx / max_speed
            dt = min(dt, cfg.t_end - t)

        if next_snapshot is not None and t + dt > next_snapshot:
            dt = next_snapshot - t

        u = step(u, dx, dt)
        t += dt

        if next_snapshot is not None and t >= next_snapshot - 1.0e-12:
            write_snapshot(snapshot_dir, snapshot_index, t, x, u)
            snapshot_index += 1
            next_snapshot += cfg.snapshot_interval

    return x, u0, u


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inviscid Burgers equation simulation (periodic boundary)."
    )
    parser.add_argument("--n", type=int, default=200, help="Number of grid points")
    parser.add_argument("--length", type=float, default=2.0 * np.pi, help="Domain length")
    parser.add_argument("--t_end", type=float, default=1.0, help="Final time")
    parser.add_argument("--cfl", type=float, default=0.5, help="CFL number")
    parser.add_argument(
        "--ic",
        choices=["sine", "square", "gaussian"],
        default="gaussian",
        help="Initial condition",
    )
    parser.add_argument("--amp", type=float, default=1.0, help="Amplitude of IC")
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
        n=args.n,
        length=args.length,
        t_end=args.t_end,
        cfl=args.cfl,
        ic=args.ic,
        amp=args.amp,
        snapshot_interval=args.snapshot_interval,
        snapshot_dir=args.snapshot_dir,
    )

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
    plt.title("Inviscid Burgers Equation (Periodic)")
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    main()
