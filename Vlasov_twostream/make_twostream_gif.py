#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


def _resolve_paths(input_path: str, output_path: str | None) -> tuple[Path, Path]:
    in_path = Path(input_path)
    if in_path.is_dir():
        data_path = in_path / "animation_data.npz"
        default_out = in_path / "twostream.gif"
    else:
        data_path = in_path
        default_out = in_path.with_suffix(".gif")

    out_path = Path(output_path) if output_path is not None else default_out
    return data_path, out_path


def _load_animation_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Animation data not found: {path}")
    data = np.load(path)
    required = {"t", "f", "x", "v"}
    missing = required.difference(data.files)
    if missing:
        miss = ", ".join(sorted(missing))
        raise KeyError(f"Missing keys in {path}: {miss}")
    return data["t"], data["f"], data["x"], data["v"]


def make_gif(
    input_path: str,
    output_path: str | None = None,
    fps: int = 15,
    frame_step: int = 1,
    dpi: int = 120,
    cmap: str = "inferno",
    title_prefix: str = "Two-stream instability",
) -> Path:
    data_path, out_path = _resolve_paths(input_path, output_path)
    t, f, x, v = _load_animation_npz(data_path)

    frame_step = max(1, int(frame_step))
    idx = np.arange(0, len(t), frame_step, dtype=int)
    if idx.size == 0:
        raise ValueError("No frames selected. Check frame_step and input data.")

    t_plot = t[idx]
    f_plot = f[idx]

    fmin = float(np.min(f_plot))
    fmax = float(np.max(f_plot))

    fig, ax0 = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)

    im = ax0.imshow(
        f_plot[0].T,
        origin="lower",
        aspect="auto",
        extent=[float(x[0]), float(x[-1]), float(v[0]), float(v[-1])],
        cmap=cmap,
        vmin=fmin,
        vmax=fmax,
    )
    ax0.set_ylabel("v")
    cbar = fig.colorbar(im, ax=ax0, pad=0.01)
    cbar.set_label("f(x, v)")

    ax0.set_xlabel("x")

    title = ax0.set_title(f"{title_prefix} | t={t_plot[0]:.3f}")

    def update(i: int):
        im.set_data(f_plot[i].T)
        title.set_text(f"{title_prefix} | t={t_plot[i]:.3f}")
        return im, title

    anim = FuncAnimation(fig, update, frames=len(t_plot), interval=1000 / max(1, fps), blit=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    except Exception as exc:
        msg = (
            "Failed to write GIF with PillowWriter. "
            "Install pillow (and matplotlib if needed), then retry."
        )
        raise RuntimeError(msg) from exc
    finally:
        plt.close(fig)

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GIF directly from a case directory or animation_data.npz."
    )
    parser.add_argument(
        "input",
        help="Case directory containing animation_data.npz, or the .npz file itself.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output GIF path. Default: <case_dir>/twostream.gif",
    )
    parser.add_argument("--fps", type=int, default=15, help="Frames per second for the GIF.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth frame from animation_data.npz.",
    )
    parser.add_argument("--dpi", type=int, default=120, help="GIF render DPI.")
    parser.add_argument("--cmap", default="inferno", help="Matplotlib colormap for f(x,v).")
    parser.add_argument(
        "--title-prefix",
        default="Two-stream instability",
        help="Prefix text for frame title.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = make_gif(
        input_path=args.input,
        output_path=args.output,
        fps=args.fps,
        frame_step=args.frame_step,
        dpi=args.dpi,
        cmap=args.cmap,
        title_prefix=args.title_prefix,
    )
    print(f"[OK] GIF saved: {out}")


if __name__ == "__main__":
    main()
