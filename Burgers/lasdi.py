#!/usr/bin/env python3
# type: ignore
"""LaSDI workflow for 1D Burgers.

This implementation is intentionally aligned with the reference ``LaSDI/LaSDI.py``
class interface while keeping a practical CLI for this repository.

Subcommands:
- ``train``: train AE + latent dynamics on dataset samples
- ``predict``: generate ROM trajectory for a new ``(a, w)``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from burgers_simulation import initial_condition

DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "lasdi_model"

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover - runtime dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import pysindy as ps
except Exception as exc:  # pragma: no cover - runtime dependency
    ps = None
    PYSINDY_IMPORT_ERROR = exc
else:
    PYSINDY_IMPORT_ERROR = None


@dataclass
class AEConfig:
    input_dim: int
    latent_dim: int
    hidden_sizes: Sequence[int]
    activation: str


@dataclass
class TrainConfig:
    epochs: int = 5000
    batch_size: int = 512
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-6
    time_stride: int = 1
    drop_endpoint: bool = False
    sindy_degree: int = 1
    include_interaction: bool = False
    sindy_threshold: float = 0.0
    local: bool = True
    coef_interp: bool = True
    nearest_neigh: int = 4


class AutoEncoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        act = activation_from_name(cfg.activation)

        enc_layers: List[nn.Module] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_sizes:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(act())
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_sizes):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(act())
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class EncoderWithNorm(nn.Module):
    def __init__(self, ae: AutoEncoder, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.ae = ae
        self.register_buffer("mean", torch.tensor(mean.astype(np.float32)))
        self.register_buffer("std", torch.tensor(std.astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ae.encode((x - self.mean) / self.std)


class DecoderWithDenorm(nn.Module):
    def __init__(self, ae: AutoEncoder, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.ae = ae
        self.register_buffer("mean", torch.tensor(mean.astype(np.float32)))
        self.register_buffer("std", torch.tensor(std.astype(np.float32)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_norm = self.ae.decode(z)
        return x_norm * self.std + self.mean


def activation_from_name(name: str):
    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    if name == "silu":
        return nn.SiLU
    raise ValueError(f"Unknown activation: {name}")


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for lasdi.py. Install torch before running this script."
        ) from TORCH_IMPORT_ERROR


def require_pysindy() -> None:
    if ps is None:
        raise RuntimeError(
            "PySINDy is required for lasdi.py. Install pysindy before running this script."
        ) from PYSINDY_IMPORT_ERROR


def resolve_device(device_arg: str) -> torch.device:  # type: ignore
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_index(dataset_dir: Path) -> List[Tuple[str, float, float]]:
    index_path = dataset_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"index.csv not found in {dataset_dir}")

    rows: List[Tuple[str, float, float]] = []
    with index_path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if "filename" not in header:
            raise ValueError("index.csv header is invalid")
        for line in handle:
            filename, a_str, w_str = line.strip().split(",")
            rows.append((filename, float(a_str), float(w_str)))
    return rows


def select_grid_rows(
    rows: Sequence[Tuple[str, float, float]],
    samples_per_axis: int,
) -> List[Tuple[str, float, float]]:
    if not rows:
        return []

    if samples_per_axis <= 0:
        return list(rows)

    a_values = np.array(sorted({a for _, a, _ in rows}), dtype=float)
    w_values = np.array(sorted({w for _, _, w in rows}), dtype=float)

    a_idx = np.round(np.linspace(0, len(a_values) - 1, samples_per_axis)).astype(int)
    w_idx = np.round(np.linspace(0, len(w_values) - 1, samples_per_axis)).astype(int)
    a_idx = np.asarray(list(dict.fromkeys(a_idx.tolist())), dtype=int)
    w_idx = np.asarray(list(dict.fromkeys(w_idx.tolist())), dtype=int)

    target_a = a_values[a_idx]
    target_w = w_values[w_idx]
    targets = np.array([(a, w) for a in target_a for w in target_w], dtype=float)

    params = np.asarray([[a, w] for _, a, w in rows], dtype=float)
    selected_idx: List[int] = []
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


def downsample_time(u: np.ndarray, t: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    if stride <= 1:
        return u, t
    return u[::stride], t[::stride]


def load_series(
    dataset_dir: Path,
    filename: str,
    drop_endpoint: bool,
    time_stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(dataset_dir / filename)
    u = data["u"]
    t = data["t"]
    x = data["x"]
    if drop_endpoint:
        u = u[:, :-1]
        x = x[:-1]
    u, t = downsample_time(u, t, time_stride)
    return x, t, u


def collect_snapshots(
    dataset_dir: Path,
    rows: Sequence[Tuple[str, float, float]],
    time_stride: int,
    drop_endpoint: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    trajectories: List[np.ndarray] = []
    params: List[List[float]] = []
    x_ref = None
    t_ref = None

    for filename, a, w in rows:
        x, t, u = load_series(dataset_dir, filename, drop_endpoint, time_stride)
        if x_ref is None:
            x_ref = x
            t_ref = t
        trajectories.append(u.astype(np.float32))
        params.append([a, w])

    if not trajectories:
        raise RuntimeError("No trajectories loaded from dataset")

    return np.asarray(params, dtype=float), x_ref, t_ref, trajectories


def normalize_data(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = u.mean(axis=0)
    std = u.std(axis=0)
    std[std < 1.0e-12] = 1.0
    u_norm = (u - mean) / std
    return u_norm, mean, std


def train_autoencoder(
    snapshots_norm: np.ndarray,
    ae_cfg: AEConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> AutoEncoder:
    model = AutoEncoder(ae_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.tensor(snapshots_norm.astype(np.float32)))
    batch_size = min(train_cfg.batch_size, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(train_cfg.epochs):
        running = 0.0
        count = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * batch.shape[0]
            count += int(batch.shape[0])

        if (epoch + 1) % max(1, train_cfg.epochs // 10) == 0:
            print(f"[AE] epoch {epoch + 1:5d}/{train_cfg.epochs} mse={running / count:.3e}")

    model.eval()
    return model


def build_terms(
    latent_dim: int,
    degree: int,
    include_constant: bool = True,
    include_interaction: bool = True,
) -> List[Tuple[int, ...]]:
    terms: List[Tuple[int, ...]] = [()] if include_constant else []
    for deg in range(1, degree + 1):
        if include_interaction:
            for combo in combinations_with_replacement(range(latent_dim), deg):
                terms.append(combo)
        else:
            for dim in range(latent_dim):
                terms.append((dim,) * deg)
    return terms


def evaluate_terms(z: np.ndarray, terms: Sequence[Tuple[int, ...]]) -> np.ndarray:
    values = np.empty(len(terms), dtype=float)
    for i, term in enumerate(terms):
        if len(term) == 0:
            values[i] = 1.0
        else:
            values[i] = float(np.prod(z[list(term)]))
    return values


def sindy_rhs(z: np.ndarray, coeffs: np.ndarray, terms: Sequence[Tuple[int, ...]]) -> np.ndarray:
    # coeffs shape: (latent_dim, n_terms)
    theta = evaluate_terms(z, terms)
    return coeffs @ theta


def integrate_latent(
    z0: np.ndarray,
    coeffs: np.ndarray,
    terms: Sequence[Tuple[int, ...]],
    t: np.ndarray,
) -> np.ndarray:
    z = np.empty((len(t), len(z0)), dtype=float)
    z[0] = z0
    for n in range(len(t) - 1):
        h = float(t[n + 1] - t[n])
        zn = z[n]
        k1 = sindy_rhs(zn, coeffs, terms)
        k2 = sindy_rhs(zn + 0.5 * h * k1, coeffs, terms)
        k3 = sindy_rhs(zn + 0.5 * h * k2, coeffs, terms)
        k4 = sindy_rhs(zn + h * k3, coeffs, terms)
        z[n + 1] = zn + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return z


class LaSDI:
    """Reference-style LaSDI class (aligned with ``LaSDI/LaSDI.py``)."""

    def __init__(
        self,
        encoder,
        decoder,
        NN: bool = False,
        device: str = "cpu",
        Local: bool = False,
        Coef_interp: bool = False,
        nearest_neigh: int = 4,
        Coef_interp_method: Callable | None = None,
    ):
        self.Local = Local
        self.Coef_interp = Coef_interp
        self.nearest_neigh = nearest_neigh
        self.NN = NN
        self.device = device
        self.Coef_interp_method = Coef_interp_method

        if self.Coef_interp and self.nearest_neigh < 4:
            raise ValueError("At least 4 nearest neighbors are recommended for interpolation")

        if not NN:
            self.IC_gen = lambda params: np.matmul(encoder, params)
            self.decoder = lambda traj: np.matmul(decoder, traj.T)
        else:
            import torch as _torch

            encoder.eval()
            decoder.eval()

            def _encode(IC: np.ndarray) -> np.ndarray:
                with _torch.no_grad():
                    tensor = _torch.tensor(np.asarray(IC, dtype=np.float32)).to(device)
                    return encoder(tensor).cpu().numpy()

            def _decode(traj: np.ndarray) -> np.ndarray:
                with _torch.no_grad():
                    tensor = _torch.tensor(np.asarray(traj, dtype=np.float32)).to(device)
                    return decoder(tensor).cpu().numpy()

            self.IC_gen = _encode
            self.decoder = _decode

    def train_dynamics(
        self,
        ls_trajs: Sequence[np.ndarray],
        training_values: np.ndarray,
        dt: float,
        normal: float = 1.0,
        degree: int = 1,
        include_interaction: bool = False,
        threshold: float = 0.0,
    ):
        self.normal = normal
        self.dt = dt
        self.degree = degree
        self.include_interaction = include_interaction
        self.training_values = np.asarray(training_values, dtype=float)

        data_LS = [np.asarray(traj, dtype=float) / normal for traj in ls_trajs]
        self.length = int(data_LS[0].shape[0])

        poly_library = ps.PolynomialLibrary(
            include_interaction=include_interaction,
            degree=degree,
            include_bias=True,
        )
        optimizer = ps.STLSQ(
            alpha=0.0,
            copy_X=True,
            max_iter=20,
            ridge_kw=None,
            threshold=threshold,
        )

        if not self.Local:
            model = ps.SINDy(feature_library=poly_library, optimizer=optimizer)
            model.fit(data_LS, t=dt, multiple_trajectories=True)
            self.model = model
            self.global_coeffs = np.asarray(model.coefficients(), dtype=float)
            self.terms = build_terms(
                latent_dim=self.global_coeffs.shape[0],
                degree=degree,
                include_constant=True,
                include_interaction=include_interaction,
            )
            return self.global_coeffs

        if self.Coef_interp:
            self.model_list = []
            for i in range(len(data_LS)):
                model = ps.SINDy(feature_library=poly_library, optimizer=optimizer)
                model.fit(data_LS[i], t=dt)
                self.model_list.append(np.asarray(model.coefficients(), dtype=float))

            self.model_list = np.asarray(self.model_list, dtype=float)
            self.terms = build_terms(
                latent_dim=self.model_list.shape[1],
                degree=degree,
                include_constant=True,
                include_interaction=include_interaction,
            )
            return self.model_list

        self.data_LS = data_LS
        self.poly_library = poly_library
        self.optimizer = optimizer
        self.threshold = threshold
        return None

    def _neighbor_indices(self, pred_value: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(self.training_values - pred_value[None, :], axis=1)
        k = min(self.nearest_neigh, len(dist))
        return np.argsort(dist)[:k]

    def _interpolate_coeffs(self, pred_value: np.ndarray, idx: np.ndarray) -> np.ndarray:
        neigh_params = self.training_values[idx]
        neigh_coeffs = self.model_list[idx]  # (k, latent_dim, n_terms)

        if self.Coef_interp_method is not None:
            coeffs = np.empty_like(neigh_coeffs[0])
            for i in range(coeffs.shape[0]):
                for j in range(coeffs.shape[1]):
                    f = self.Coef_interp_method(
                        neigh_params[:, 0],
                        neigh_params[:, 1],
                        neigh_coeffs[:, i, j],
                    )
                    coeffs[i, j] = float(f(pred_value[0], pred_value[1]))
            return coeffs

        dist = np.linalg.norm(neigh_params - pred_value[None, :], axis=1)
        if np.any(dist < 1.0e-14):
            return neigh_coeffs[int(np.argmin(dist))]
        w = 1.0 / np.maximum(dist, 1.0e-12)
        w /= w.sum()
        return np.tensordot(w, neigh_coeffs, axes=(0, 0))

    def generate_ROM(self, pred_IC: np.ndarray, pred_value: np.ndarray, t: np.ndarray) -> np.ndarray:
        IC = np.asarray(self.IC_gen(pred_IC), dtype=float).reshape(-1)
        pred_value = np.asarray(pred_value, dtype=float).reshape(-1)

        if not self.Local:
            coeffs = np.asarray(self.global_coeffs, dtype=float)
            latent = self.normal * integrate_latent(IC / self.normal, coeffs, self.terms, t)
            FOM_recon = self.decoder(latent)
            return FOM_recon if self.NN else FOM_recon.T

        idx = self._neighbor_indices(pred_value)

        if not self.Coef_interp:
            local = [self.data_LS[i] for i in idx]
            model = ps.SINDy(feature_library=self.poly_library, optimizer=self.optimizer)
            model.fit(local, t=self.dt, multiple_trajectories=True)
            coeffs = np.asarray(model.coefficients(), dtype=float)
            terms = build_terms(
                latent_dim=coeffs.shape[0],
                degree=self.degree,
                include_constant=True,
                include_interaction=self.include_interaction,
            )
            latent = self.normal * integrate_latent(IC / self.normal, coeffs, terms, t)
            FOM_recon = self.decoder(latent)
            return FOM_recon if self.NN else FOM_recon.T

        coeffs = self._interpolate_coeffs(pred_value, idx)
        latent = self.normal * integrate_latent(IC / self.normal, coeffs, self.terms, t)
        FOM_recon = self.decoder(latent)
        return FOM_recon if self.NN else FOM_recon.T


def save_model(
    model_dir: Path,
    ae: AutoEncoder,
    ae_cfg: AEConfig,
    mean: np.ndarray,
    std: np.ndarray,
    params: np.ndarray,
    coeffs: np.ndarray,
    terms: Sequence[Tuple[int, ...]],
    x: np.ndarray,
    t: np.ndarray,
    train_cfg: TrainConfig,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "ae_config": asdict(ae_cfg),
            "state_dict": ae.state_dict(),
        },
        model_dir / "ae.pt",
    )
    np.savez(model_dir / "normalization.npz", mean=mean, std=std)
    np.save(model_dir / "params.npy", params)
    np.save(model_dir / "sindy_coeffs.npy", coeffs)
    np.savez(model_dir / "grid.npz", x=x, t=t)

    with (model_dir / "sindy_terms.json").open("w", encoding="utf-8") as handle:
        json.dump([[int(v) for v in term] for term in terms], handle)

    with (model_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_type": "LaSDI",
                "mode": "local_coef_interp" if train_cfg.local and train_cfg.coef_interp else "global",
                "train": asdict(train_cfg),
                "ae": asdict(ae_cfg),
            },
            handle,
            indent=2,
        )


def load_model(model_dir: Path, device: torch.device):
    checkpoint = torch.load(model_dir / "ae.pt", map_location=device)
    ae_cfg = AEConfig(**checkpoint["ae_config"])
    ae = AutoEncoder(ae_cfg).to(device)
    ae.load_state_dict(checkpoint["state_dict"])
    ae.eval()

    norm = np.load(model_dir / "normalization.npz")
    mean = norm["mean"]
    std = norm["std"]

    encoder = EncoderWithNorm(ae, mean, std).to(device).eval()
    decoder = DecoderWithDenorm(ae, mean, std).to(device).eval()

    params = np.load(model_dir / "params.npy")
    coeffs = np.load(model_dir / "sindy_coeffs.npy")
    grid = np.load(model_dir / "grid.npz")
    x = grid["x"]
    t = grid["t"]

    with (model_dir / "sindy_terms.json").open("r", encoding="utf-8") as handle:
        terms = [tuple(int(v) for v in term) for term in json.load(handle)]

    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    return {
        "ae": ae,
        "encoder": encoder,
        "decoder": decoder,
        "params": params,
        "coeffs": coeffs,
        "terms": terms,
        "x": x,
        "t": t,
        "config": cfg,
    }


def encode_trajectories(
    encoder: EncoderWithNorm,
    trajectories: Sequence[np.ndarray],
    device: torch.device,  # type: ignore
) -> List[np.ndarray]:
    with torch.no_grad():
        return [
            encoder(torch.tensor(u.astype(np.float32)).to(device)).cpu().numpy()
            for u in trajectories
        ]


def collect_coefficients_for_save(lasdi: "LaSDI", train_cfg: TrainConfig) -> np.ndarray:
    if train_cfg.local and train_cfg.coef_interp:
        return np.asarray(lasdi.model_list, dtype=float)
    if not train_cfg.local:
        return np.asarray(lasdi.global_coeffs, dtype=float)[None, ...]
    raise RuntimeError(
        "Saving Local=True and Coef_interp=False models is not supported in this CLI. "
        "Use default interpolation mode or --global-model."
    )


def train_pipeline(args: argparse.Namespace) -> None:
    require_torch()
    require_pysindy()

    dataset_dir = Path(args.dataset_dir).resolve()
    model_dir = Path(args.model_dir).resolve()

    rows = load_index(dataset_dir)
    rows_sel = select_grid_rows(rows, args.samples_per_axis)
    print(
        f"Selected {len(rows_sel)} training cases "
        f"(samples_per_axis={args.samples_per_axis})"
    )
    params, x, t, trajectories = collect_snapshots(
        dataset_dir,
        rows_sel,
        time_stride=args.time_stride,
        drop_endpoint=args.drop_endpoint,
    )

    snapshots = np.vstack(trajectories)
    snapshots_norm, mean, std = normalize_data(snapshots)

    ae_cfg = AEConfig(
        input_dim=snapshots.shape[1],
        latent_dim=args.latent_dim,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        time_stride=args.time_stride,
        drop_endpoint=args.drop_endpoint,
        sindy_degree=args.sindy_degree,
        include_interaction=args.include_interaction,
        sindy_threshold=args.sindy_threshold,
        local=(not args.global_model),
        coef_interp=(not args.no_coef_interp),
        nearest_neigh=args.nearest_neigh,
    )

    device = resolve_device(args.device)

    ae = train_autoencoder(snapshots_norm, ae_cfg, train_cfg, device)

    encoder = EncoderWithNorm(ae, mean, std).to(device).eval()
    decoder = DecoderWithDenorm(ae, mean, std).to(device).eval()

    ls_trajs = encode_trajectories(encoder, trajectories, device)

    lasdi = LaSDI(
        encoder=encoder,
        decoder=decoder,
        NN=True,
        device=str(device),
        Local=train_cfg.local,
        Coef_interp=train_cfg.coef_interp,
        nearest_neigh=train_cfg.nearest_neigh,
    )

    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0

    tic = time.time()
    lasdi.train_dynamics(
        ls_trajs=ls_trajs,
        training_values=params,
        dt=dt,
        normal=1.0,
        degree=train_cfg.sindy_degree,
        include_interaction=train_cfg.include_interaction,
        threshold=train_cfg.sindy_threshold,
    )
    elapsed = time.time() - tic

    coeffs = collect_coefficients_for_save(lasdi, train_cfg)

    save_model(
        model_dir=model_dir,
        ae=ae,
        ae_cfg=ae_cfg,
        mean=mean,
        std=std,
        params=params,
        coeffs=coeffs,
        terms=lasdi.terms,
        x=x,
        t=t,
        train_cfg=train_cfg,
    )

    print(f"Trained on {len(rows_sel)} parameter cases")
    print(f"Dynamics fit time: {elapsed:.3f} s")
    print(f"Saved model to {model_dir}")

    if args.no_error_map:
        print("Skipped LaSDI error map output (--no-error-map).")
        return

    emap_args = argparse.Namespace(
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        output=(Path(args.error_map_output) if args.error_map_output else None),
        vmax=float(args.error_map_vmax),
        nearest_neigh=int(args.error_map_nearest_neigh),
        max_cases=int(args.error_map_max_cases),
        device=args.device,
    )
    error_map_pipeline(emap_args)


def predict_pipeline(args: argparse.Namespace) -> None:
    require_torch()

    model_dir = Path(args.model_dir).resolve()
    device = resolve_device(args.device)

    payload = load_model(model_dir, device)
    lasdi = _build_lasdi_for_inference(payload, device, args.nearest_neigh)

    x = np.asarray(payload["x"], dtype=float)
    t = np.asarray(payload["t"], dtype=float)
    pred_value = np.array([args.a, args.w], dtype=float)
    u0 = initial_condition(x, args.a, args.w)

    u_pred = lasdi.generate_ROM(u0, pred_value, t)

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = model_dir / f"prediction_a_{args.a:.6f}_w_{args.w:.6f}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(out_path, x=x, t=t, u=u_pred, a=float(args.a), w=float(args.w))
    print(f"Saved prediction to {out_path}")


def max_relative_error_percent(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    n_t = min(u_true.shape[0], u_pred.shape[0])
    n_x = min(u_true.shape[1], u_pred.shape[1])
    if n_t <= 0 or n_x <= 0:
        return float("nan")

    diff = u_true[:n_t, :n_x] - u_pred[:n_t, :n_x]
    denom = np.linalg.norm(u_true[:n_t, :n_x], axis=1)
    denom = np.maximum(denom, 1.0e-12)
    rel = np.linalg.norm(diff, axis=1) / denom
    return 100.0 * float(np.max(rel))


def cell_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        v = float(values[0])
        return np.array([v - 0.5, v + 0.5], dtype=float)
    mids = 0.5 * (values[:-1] + values[1:])
    left = values[0] - 0.5 * (values[1] - values[0])
    right = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[left], mids, [right]])


def plot_lasdi_error_map(
    a_values: np.ndarray,
    w_values: np.ndarray,
    error_grid: np.ndarray,
    train_params: np.ndarray,
    output_path: Path,
    vmax: float,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:
        raise RuntimeError("matplotlib is required to output error-map image.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    a_edges = cell_edges(a_values)
    w_edges = cell_edges(w_values)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    mesh = ax.pcolormesh(
        a_edges,
        w_edges,
        error_grid,
        shading="auto",
        cmap="coolwarm",
        vmin=0.0,
        vmax=vmax,
    )

    for i, w in enumerate(w_values):
        for j, a in enumerate(a_values):
            value = error_grid[i, j]
            if np.isnan(value):
                continue
            ax.text(
                float(a),
                float(w),
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=5,
                color="black",
            )

    a_index = {float(a): j for j, a in enumerate(a_values)}
    w_index = {float(w): i for i, w in enumerate(w_values)}
    for a, w in train_params:
        j = a_index.get(float(a))
        i = w_index.get(float(w))
        if j is None or i is None:
            continue
        rect = Rectangle(
            (a_edges[j], w_edges[i]),
            a_edges[j + 1] - a_edges[j],
            w_edges[i + 1] - w_edges[i],
            fill=False,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(rect)

    ax.set_xlabel("a")
    ax.set_ylabel("w")
    ax.set_title("LaSDI")
    ax.set_xlim(a_edges[0], a_edges[-1])
    ax.set_ylim(w_edges[0], w_edges[-1])
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("Maximum relative error (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _build_lasdi_for_inference(
    payload: dict[str, Any],
    device: torch.device,  # type: ignore
    nearest_neigh_override: int,
) -> LaSDI:
    cfg = payload["config"]
    train_cfg = cfg.get("train", {})
    local = bool(train_cfg.get("local", True))
    coef_interp = bool(train_cfg.get("coef_interp", True))
    nearest_neigh = (
        int(nearest_neigh_override)
        if nearest_neigh_override > 0
        else int(train_cfg.get("nearest_neigh", 4))
    )

    lasdi = LaSDI(
        encoder=payload["encoder"],
        decoder=payload["decoder"],
        NN=True,
        device=str(device),
        Local=local,
        Coef_interp=coef_interp,
        nearest_neigh=nearest_neigh,
    )
    lasdi.training_values = np.asarray(payload["params"], dtype=float)
    lasdi.terms = payload["terms"]
    lasdi.normal = 1.0
    lasdi.degree = int(train_cfg.get("sindy_degree", 1))
    lasdi.include_interaction = bool(train_cfg.get("include_interaction", False))

    coeffs = np.asarray(payload["coeffs"], dtype=float)
    if local and coef_interp:
        if coeffs.ndim != 3:
            raise RuntimeError("Expected coeff tensor with shape (n_cases, latent_dim, n_terms)")
        lasdi.model_list = coeffs
    elif not local:
        lasdi.global_coeffs = coeffs[0] if coeffs.ndim == 3 else coeffs
    else:
        raise RuntimeError("This model mode is not supported for inference")
    return lasdi


def build_error_grid(
    lasdi: LaSDI,
    dataset_dir: Path,
    rows: Sequence[Tuple[str, float, float]],
    t: np.ndarray,
    drop_endpoint: bool,
    time_stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_values = np.array(sorted({a for _, a, _ in rows}), dtype=float)
    w_values = np.array(sorted({w for _, _, w in rows}), dtype=float)
    a_index = {float(a): j for j, a in enumerate(a_values)}
    w_index = {float(w): i for i, w in enumerate(w_values)}
    error_grid = np.full((w_values.size, a_values.size), np.nan, dtype=float)

    for idx, (filename, a, w) in enumerate(rows, start=1):
        _, _, u_true = load_series(dataset_dir, filename, drop_endpoint, time_stride)
        u0 = np.asarray(u_true[0], dtype=float)
        pred_value = np.array([a, w], dtype=float)
        u_pred = lasdi.generate_ROM(u0, pred_value, t)
        err = max_relative_error_percent(u_true, u_pred)
        error_grid[w_index[float(w)], a_index[float(a)]] = err
        if idx % 50 == 0 or idx == len(rows):
            print(f"[error-map] processed {idx}/{len(rows)}")

    return a_values, w_values, error_grid


def error_map_pipeline(args: argparse.Namespace) -> None:
    require_torch()
    model_dir = Path(args.model_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    device = resolve_device(args.device)

    payload = load_model(model_dir, device)
    cfg = payload["config"]
    train_cfg = cfg.get("train", {})
    drop_endpoint = bool(train_cfg.get("drop_endpoint", False))
    time_stride = int(train_cfg.get("time_stride", 1))
    t = np.asarray(payload["t"], dtype=float)

    lasdi = _build_lasdi_for_inference(payload, device, args.nearest_neigh)
    rows = load_index(dataset_dir)
    if args.max_cases > 0:
        rows = rows[: args.max_cases]
    if not rows:
        raise RuntimeError("No rows found in dataset index")

    a_values, w_values, error_grid = build_error_grid(
        lasdi=lasdi,
        dataset_dir=dataset_dir,
        rows=rows,
        t=t,
        drop_endpoint=drop_endpoint,
        time_stride=time_stride,
    )

    out_path = Path(args.output).resolve() if args.output else model_dir / "lasdi_error_map.png"
    plot_lasdi_error_map(
        a_values=a_values,
        w_values=w_values,
        error_grid=error_grid,
        train_params=np.asarray(payload["params"], dtype=float),
        output_path=out_path,
        vmax=float(args.vmax),
    )
    np.savez(
        out_path.with_suffix(".npz"),
        a_values=a_values,
        w_values=w_values,
        max_rel_err_percent=error_grid,
        train_params=np.asarray(payload["params"], dtype=float),
    )
    print(f"Saved LaSDI error map to {out_path}")


def add_train_arguments(train: argparse.ArgumentParser) -> None:
    train.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    train.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    train.add_argument("--samples-per-axis", type=int, default=5, help="Grid samples per parameter axis")
    train.add_argument("--latent-dim", type=int, default=5)
    train.add_argument("--hidden-sizes", type=int, nargs="+", default=[100])
    train.add_argument("--activation", type=str, default="silu", choices=["sigmoid", "tanh", "relu", "silu"])
    train.add_argument("--epochs", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--lr", type=float, default=1.0e-3)
    train.add_argument("--weight-decay", type=float, default=1.0e-6)
    train.add_argument("--time-stride", type=int, default=1)
    train.add_argument("--drop-endpoint", action="store_true")
    train.add_argument("--sindy-degree", type=int, default=1)
    train.add_argument("--include-interaction", action="store_true")
    train.add_argument("--sindy-threshold", type=float, default=0.0)
    train.add_argument("--global-model", action="store_true", help="Use one global SINDy model")
    train.add_argument("--no-coef-interp", action="store_true", help="Disable coefficient interpolation in local mode")
    train.add_argument("--nearest-neigh", type=int, default=4)
    train.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    train.add_argument(
        "--no-error-map",
        action="store_true",
        help="Skip automatic error-map output after training",
    )
    train.add_argument(
        "--error-map-output",
        type=str,
        default="",
        help="Output path for error-map image (default: <model-dir>/lasdi_error_map.png)",
    )
    train.add_argument(
        "--error-map-vmax",
        type=float,
        default=5.0,
        help="Colorbar max value (percent) for automatic error map",
    )
    train.add_argument(
        "--error-map-nearest-neigh",
        type=int,
        default=-1,
        help="Override kNN when building automatic error map (<=0 uses trained value)",
    )
    train.add_argument(
        "--error-map-max-cases",
        type=int,
        default=0,
        help="Limit number of dataset rows for automatic error map (0 = all)",
    )


def add_predict_arguments(predict: argparse.ArgumentParser) -> None:
    predict.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    predict.add_argument("--a", type=float, required=True)
    predict.add_argument("--w", type=float, required=True)
    predict.add_argument("--output", type=Path, default=None)
    predict.add_argument("--nearest-neigh", type=int, default=-1, help="Override trained kNN if > 0")
    predict.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")


def add_error_map_arguments(emap: argparse.ArgumentParser) -> None:
    emap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    emap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    emap.add_argument("--output", type=Path, default=None)
    emap.add_argument("--vmax", type=float, default=5.0, help="Colorbar max value in percent")
    emap.add_argument("--nearest-neigh", type=int, default=-1, help="Override trained kNN if > 0")
    emap.add_argument("--max-cases", type=int, default=0, help="Process only first N index rows (0 = all)")
    emap.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaSDI for Burgers (reference-style rewrite)")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train AE + LaSDI latent dynamics")
    add_train_arguments(train)

    predict = sub.add_parser("predict", help="Predict trajectory for new (a,w)")
    add_predict_arguments(predict)

    emap = sub.add_parser("error-map", help="Build error-map over dataset parameters")
    add_error_map_arguments(emap)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_pipeline(args)
        return
    if args.command == "predict":
        predict_pipeline(args)
        return
    if args.command == "error-map":
        error_map_pipeline(args)
        return
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
