#!/usr/bin/env python3
# type: ignore
"""
gLaSDI workflow for 1D Burgers (discrete parameter space).

This file extends the baseline LaSDI code by adding:
  - Joint AE + local SINDy training with gLaSDI loss (Eq. in main.tex)
  - Residual-based greedy sampling over Dh (Algorithm 1)
  - Residual-based error indicator (Eq. 23) to pick next parameter (Eq. 24)
  - k-NN convex interpolation of SINDy coefficients (Algorithm 3)

Notes:
- The greedy sampling is performed over the discrete parameter set Dh given by index.csv.
- `--nsubset` can optionally restrict candidate evaluations to a random subset for speed.

Dataset assumption:
- dataset_dir/index.csv has: filename,a,w
- each .npz has keys: u (Nt+1, Nx), t (Nt+1,), x (Nx,)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Sequence, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from burgers_simulation import initial_condition  # noqa: E402


LEARNING_CASES = 5  # baseline: 4 corners + center for (a,w)
DEFAULT_DATASET_DIR = SCRIPT_DIR / "dataset"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "lasdi_model"
DEFAULT_GMODEL_DIR = SCRIPT_DIR / "glasdi_model"

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover
    import torch as torch_typing


# -----------------------
# Configs / Model
# -----------------------

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
    time_stride: int = 1
    drop_endpoint: bool = False
    sindy_degree: int = 1
    sindy_threshold: float = 0.05
    sindy_max_iter: int = 10


def activation_from_name(name: str):
    name = name.lower()
    if name == "sigmoid":
        return nn.Sigmoid
    if name == "tanh":
        return nn.Tanh
    if name == "relu":
        return nn.ReLU
    raise ValueError(f"Unknown activation: {name}")


class AutoEncoder(nn.Module):
    def __init__(self, cfg: AEConfig):
        super().__init__()
        act = activation_from_name(cfg.activation)

        enc_layers: List["torch_typing.nn.Module"] = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_sizes:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(act())
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List["torch_typing.nn.Module"] = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_sizes):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(act())
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.decode(self.encode(x))


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required. Install torch before running this script."
        ) from TORCH_IMPORT_ERROR


# -----------------------
# IO / dataset helpers
# -----------------------

def load_index(dataset_dir: Path) -> List[Tuple[str, float, float]]:
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


def downsample_time(u: np.ndarray, t: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    if stride <= 1:
        return u, t
    return u[::stride], t[::stride]


def load_series(
    dataset_dir: Path, filename: str, drop_endpoint: bool, time_stride: int
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


def select_corner_center_rows(
    rows: Sequence[Tuple[str, float, float]],
) -> List[Tuple[str, float, float]]:
    if not rows:
        return []
    params = np.asarray([[a, w] for _, a, w in rows], dtype=float)
    a_min, w_min = params.min(axis=0)
    a_max, w_max = params.max(axis=0)
    targets = np.asarray(
        [
            [a_min, w_min],
            [a_min, w_max],
            [a_max, w_min],
            [a_max, w_max],
            [0.5 * (a_min + a_max), 0.5 * (w_min + w_max)],
        ],
        dtype=float,
    )
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
    if len(selected_idx) < LEARNING_CASES:
        for idx in range(len(rows)):
            if idx not in selected_set:
                selected_idx.append(idx)
                selected_set.add(idx)
            if len(selected_idx) == LEARNING_CASES:
                break
    return [rows[idx] for idx in selected_idx]


def select_corner_rows(
    rows: Sequence[Tuple[str, float, float]],
) -> List[Tuple[str, float, float]]:
    if not rows:
        return []
    params = np.asarray([[a, w] for _, a, w in rows], dtype=float)
    a_min, w_min = params.min(axis=0)
    a_max, w_max = params.max(axis=0)
    targets = np.asarray(
        [
            [a_min, w_min],
            [a_min, w_max],
            [a_max, w_min],
            [a_max, w_max],
        ],
        dtype=float,
    )
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
    return [rows[idx] for idx in selected_idx]


def collect_snapshots_from_indices(
    dataset_dir: Path,
    rows: Sequence[Tuple[str, float, float]],
    selected_indices: Sequence[int],
    time_stride: int,
    drop_endpoint: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    snapshots = []
    params = []
    x_ref = None
    t_ref = None
    for idx in selected_indices:
        filename, a, w = rows[idx]
        x, t, u = load_series(dataset_dir, filename, drop_endpoint, time_stride)
        if x_ref is None:
            x_ref = x
            t_ref = t
        snapshots.append(u)
        params.append([a, w])
    if not snapshots:
        raise RuntimeError("No snapshots loaded for selected indices")
    u_all = np.vstack(snapshots)
    return np.asarray(params, dtype=float), x_ref, t_ref, u_all


def collect_sequences_from_indices(
    dataset_dir: Path,
    rows: Sequence[Tuple[str, float, float]],
    selected_indices: Sequence[int],
    time_stride: int,
    drop_endpoint: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = []
    sequences = []
    x_ref = None
    t_ref = None
    for idx in selected_indices:
        filename, a, w = rows[idx]
        x, t, u = load_series(dataset_dir, filename, drop_endpoint, time_stride)
        if x_ref is None:
            x_ref = x
            t_ref = t
        sequences.append(u)
        params.append([a, w])
    if not sequences:
        raise RuntimeError("No snapshots loaded for selected indices")
    u_seq = np.stack(sequences, axis=0)  # (Nmu, Nt+1, Nx)
    u_all = np.vstack(sequences)
    return np.asarray(params, dtype=float), x_ref, t_ref, u_seq, u_all


def normalize_data(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = u.mean(axis=0)
    std = u.std(axis=0)
    std[std < 1.0e-12] = 1.0
    u_norm = (u - mean) / std
    return u_norm, mean, std


# -----------------------
# AE training / encoding
# -----------------------

def train_autoencoder(
    u_norm: np.ndarray,
    ae_cfg: AEConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,  # type: ignore
    model: Optional[AutoEncoder] = None,
) -> AutoEncoder:
    require_torch()
    if model is None:
        model = AutoEncoder(ae_cfg).to(device)

    dataset = TensorDataset(torch.from_numpy(u_norm).float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    pbar = tqdm(range(epochs), desc="Training AutoEncoder", leave=False)
    for _ in pbar:
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch.size(0)
        avg = running / len(dataset)
        pbar.set_postfix({"loss": f"{avg:.3e}"})
    pbar.close()
    return model


def encode_series(
    model: AutoEncoder,
    u: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,  # type: ignore
    batch_size: int,
) -> np.ndarray:
    require_torch()
    model.eval()
    u_norm = (u - mean) / std
    z_list = []
    with torch.no_grad():
        for i in range(0, u_norm.shape[0], batch_size):
            batch = torch.from_numpy(u_norm[i:i + batch_size]).float().to(device)
            z = model.encode(batch).cpu().numpy()
            z_list.append(z)
    return np.vstack(z_list)


def decode_series(
    model: AutoEncoder,
    z: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,  # type: ignore
    batch_size: int,
) -> np.ndarray:
    require_torch()
    model.eval()
    u_list = []
    with torch.no_grad():
        for i in range(0, z.shape[0], batch_size):
            batch = torch.from_numpy(z[i:i + batch_size]).float().to(device)
            u_norm = model.decode(batch).cpu().numpy()
            u_list.append(u_norm)
    u_norm = np.vstack(u_list)
    return u_norm * std + mean


# -----------------------
# SINDy helpers (baseline)
# -----------------------

def build_terms(latent_dim: int, degree: int, include_constant: bool = True):
    terms = []
    if include_constant:
        terms.append(("1",))
    if degree >= 1:
        for i in range(latent_dim):
            terms.append(("z", i))
    if degree >= 2:
        for i in range(latent_dim):
            for j in range(i, latent_dim):
                terms.append(("zz", i, j))
    return terms


def term_names(terms) -> List[str]:
    names = []
    for term in terms:
        if term[0] == "1":
            names.append("1")
        elif term[0] == "z":
            names.append(f"z{term[1]}")
        elif term[0] == "zz":
            names.append(f"z{term[1]}*z{term[2]}")
    return names


def evaluate_terms(z: np.ndarray, terms) -> np.ndarray:
    values = np.empty(len(terms), dtype=z.dtype)
    for k, term in enumerate(terms):
        if term[0] == "1":
            values[k] = 1.0
        elif term[0] == "z":
            values[k] = z[term[1]]
        elif term[0] == "zz":
            values[k] = z[term[1]] * z[term[2]]
    return values


def build_library(z: np.ndarray, terms) -> np.ndarray:
    theta = np.empty((z.shape[0], len(terms)), dtype=z.dtype)
    for i in range(z.shape[0]):
        theta[i] = evaluate_terms(z[i], terms)
    return theta


def time_derivative(z: np.ndarray, dt: float) -> np.ndarray:
    dzdt = np.empty_like(z)
    dzdt[1:-1] = (z[2:] - z[:-2]) / (2.0 * dt)
    dzdt[0] = (z[1] - z[0]) / dt
    dzdt[-1] = (z[-1] - z[-2]) / dt
    return dzdt


def stlsq(theta: np.ndarray, dzdt: np.ndarray, threshold: float, max_iter: int) -> np.ndarray:
    xi = np.linalg.lstsq(theta, dzdt, rcond=None)[0]  # (n_terms, latent_dim)
    for _ in range(max_iter):
        small = np.abs(xi) < threshold
        xi[small] = 0.0
        for k in range(xi.shape[1]):
            big = ~small[:, k]
            if not np.any(big):
                continue
            xi[big, k] = np.linalg.lstsq(theta[:, big], dzdt[:, k], rcond=None)[0]
    return xi


def fit_sindy(z: np.ndarray, dt: float, degree: int, threshold: float, max_iter: int):
    terms = build_terms(z.shape[1], degree)
    theta = build_library(z, terms)
    dzdt = time_derivative(z, dt)
    xi = stlsq(theta, dzdt, threshold, max_iter)
    return xi, terms


def build_library_torch(z: torch.Tensor, terms) -> torch.Tensor:  # type: ignore
    cols = []
    for term in terms:
        if term[0] == "1":
            cols.append(torch.ones(z.shape[0], dtype=z.dtype, device=z.device))
        elif term[0] == "z":
            cols.append(z[:, term[1]])
        elif term[0] == "zz":
            cols.append(z[:, term[1]] * z[:, term[2]])
        else:
            raise ValueError(f"Unknown term kind: {term[0]}")
    return torch.stack(cols, dim=1)


def time_derivative_torch(y: torch.Tensor, dt: float) -> torch.Tensor:  # type: ignore
    dydt = torch.empty_like(y)
    dydt[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    dydt[0] = (y[1] - y[0]) / dt
    dydt[-1] = (y[-1] - y[-2]) / dt
    return dydt


def jvp_tensor(
    func,
    primal: torch.Tensor,  # type: ignore
    tangent: torch.Tensor,  # type: ignore
) -> torch.Tensor:
    """
    Jacobian-vector product helper: returns J_func(primal) @ tangent.
    """
    _, jvp_out = torch.autograd.functional.jvp(
        func,
        (primal,),
        (tangent,),
        create_graph=True,
        strict=False,
    )
    return jvp_out


def init_sindy_coeffs_from_model(
    model: AutoEncoder,
    u_seq_norm: np.ndarray,  # (Nmu, Nt+1, Nx), normalized
    dt: float,
    terms,
    threshold: float,
    max_iter: int,
    device: torch.device,  # type: ignore
) -> np.ndarray:
    model.eval()
    coeffs = []
    with torch.no_grad():
        for i in range(u_seq_norm.shape[0]):
            u_t = torch.from_numpy(u_seq_norm[i]).float().to(device)
            z = model.encode(u_t).cpu().numpy()
            theta = build_library(z, terms)
            dzdt = time_derivative(z, dt)
            xi = stlsq(theta, dzdt, threshold, max_iter)
            coeffs.append(xi)
    return np.stack(coeffs, axis=0)


def train_joint_glasdi(
    model: Optional[AutoEncoder],
    ae_cfg: AEConfig,
    u_seq_norm: np.ndarray,  # (Nmu, Nt+1, Nx), normalized
    dt: float,
    terms,
    epochs: int,
    batch_size: int,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    beta4: float,
    sindy_threshold: float,
    sindy_max_iter: int,
    device: torch.device,  # type: ignore
) -> Tuple[AutoEncoder, np.ndarray, Dict[str, float]]:
    """
    Joint optimization with gLaSDI-style loss:
      L = beta1*L_AE + beta2*L_SINDy + beta3*L_VEL + beta4*||Xi||^2.
    """
    require_torch()
    if model is None:
        model = AutoEncoder(ae_cfg).to(device)
    else:
        model = model.to(device)

    xi_init = init_sindy_coeffs_from_model(
        model=model,
        u_seq_norm=u_seq_norm,
        dt=dt,
        terms=terms,
        threshold=sindy_threshold,
        max_iter=sindy_max_iter,
        device=device,
    )
    xi_param = nn.Parameter(torch.from_numpy(xi_init).float().to(device))

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": [xi_param]}],
        lr=lr,
    )
    mse = nn.MSELoss()

    u_seq_t = torch.from_numpy(u_seq_norm).float().to(device)
    n_cases = u_seq_t.shape[0]
    n_snap = n_cases * u_seq_t.shape[1]
    flat_u = u_seq_t.reshape(n_snap, u_seq_t.shape[2])
    ae_batch = max(1, min(int(batch_size), n_snap))

    stats = {"loss": 0.0, "lae": 0.0, "lsindy": 0.0, "lvel": 0.0, "l2": 0.0}
    pbar = tqdm(range(epochs), desc="Training gLaSDI (joint)", leave=False)
    for _ in pbar:
        model.train()
        optimizer.zero_grad()

        if ae_batch == n_snap:
            u_batch = flat_u
        else:
            pick = torch.randperm(n_snap, device=device)[:ae_batch]
            u_batch = flat_u[pick]
        u_batch_hat = model(u_batch)
        loss_ae = mse(u_batch_hat, u_batch)

        loss_sindy = torch.tensor(0.0, device=device)
        loss_vel = torch.tensor(0.0, device=device)
        for i in range(n_cases):
            u_i = u_seq_t[i]
            z_i = model.encode(u_i)

            theta_i = build_library_torch(z_i, terms)
            u_dot_true = time_derivative_torch(u_i, dt)
            # z_dot_true = J_enc(u_i) @ u_dot_true
            z_dot_true = jvp_tensor(model.encode, u_i, u_dot_true)
            z_dot_hat = theta_i @ xi_param[i]
            loss_sindy = loss_sindy + mse(z_dot_hat, z_dot_true)

            # u_dot_hat = J_dec(z_i) @ z_dot_hat
            u_dot_hat = jvp_tensor(model.decode, z_i, z_dot_hat)
            loss_vel = loss_vel + mse(u_dot_hat, u_dot_true)

        loss_sindy = loss_sindy / n_cases
        loss_vel = loss_vel / n_cases
        loss_l2 = torch.mean(xi_param * xi_param)

        loss = beta1 * loss_ae + beta2 * loss_sindy + beta3 * loss_vel + beta4 * loss_l2
        loss.backward()
        optimizer.step()

        stats = {
            "loss": float(loss.detach().cpu().item()),
            "lae": float(loss_ae.detach().cpu().item()),
            "lsindy": float(loss_sindy.detach().cpu().item()),
            "lvel": float(loss_vel.detach().cpu().item()),
            "l2": float(loss_l2.detach().cpu().item()),
        }
        pbar.set_postfix(
            {
                "L": f"{stats['loss']:.2e}",
                "LAE": f"{stats['lae']:.2e}",
                "LS": f"{stats['lsindy']:.2e}",
                "LV": f"{stats['lvel']:.2e}",
            }
        )
    pbar.close()

    coeffs = xi_param.detach().cpu().numpy()
    return model, coeffs, stats


# -----------------------
# kNN convex interpolation (Algorithm 3)
# -----------------------

def mahalanobis_dist2(x: np.ndarray, y: np.ndarray, inv_cov: np.ndarray) -> float:
    d = x - y
    return float(d.T @ inv_cov @ d)


def interpolate_coeffs_knn_convex(
    params: np.ndarray,      # (N, p)
    coeffs: np.ndarray,      # (N, n_terms, r)
    target: np.ndarray,      # (p,)
    k: int,
    use_mahalanobis: bool = True,
) -> np.ndarray:
    """
    Convex kNN interpolation. We enforce weights >=0 and sum(weights)=1 by normalized inverse-distance^2.
    Similar structure to review Eq.(25)-(26) (kNN with partition of unity / convexity). 
    """
    if params.shape[0] == 1 or k <= 1:
        idx = int(np.argmin(np.linalg.norm(params - target, axis=1)))
        return coeffs[idx]

    k = min(k, params.shape[0])

    if use_mahalanobis:
        cov = np.cov(params.T)
        cov = cov + 1.0e-12 * np.eye(cov.shape[0])
        inv_cov = np.linalg.inv(cov)
        d2 = np.array([mahalanobis_dist2(p, target, inv_cov) for p in params], dtype=float)
        d = np.sqrt(np.maximum(d2, 0.0))
    else:
        d = np.linalg.norm(params - target, axis=1)

    idx = np.argsort(d)[:k]
    # weights ~ 1 / d^2, normalized => sum=1
    w = 1.0 / (d[idx] ** 2 + 1.0e-12)
    w = w / w.sum()
    return np.tensordot(w, coeffs[idx], axes=(0, 0))


# -----------------------
# Latent integration (baseline)
# -----------------------

def integrate_latent(z0: np.ndarray, coeffs: np.ndarray, terms, dt: float, steps: int) -> np.ndarray:
    z = np.empty((steps + 1, z0.size), dtype=z0.dtype)
    z[0] = z0

    def rhs(z_state: np.ndarray) -> np.ndarray:
        theta = evaluate_terms(z_state, terms)
        return theta @ coeffs

    for n in range(steps):
        k1 = rhs(z[n])
        k2 = rhs(z[n] + 0.5 * dt * k1)
        k3 = rhs(z[n] + 0.5 * dt * k2)
        k4 = rhs(z[n] + dt * k3)
        z[n + 1] = z[n] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return z


# -----------------------
# Residual-based error indicator (Eq. 23)
# -----------------------

def dudx_central_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    # periodic central difference
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def burgers_residual(u_n: np.ndarray, u_nm1: np.ndarray, dt: float, dx: float) -> np.ndarray:
    """
    Discrete residual approximation for inviscid Burgers:
      u_t + u u_x = 0
    Use:
      (u_n - u_{n-1})/dt + u_n * (du_n/dx) ≈ 0
    """
    ut = (u_n - u_nm1) / dt
    ux = dudx_central_periodic(u_n, dx)
    return ut + u_n * ux


def residual_indicator(u_hat: np.ndarray, dt: float, dx: float, nts_frac: float = 0.1) -> float:
    """
    Time-averaged residual indicator, analogous to Eq.(23) in the gLaSDI paper:
      e_res = (1/(Nts+1)) sum_{n=0}^{Nts} || r(u_n; u_{n-1}, mu) ||_L2
    Here we start from n=1.
    """
    Nt = u_hat.shape[0] - 1
    Nts = max(1, int(math.ceil(nts_frac * Nt)))
    # evaluate at early time steps for efficiency
    norms = []
    for n in range(1, Nts + 1):
        r = burgers_residual(u_hat[n], u_hat[n - 1], dt, dx)
        # L2 over space with dx
        norms.append(float(np.sqrt(np.sum(r * r) * dx)))
    return float(np.mean(norms))


def max_relative_error(u_true: np.ndarray, u_hat: np.ndarray, dx: float) -> float:
    """
    emax(U, Uhat) = max_n ||u_n - uhat_n|| / ||u_n|| (L2)
    """
    assert u_true.shape == u_hat.shape
    Nt = u_true.shape[0]
    errs = []
    for n in range(Nt):
        num = np.sqrt(np.sum((u_true[n] - u_hat[n]) ** 2) * dx)
        den = np.sqrt(np.sum((u_true[n]) ** 2) * dx) + 1.0e-12
        errs.append(float(num / den))
    return float(np.max(errs))


# -----------------------
# Core ROM evaluation (Algorithm 3)
# -----------------------

def rom_predict_u(
    model: AutoEncoder,
    mean: np.ndarray,
    std: np.ndarray,
    params_train: np.ndarray,
    coeffs_train: np.ndarray,
    terms,
    x: np.ndarray,
    t: np.ndarray,
    a: float,
    w: float,
    device: torch.device,  # type: ignore
    batch_size: int,
    k: int,
    drop_endpoint: bool,
    use_mahalanobis: bool = True,
) -> np.ndarray:
    """
    Algorithm 3-like evaluation:
      1) find kNN in parameter space
      2) convex-interpolate Xi
      3) z0 = enc(u0)
      4) integrate latent ODE
      5) decode back to u
    """
    dt = float(t[1] - t[0])
    dx = float(x[1] - x[0])
    steps = t.size - 1

    target = np.array([a, w], dtype=float)
    xi_interp = interpolate_coeffs_knn_convex(params_train, coeffs_train, target, k, use_mahalanobis)

    u0 = initial_condition(x, a, w)
    if not drop_endpoint:
        u0[-1] = u0[0]
    z0 = encode_series(model, u0[None, :], mean, std, device, batch_size)[0]

    z = integrate_latent(z0, xi_interp, terms, dt, steps)
    u_hat = decode_series(model, z, mean, std, device, batch_size)

    if not drop_endpoint:
        u_hat[:, -1] = u_hat[:, 0]
    return u_hat


# -----------------------
# gLaSDI training (Algorithm 1-2)
# -----------------------

def gtrain_pipeline(args: argparse.Namespace) -> None:
    require_torch()
    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)

    rows = load_index(dataset_dir)
    if not rows:
        raise RuntimeError("Dataset index is empty")

    # Discrete parameter space Dh := all rows
    Dh_indices = list(range(len(rows)))

    # Initialize D0
    rng = np.random.default_rng(args.seed)
    if args.init == "corners":
        init_rows = select_corner_rows(rows)
        init_indices = [rows.index(r) for r in init_rows]
    elif args.init == "corner_center":
        init_rows = select_corner_center_rows(rows)
        init_indices = [rows.index(r) for r in init_rows]
    else:
        init_indices = rng.choice(Dh_indices, size=min(args.n0, len(Dh_indices)), replace=False).tolist()

    sampled: List[int] = list(init_indices)
    sampled_set = set(sampled)

    print(f"[gLaSDI] init |Dv|={len(sampled)}")
    for idx in sampled:
        fn, a, w = rows[idx]
        print(f"  init idx={idx} (a={a:.6f}, w={w:.6f}) file={fn}")

    # device
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("device:", device)

    # bookkeeping
    v = 1
    w_level = 1  # random-subset level (Algorithm 2)
    Nsubset = int(args.nsubset)
    model: Optional[AutoEncoder] = None
    mean = None
    std = None
    x_ref = None
    t_ref = None
    params_train = None
    coeffs_train = None
    ae_cfg: Optional[AEConfig] = None
    terms = build_terms(args.latent_dim, args.sindy_degree)

    # store history
    history: List[Dict[str, float]] = []

    total_epochs = int(args.nepoch)
    nup = max(1, int(args.nup))
    epoch = 0

    while epoch < total_epochs:
        # (A) build DB_{v-1} snapshots and normalize
        params_train, x_ref, t_ref, u_seq, u_all = collect_sequences_from_indices(
            dataset_dir, rows, sampled, args.time_stride, args.drop_endpoint
        )
        if t_ref.size < 2:
            raise RuntimeError("Time grid must contain at least two points")
        u_norm, mean, std = normalize_data(u_all)
        u_seq_norm = (u_seq - mean[None, None, :]) / std[None, None, :]

        ae_cfg = AEConfig(
            input_dim=u_norm.shape[1],
            latent_dim=args.latent_dim,
            hidden_sizes=tuple(args.hidden_sizes),
            activation=args.activation,
        )

        # (B) Jointly update theta_enc, theta_dec, Xi with gLaSDI loss
        dt = float(t_ref[1] - t_ref[0])
        this_round = min(nup, total_epochs - epoch)
        model, coeffs_train, loss_stats = train_joint_glasdi(
            model=model,
            ae_cfg=ae_cfg,
            u_seq_norm=u_seq_norm,
            dt=dt,
            terms=terms,
            epochs=this_round,
            batch_size=args.batch_size,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            beta3=args.beta3,
            beta4=args.beta4,
            sindy_threshold=args.sindy_threshold,
            sindy_max_iter=args.sindy_max_iter,
            device=device,
        )
        epoch += this_round

        # Greedy update every Nup epochs (and optionally at the final partial round)
        if (epoch % nup != 0) and (epoch < total_epochs):
            continue

        # (C) evaluate residual indicator over unsampled Dh (Eq. 23)
        unsampled = [i for i in Dh_indices if i not in sampled_set]
        if not unsampled:
            print("[gLaSDI] no remaining unsampled parameters; stopping.")
            break

        if Nsubset > 0:
            subset_size = min(int(Nsubset), len(unsampled))
            candidates = rng.choice(unsampled, size=subset_size, replace=False).tolist()
        else:
            candidates = unsampled
            subset_size = len(candidates)

        # choose mu* by Eq. 24
        dx = float(x_ref[1] - x_ref[0])
        e_res_candidates: Dict[int, float] = {}
        for idx in tqdm(candidates, desc=f"[gLaSDI] eval candidates (size={subset_size})", leave=False):
            _, a, w = rows[idx]
            u_hat = rom_predict_u(
                model=model,
                mean=mean,
                std=std,
                params_train=params_train,
                coeffs_train=coeffs_train,
                terms=terms,
                x=x_ref,
                t=t_ref,
                a=a,
                w=w,
                device=device,
                batch_size=args.batch_size,
                k=args.knn,
                drop_endpoint=args.drop_endpoint,
                use_mahalanobis=not args.no_mahalanobis,
            )
            e_res_candidates[idx] = residual_indicator(u_hat, dt=dt, dx=dx, nts_frac=args.nts_frac)

        mu_star, eres_star = max(e_res_candidates.items(), key=lambda kv: kv[1])
        fn_star, a_star, w_star = rows[mu_star]
        print(f"[gLaSDI] v={v} picked mu* idx={mu_star} (a={a_star:.6f}, w={w_star:.6f}) eres={eres_star:.3e}")

        # (D) "Collect U^(Nmu+1)" step: add selected data point from discrete Dh
        sampled.append(mu_star)
        sampled_set.add(mu_star)

        # (E) Eq.(25)-(27): estimate maximum relative error from sampled set
        Emax = []
        Eres = []
        for idx in sampled:
            filename, a, w = rows[idx]
            _, _, u_true = load_series(dataset_dir, filename, args.drop_endpoint, args.time_stride)
            u_hat = rom_predict_u(
                model=model,
                mean=mean,
                std=std,
                params_train=params_train,
                coeffs_train=coeffs_train,
                terms=terms,
                x=x_ref,
                t=t_ref,
                a=a,
                w=w,
                device=device,
                batch_size=args.batch_size,
                k=args.knn,
                drop_endpoint=args.drop_endpoint,
                use_mahalanobis=not args.no_mahalanobis,
            )
            Emax.append(max_relative_error(u_true, u_hat, dx=dx))
            Eres.append(residual_indicator(u_hat, dt=dt, dx=dx, nts_frac=args.nts_frac))

        Emax_arr = np.asarray(Emax, dtype=float)
        Eres_arr = np.asarray(Eres, dtype=float)
        A = np.column_stack([Eres_arr, np.ones_like(Eres_arr)])
        sol, *_ = np.linalg.lstsq(A, Emax_arr, rcond=None)
        k_star, b_star = float(sol[0]), float(sol[1])
        emax_est = float(k_star * float(np.max(Eres_arr)) + b_star)

        history.append(
            {
                "epoch": float(epoch),
                "v": float(v),
                "Dv": float(len(sampled)),
                "candidates": float(subset_size),
                "Nsubset": float(Nsubset),
                "w_level": float(w_level),
                "mu_star": float(mu_star),
                "e_res_star": float(eres_star),
                "k_star": k_star,
                "b_star": b_star,
                "emax_est": emax_est,
                "loss": float(loss_stats["loss"]),
                "loss_ae": float(loss_stats["lae"]),
                "loss_sindy": float(loss_stats["lsindy"]),
                "loss_vel": float(loss_stats["lvel"]),
                "loss_l2": float(loss_stats["l2"]),
            }
        )
        print(
            f"[gLaSDI] v={v} |Dv|={len(sampled)} emax_est={emax_est:.3e} "
            f"(k*={k_star:.3e}, b*={b_star:.3e}) "
            f"L={loss_stats['loss']:.3e}"
        )

        # (F) two-level random-subset policy (Algorithm 2, line 14-16)
        if args.tol > 0.0 and emax_est <= args.tol and w_level < 2:
            Nsubset = min(len(Dh_indices), max(1, 2 * max(1, Nsubset)))
            w_level += 1
            print(f"[gLaSDI] tol reached -> increase subset: Nsubset={Nsubset}, w_level={w_level}")

        # (G) termination criteria (Algorithm 1)
        if args.tol > 0.0 and emax_est <= args.tol and w_level == 2:
            print("[gLaSDI] terminate: reached tol at level 2")
            break
        if len(sampled) > args.nmax:
            print("[gLaSDI] terminate: exceeded Nmax")
            break

        v += 1

    if model is None or mean is None or std is None or params_train is None or coeffs_train is None or x_ref is None or t_ref is None or ae_cfg is None:
        raise RuntimeError("gLaSDI training did not produce a model")

    # final save
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "ae.pt")
    np.savez(model_dir / "normalization.npz", mean=mean, std=std)
    np.save(model_dir / "params.npy", params_train)
    np.save(model_dir / "sindy_coeffs.npy", coeffs_train)
    np.savez(model_dir / "grid.npz", x=x_ref, t=t_ref)
    with (model_dir / "sindy_terms.json").open("w", encoding="utf-8") as handle:
        json.dump(term_names(terms), handle, indent=2)

    # store which Dh indices were sampled and training metadata
    meta = {
        "mode": "gLaSDI-greedy",
        "sampled_indices": sampled,
        "knn": int(args.knn),
        "use_mahalanobis": bool(not args.no_mahalanobis),
        "nts_frac": float(args.nts_frac),
        "tol": float(args.tol),
        "nmax": int(args.nmax),
        "nup": int(args.nup),
        "nepoch": int(args.nepoch),
        "nsubset_init": int(args.nsubset),
        "nsubset_final": int(Nsubset),
        "w_level_final": int(w_level),
        "beta1": float(args.beta1),
        "beta2": float(args.beta2),
        "beta3": float(args.beta3),
        "beta4": float(args.beta4),
        "seed": int(args.seed),
        "init": args.init,
        "drop_endpoint": bool(args.drop_endpoint),
        "time_stride": int(args.time_stride),
        "ae": {
            "input_dim": int(ae_cfg.input_dim),
            "latent_dim": int(ae_cfg.latent_dim),
            "hidden_sizes": list(ae_cfg.hidden_sizes),
            "activation": ae_cfg.activation,
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
        },
        "sindy": {
            "degree": int(args.sindy_degree),
            "threshold": float(args.sindy_threshold),
            "max_iter": int(args.sindy_max_iter),
        },
        "history": history,
    }
    with (model_dir / "gmeta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"[gLaSDI] saved model to {model_dir}")


# -----------------------
# Baseline train/predict (kept)
# -----------------------

def save_model_baseline(
    model_dir: Path,
    model: AutoEncoder,
    ae_cfg: AEConfig,
    train_cfg: TrainConfig,
    mean: np.ndarray,
    std: np.ndarray,
    params: np.ndarray,
    coeffs: np.ndarray,
    terms,
    x: np.ndarray,
    t: np.ndarray,
):
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "ae.pt")
    np.savez(model_dir / "normalization.npz", mean=mean, std=std)
    np.save(model_dir / "params.npy", params)
    np.save(model_dir / "sindy_coeffs.npy", coeffs)
    np.savez(model_dir / "grid.npz", x=x, t=t)
    with (model_dir / "sindy_terms.json").open("w", encoding="utf-8") as handle:
        json.dump(term_names(terms), handle, indent=2)

    config = {
        "input_dim": int(ae_cfg.input_dim),
        "latent_dim": int(ae_cfg.latent_dim),
        "hidden_sizes": list(ae_cfg.hidden_sizes),
        "activation": ae_cfg.activation,
        "epochs": int(train_cfg.epochs),
        "batch_size": int(train_cfg.batch_size),
        "lr": float(train_cfg.lr),
        "time_stride": int(train_cfg.time_stride),
        "drop_endpoint": bool(train_cfg.drop_endpoint),
        "sindy_degree": int(train_cfg.sindy_degree),
        "sindy_threshold": float(train_cfg.sindy_threshold),
        "sindy_max_iter": int(train_cfg.sindy_max_iter),
    }
    with (model_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def load_model(model_dir: Path, device: torch.device):  # type: ignore
    require_torch()
    # baseline config.json OR gmeta.json (gLaSDI)
    if (model_dir / "config.json").exists():
        with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        ae_cfg = AEConfig(
            input_dim=cfg["input_dim"],
            latent_dim=cfg["latent_dim"],
            hidden_sizes=cfg["hidden_sizes"],
            activation=cfg["activation"],
        )
        drop_endpoint = bool(cfg["drop_endpoint"])
        sindy_degree = int(cfg["sindy_degree"])
        extra = {"drop_endpoint": drop_endpoint, "sindy_degree": sindy_degree}
    else:
        with (model_dir / "gmeta.json").open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        ae_cfg = AEConfig(
            input_dim=meta["ae"]["input_dim"],
            latent_dim=meta["ae"]["latent_dim"],
            hidden_sizes=meta["ae"]["hidden_sizes"],
            activation=meta["ae"]["activation"],
        )
        extra = {"drop_endpoint": bool(meta["drop_endpoint"]), "sindy_degree": int(meta["sindy"]["degree"])}

    model = AutoEncoder(ae_cfg).to(device)
    model.load_state_dict(torch.load(model_dir / "ae.pt", map_location=device))
    model.eval()

    norm = np.load(model_dir / "normalization.npz")
    mean = norm["mean"]
    std = norm["std"]
    params = np.load(model_dir / "params.npy")
    coeffs = np.load(model_dir / "sindy_coeffs.npy")
    grid = np.load(model_dir / "grid.npz")
    x = grid["x"]
    t = grid["t"]
    with (model_dir / "sindy_terms.json").open("r", encoding="utf-8") as handle:
        term_labels = json.load(handle)

    terms = build_terms(ae_cfg.latent_dim, extra["sindy_degree"])
    if term_names(terms) != term_labels:
        raise RuntimeError("Stored SINDy terms do not match generated terms")

    return model, mean, std, params, coeffs, terms, x, t, extra


def train_pipeline(args: argparse.Namespace) -> None:
    """
    Baseline LaSDI training (fixed cases, no greedy sampling).
    """
    require_torch()
    dataset_dir = Path(args.dataset_dir)
    model_dir = Path(args.model_dir)
    rows = load_index(dataset_dir)
    learning_rows = select_corner_center_rows(rows)
    if not learning_rows:
        raise RuntimeError("Dataset index is empty")

    print(f"Using {len(learning_rows)} training cases (4 corners + center)")
    for _, a, w in learning_rows:
        print(f"  selected (a={a:.6f}, w={w:.6f})")

    params = np.asarray([[a, w] for _, a, w in learning_rows], dtype=float)

    snapshots = []
    x_ref = None
    t_ref = None
    for filename, a, w in learning_rows:
        x, t, u = load_series(dataset_dir, filename, args.drop_endpoint, args.time_stride)
        if x_ref is None:
            x_ref = x
            t_ref = t
        snapshots.append(u)
    u_all = np.vstack(snapshots)

    dt = float(t_ref[1] - t_ref[0])

    u_norm, mean, std = normalize_data(u_all)

    ae_cfg = AEConfig(
        input_dim=u_norm.shape[1],
        latent_dim=args.latent_dim,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
    )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        time_stride=args.time_stride,
        drop_endpoint=args.drop_endpoint,
        sindy_degree=args.sindy_degree,
        sindy_threshold=args.sindy_threshold,
        sindy_max_iter=args.sindy_max_iter,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model = train_autoencoder(u_norm, ae_cfg, args.epochs, args.batch_size, args.lr, device, model=None)
    print("device:", device)

    coeffs_list = []
    for filename, a, w in learning_rows:
        _, t_series, u_series = load_series(dataset_dir, filename, args.drop_endpoint, args.time_stride)
        z = encode_series(model, u_series, mean, std, device, args.batch_size)
        xi, terms = fit_sindy(z, dt, args.sindy_degree, args.sindy_threshold, args.sindy_max_iter)
        coeffs_list.append(xi)

    coeffs = np.stack(coeffs_list, axis=0)
    save_model_baseline(model_dir, model, ae_cfg, train_cfg, mean, std, params, coeffs, terms, x_ref, t_ref)
    print(f"Saved model to {model_dir}")


def predict_pipeline(args: argparse.Namespace) -> None:
    """
    Works for both baseline and gLaSDI models (loads ae/normalization/params/coeffs/terms).
    """
    require_torch()
    model_dir = Path(args.model_dir)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model, mean, std, params, coeffs, terms, x, t, extra = load_model(model_dir, device)

    a = float(args.a)
    w = float(args.w)
    drop_endpoint = bool(extra["drop_endpoint"])

    u_hat = rom_predict_u(
        model=model,
        mean=mean,
        std=std,
        params_train=params,
        coeffs_train=coeffs,
        terms=terms,
        x=x,
        t=t,
        a=a,
        w=w,
        device=device,
        batch_size=args.batch_size,
        k=args.knn,
        drop_endpoint=drop_endpoint,
        use_mahalanobis=not args.no_mahalanobis,
    )

    out = Path(args.output)
    np.savez(out, a=a, w=w, x=x, t=t, u=u_hat)
    print(f"Saved prediction to {out}")


# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaSDI / gLaSDI for Burgers.")
    sub = parser.add_subparsers(dest="command", required=True)

    # baseline train
    train = sub.add_parser("train", help="Baseline LaSDI: Train AE + local SINDy on fixed dataset")
    train.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    train.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    train.add_argument("--latent-dim", type=int, default=5)
    train.add_argument("--hidden-sizes", type=lambda s: [int(v) for v in s.split(",") if v], default=[100])
    train.add_argument("--activation", type=str, default="sigmoid")
    train.add_argument("--epochs", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--lr", type=float, default=1.0e-3)
    train.add_argument("--time-stride", type=int, default=1)
    train.add_argument("--drop-endpoint", action="store_true")
    train.add_argument("--sindy-degree", type=int, default=1)
    train.add_argument("--sindy-threshold", type=float, default=0.05)
    train.add_argument("--sindy-max-iter", type=int, default=10)
    train.add_argument("--cpu", action="store_true")

    # gLaSDI greedy training
    gtrain = sub.add_parser("gtrain", help="gLaSDI: Greedy sampling training (Algorithm 1-2)")
    gtrain.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    gtrain.add_argument("--model-dir", default=str(DEFAULT_GMODEL_DIR))
    gtrain.add_argument("--latent-dim", type=int, default=5)
    gtrain.add_argument("--hidden-sizes", type=lambda s: [int(v) for v in s.split(",") if v], default=[100])
    gtrain.add_argument("--activation", type=str, default="sigmoid")
    gtrain.add_argument("--batch-size", type=int, default=512)
    gtrain.add_argument("--lr", type=float, default=1.0e-3)
    gtrain.add_argument("--time-stride", type=int, default=1)
    gtrain.add_argument("--drop-endpoint", action="store_true")
    gtrain.add_argument("--sindy-degree", type=int, default=1)
    gtrain.add_argument("--sindy-threshold", type=float, default=0.05)
    gtrain.add_argument("--sindy-max-iter", type=int, default=10)
    gtrain.add_argument("--beta1", type=float, default=1.0, help="Weight for AE loss")
    gtrain.add_argument("--beta2", type=float, default=0.1, help="Weight for SINDy loss")
    gtrain.add_argument("--beta3", type=float, default=0.1, help="Weight for velocity loss")
    gtrain.add_argument("--beta4", type=float, default=0.0, help="Weight for Xi L2 regularization")

    # Algorithm 1-2 parameters
    gtrain.add_argument("--nepoch", type=int, default=50000, help="Maximum training epochs")
    gtrain.add_argument("--nup", type=int, default=2000, help="Greedy sampling frequency (epochs)")
    gtrain.add_argument("--knn", type=int, default=1, help="k for kNN convex interpolation during gtrain")
    gtrain.add_argument("--nsubset", type=int, default=30, help="Initial random subset size for greedy evaluation")
    gtrain.add_argument("--tol", type=float, default=5.0e-2, help="Target tolerance for estimated max relative error")
    gtrain.add_argument("--nmax", type=int, default=25, help="Max number of sampled parameters")
    gtrain.add_argument("--nts-frac", type=float, default=0.1, help="Nts/Nt fraction for residual indicator")
    gtrain.add_argument("--seed", type=int, default=0)
    gtrain.add_argument("--init", type=str, choices=["corners", "corner_center", "random"], default="corners")
    gtrain.add_argument("--n0", type=int, default=4, help="Only used for init=random")
    gtrain.add_argument("--no-mahalanobis", action="store_true", help="Use Euclidean distance instead of Mahalanobis")
    gtrain.add_argument("--cpu", action="store_true")

    # prediction
    predict = sub.add_parser("predict", help="Predict for a new (a,w) using saved model (baseline or gLaSDI)")
    predict.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    predict.add_argument("--a", type=float, required=True)
    predict.add_argument("--w", type=float, required=True)
    predict.add_argument("--knn", type=int, default=3)
    predict.add_argument("--no-mahalanobis", action="store_true")
    predict.add_argument("--output", type=str, default="prediction.npz")
    predict.add_argument("--batch-size", type=int, default=512)
    predict.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_pipeline(args)
    elif args.command == "gtrain":
        gtrain_pipeline(args)
    elif args.command == "predict":
        predict_pipeline(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
