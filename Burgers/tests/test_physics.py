from __future__ import annotations

import numpy as np

from burgers_simulation import SimConfig, simulate


def _mass(u: np.ndarray, dx: float, include_endpoint: bool) -> float:
    if include_endpoint:
        return float(np.sum(u[:-1]) * dx)
    return float(np.sum(u) * dx)


def test_mass_conservation_periodic():
    cfg = SimConfig(t_end=0.2, a=0.8, w=1.0, include_endpoint=True)
    x, u0, u = simulate(cfg)
    dx = cfg.dx
    mass0 = _mass(u0, dx, cfg.include_endpoint)
    mass1 = _mass(u, dx, cfg.include_endpoint)
    assert np.isclose(mass1, mass0, rtol=1e-10, atol=1e-12)


def test_periodic_endpoint_matches():
    cfg = SimConfig(t_end=0.05, a=0.85, w=1.0, include_endpoint=True)
    _, _, u = simulate(cfg)
    assert np.isclose(u[0], u[-1])


def test_rest_state_remains_zero():
    cfg = SimConfig(t_end=0.1, a=0.0, w=1.0, include_endpoint=False)
    _, _, u = simulate(cfg)
    assert np.allclose(u, 0.0)
