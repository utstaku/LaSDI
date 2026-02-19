from __future__ import annotations

import numpy as np

from burgers_simulation import (
    SimConfig,
    append_endpoint,
    backward_euler_step,
    flux_derivative,
    initial_condition,
    make_grid,
)


def test_make_grid_default_size_and_bounds():
    cfg = SimConfig()
    x = make_grid(cfg)
    assert len(x) == 1000
    assert np.isclose(x[0], cfg.x_min)
    assert np.isclose(x[-1], cfg.x_max - cfg.dx)


def test_append_endpoint_periodic_value():
    x = np.array([0.0, 1.0])
    u = np.array([2.0, 3.0])
    x_out, u_out = append_endpoint(x, u, 2.0)
    assert np.allclose(x_out, [0.0, 1.0, 2.0])
    assert np.allclose(u_out, [2.0, 3.0, 2.0])


def test_initial_condition_gaussian_matches_formula():
    x = np.array([-1.0, 0.0, 1.0])
    a = 0.9
    w = 1.1
    expected = a * np.exp(-(x * x) / (2.0 * w * w))
    u = initial_condition(x, a, w)
    assert np.allclose(u, expected)


def test_flux_derivative_constant_zero():
    u = np.ones(8) * 1.7
    dudx = flux_derivative(u, 0.1)
    assert np.allclose(dudx, 0.0)


def test_backward_euler_step_preserves_constant_state():
    u = np.ones(16) * 2.5
    u_next, iters, converged = backward_euler_step(u, 0.1, 1.0e-3, 10, 1.0e-12)
    assert converged
    assert iters == 1
    assert np.allclose(u_next, u)
