from __future__ import annotations

import numpy as np

from burgers_simulation import SimConfig, generate_dataset, make_param_values, simulate_full


def test_make_param_values_inclusive():
    vals = make_param_values(0.7, 0.9, 0.1, "a")
    assert np.allclose(vals, [0.7, 0.8, 0.9])


def test_simulate_full_shapes_with_endpoint():
    cfg = SimConfig(x_min=0.0, x_max=2.0, dx=1.0, dt=0.1, t_end=0.2, a=1.0, w=1.0)
    x, t, u_all = simulate_full(cfg)
    assert x.shape == (3,)
    assert t.shape == (3,)
    assert u_all.shape == (3, 3)
    assert np.isclose(u_all[0, 0], u_all[0, -1])


def test_generate_dataset_writes_files(tmp_path):
    cfg = SimConfig(
        x_min=0.0,
        x_max=2.0,
        dx=1.0,
        dt=0.1,
        t_end=0.2,
        a=0.8,
        w=1.0,
        include_endpoint=False,
    )
    a_vals = np.array([0.7, 0.8])
    w_vals = np.array([0.9])
    generate_dataset(cfg, a_vals, w_vals, tmp_path)

    assert (tmp_path / "dataset_meta.npz").exists()
    assert (tmp_path / "index.csv").exists()

    files = sorted(tmp_path.glob("a_*_w_*.npz"))
    assert len(files) == 2

    data = np.load(files[0])
    assert "u" in data
