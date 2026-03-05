import importlib.util
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "Build_vlasov_data.py"
SPEC = importlib.util.spec_from_file_location("Build_vlasov_data", MODULE_PATH)
bvd = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(bvd)


def test_shift_x_semi_lagrangian_identity_for_zero_dt():
    rng = np.random.default_rng(0)
    f_in = rng.normal(size=(bvd.N, bvd.v.size))
    shifted = bvd.shift_x_semi_lagrangian(f_in, bvd.v, 0.0)
    np.testing.assert_allclose(shifted, f_in, atol=1e-12, rtol=1e-12)


def test_poisson_E_caluc_neutral_density_gives_zero_field():
    # Constant f over v such that trapz(f dv) == 1 everywhere in x.
    c = 1.0 / (bvd.v[-1] - bvd.v[0])
    f_in = np.full((bvd.N, bvd.v.size), c)

    E, dn, dn_hat = bvd.poisson_E_caluc(f_in)

    np.testing.assert_allclose(dn, 0.0, atol=1e-12)
    np.testing.assert_allclose(E, 0.0, atol=1e-12)
    np.testing.assert_allclose(dn_hat, 0.0, atol=1e-12)


def test_shift_v_lagrangian_zero_shift_identity_and_out_of_range_clamps_to_zero():
    rng = np.random.default_rng(1)
    f_in = rng.random(size=(bvd.N, bvd.v.size))

    no_shift = bvd.shift_v_lagrangian(f_in, np.zeros(bvd.N), dt_full=1.0)
    np.testing.assert_allclose(no_shift, f_in, atol=1e-10, rtol=1e-10)

    huge_E = np.full(bvd.N, 1e6)
    full_out = bvd.shift_v_lagrangian(f_in, huge_E, dt_full=1.0)
    np.testing.assert_allclose(full_out, 0.0, atol=1e-12)


def test_density_velocity_pressure_and_dq_dx_consistency():
    # Symmetric in v and constant in x -> u=0 and dq/dx=0.
    profile_v = np.exp(-(bvd.v**2))
    f = np.tile(profile_v, (bvd.N, 1))

    n = bvd.density(f)
    u = bvd.velocity(f)
    p = bvd.pressure(f)
    dqdx = bvd.dq_dx(f)

    assert n.shape == (bvd.N,)
    np.testing.assert_allclose(u, 0.0, atol=1e-12)
    expected_p = np.sum(f * (bvd.v[None, :] ** 2), axis=1) * bvd.dv
    np.testing.assert_allclose(p, expected_p, atol=1e-12)
    np.testing.assert_allclose(dqdx, 0.0, atol=1e-12)



def test_velocity_handles_zero_density_without_inf_or_nan():
    f = np.zeros((bvd.N, bvd.v.size))
    u = bvd.velocity(f)
    assert np.all(np.isfinite(u))
    np.testing.assert_allclose(u, 0.0, atol=0.0)


def test_initial_f_twostream_shape_positive_and_eps_zero_x_independent():
    f = bvd.initial_f_twostream(bvd.x, bvd.v, T=1.0, k=1.1, eps=0.0, vd=2.0)

    assert f.shape == (bvd.N, bvd.v.size)
    assert np.all(f > 0.0)
    np.testing.assert_allclose(f, np.tile(f[0, :], (bvd.N, 1)), atol=1e-12)


def test_run_vlasov_case_twostream_writes_expected_npz_files(tmp_path, monkeypatch):
    # Keep runtime small: one recorded time sample.
    monkeypatch.setattr(bvd, "tmax", 0.0)

    out_dir = tmp_path / "case"
    bvd.run_vlasov_case_twostream(T=1.0, k=1.0, save_dir=str(out_dir))

    moments_path = out_dir / "moments.npz"
    init_info_path = out_dir / "init_info.npz"
    animation_path = out_dir / "animation_data.npz"
    full_dist_path = out_dir / "distribution_full.npz"

    assert moments_path.exists()
    assert init_info_path.exists()
    assert animation_path.exists()
    assert full_dist_path.exists()

    moments = np.load(moments_path)
    for key in ["t", "n", "u", "p", "dq_dx"]:
        assert key in moments
    assert moments["t"].shape == (1,)
    assert moments["n"].shape[1] == bvd.N

    init_info = np.load(init_info_path)
    for key in ["T", "k", "x", "v", "dt", "tmax", "eps", "vd", "x_domain", "Vmax", "N", "M"]:
        assert key in init_info

    anim = np.load(animation_path)
    for key in ["t", "f", "E", "x", "v", "frame_stride", "dt", "tmax"]:
        assert key in anim
    assert anim["t"].shape == (1,)
    assert anim["f"].shape == (1, bvd.N, bvd.v.size)
    assert anim["E"].shape == (1, bvd.N)

    full_dist = np.load(full_dist_path)
    for key in ["t", "f", "x", "v", "T", "k", "dt", "tmax"]:
        assert key in full_dist
    assert full_dist["t"].shape == (1,)
    assert full_dist["f"].shape == (1, bvd.N, bvd.v.size)


def test_generate_param_grid_calls_runner_for_all_parameter_pairs(tmp_path, monkeypatch):
    calls = []

    def fake_runner(T, k, save_dir):
        calls.append((T, k, save_dir))

    monkeypatch.setattr(bvd, "run_vlasov_case_twostream", fake_runner)
    monkeypatch.setattr(bvd, "T_min", 0.9)
    monkeypatch.setattr(bvd, "T_max", 0.91)
    monkeypatch.setattr(bvd, "dT", 0.01)
    monkeypatch.setattr(bvd, "k_min", 1.0)
    monkeypatch.setattr(bvd, "k_max", 1.01)
    monkeypatch.setattr(bvd, "dk", 0.01)

    bvd.generate_param_grid(str(tmp_path / "grid"))

    assert len(calls) == 4
    got_pairs = {(round(T, 2), round(k, 2)) for T, k, _ in calls}
    assert got_pairs == {(0.9, 1.0), (0.9, 1.01), (0.91, 1.0), (0.91, 1.01)}

    # Save-dir naming should include rounded T, k with 2 decimals.
    for T, k, save_dir in calls:
        assert f"T_{T:.2f}_k_{k:.2f}" in save_dir
