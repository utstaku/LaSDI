import numpy as np
import os
from scipy.interpolate import CubicSpline

# =========================
# Config
# =========================
# grid
N = 128         # x grid
M = 64         # v grid half (total 2M points)
Vmax = 7.0
dt = 5e-3
tmax = 5.0

# two-stream parameter grid (paper-style)
T_min, T_max, dT = 0.9, 1.1, 0.01
k_min, k_max, dk = 1.0, 1.2, 0.01

# domain for Eq.(39): cos(k*pi*x) -> natural to use x in [-1, 1)
x_left, x_right = 0, 2.0*np.pi

save_root = "vlasov_twostream_param_grid"

# animation output
save_animation_data = True
animation_stride = 10  # save every Nth time step
animation_dtype = np.float32

# ML output (full state history)
save_ml_distribution_data = True
ml_distribution_dtype = np.float32

# =========================
# Grid setup
# =========================
# v grid (total 2M points)
"""
dv = 2 * Vmax / (2 * M - 1)
v_index = np.arange(-M, M, 1, dtype=float)
v = v_index * dv
"""
v = np.linspace(-Vmax, Vmax, 2*M, endpoint=True)
dv = v[1] - v[0]
# x grid (periodic)
L = x_right - x_left
dx = L / N
x = x_left + np.arange(N) * dx

# Fourier wavenumbers for periodic x
k_vec = 2 * np.pi * np.fft.fftfreq(N, d=dx)

# =========================
# Semi-Lagrangian split scheme (same as yours)
# =========================
def shift_x_semi_lagrangian(f_in, v_arr, dt_half):
    # f_in: shape (Nx, Nv)
    F = np.fft.fft(f_in, axis=0)
    phase = np.exp(-1j * k_vec[:, None] * v_arr[None, :] * dt_half)
    F_shift = F * phase
    return np.fft.ifft(F_shift, axis=0).real

def poisson_E_caluc(f_in):
    # n(x) = \int f dv (trapz)
    n = np.trapezoid(f_in, v, axis=1)
    # Poisson RHS: dn = 1 - n (neutralizing ion background of density 1)
    dn = 1.0 - n
    dn_hat = np.fft.fft(dn)
    E_hat = np.zeros_like(dn_hat, dtype=complex)
    for i, kk in enumerate(k_vec):
        if kk != 0:
            E_hat[i] = dn_hat[i] / (1j * kk)
        else:
            E_hat[i] = 0.0
    E = np.fft.ifft(E_hat).real
    return E, dn, dn_hat

def shift_v_lagrangian(f_in, E_x, dt_full):
    # v-shift = E(x) dt
    v_shift = E_x * dt_full
    f_out = np.zeros_like(f_in)

    vmin = v[0]
    vmax_eff = v[-1] + (v[1] - v[0])  # effective upper bound

    for ix in range(f_in.shape[0]):
        cs = CubicSpline(v, f_in[ix, :], bc_type="natural", extrapolate=False)
        vv = v + v_shift[ix]
        fout = cs(vv)

        # NaN/inf -> 0
        fout[~np.isfinite(fout)] = 0.0
        # out of range -> 0
        mask = (vv < vmin) | (vv > vmax_eff)
        fout[mask] = 0.0

        f_out[ix, :] = fout
    return f_out

# =========================
# Moments (same intent as yours)
# =========================
def density(f):
    return np.sum(f, axis=1) * dv

def velocity(f):
    n = density(f)
    j1 = np.sum(f * v[None, :], axis=1) * dv
    # avoid 0-division (shouldn't happen if f>=0)
    return j1 / np.maximum(n, 1e-30)

def pressure(f):
    u = velocity(f)
    vc = v[None, :] - u[:, None]
    return np.sum(f * (vc ** 2), axis=1) * dv

def dq_dx(f):
    u = velocity(f)
    vc = v[None, :] - u[:, None]
    q = np.sum(f * (vc ** 3), axis=1) * dv
    q_hat = np.fft.fft(q)
    return np.fft.ifft(1j * k_vec * q_hat).real

# =========================
# Paper-style initial condition: two-stream (Eq.(39) in the screenshot)
# =========================
def initial_f_twostream(x_arr, v_arr, T, k, eps=0.1, vd=2.0):
    """
    f(0,x,v) = (4/(pi T)) * (1 + eps*cos(k*pi*x)) *
               ( exp(-(v-vd)^2/(2T)) + exp(-(v+vd)^2/(2T)) )
    """
    X = x_arr[:, None]
    V = v_arr[None, :]

    spatial = 1.0 + eps * np.cos(k * X)
    streams = np.exp(-((V - vd) ** 2) / (2.0 * T)) + np.exp(-((V + vd) ** 2) / (2.0 * T))
    #f = (4.0 / (np.pi * T)) * spatial * streams
    f = (8.0 / (np.sqrt(2.0*np.pi*T))) * spatial * streams
    return f

# =========================
# One case runner
# =========================
def run_vlasov_case_twostream(
    T,
    k,
    save_dir,
    save_animation=save_animation_data,
    frame_stride=animation_stride,
    save_ml_distribution=save_ml_distribution_data,
):
    os.makedirs(save_dir, exist_ok=True)
    frame_stride = max(1, int(frame_stride))

    # init f
    f = initial_f_twostream(x, v, T=T, k=k, eps=0.1, vd=2.0)

    # time loop
    t_list, n_list, u_list, p_list, dqdx_list = [], [], [], [], []
    t_anim_list, f_anim_list, E_anim_list = [], [], []
    f_ml_list = []

    t = 0.0
    step = 0
    while t < tmax + 1e-12:
        n = density(f)
        u = velocity(f)
        p = pressure(f)
        dqdx = dq_dx(f)

        t_list.append(t)
        n_list.append(n)
        u_list.append(u)
        p_list.append(p)
        dqdx_list.append(dqdx)
        if save_ml_distribution:
            f_ml_list.append(f.astype(ml_distribution_dtype, copy=True))

        if save_animation and (step % frame_stride == 0):
            E_now, _, _ = poisson_E_caluc(f)
            t_anim_list.append(t)
            f_anim_list.append(f.astype(animation_dtype, copy=True))
            E_anim_list.append(E_now.astype(animation_dtype, copy=True))

        # split scheme (same as your code)
        f = shift_x_semi_lagrangian(f, v, dt * 0.5)
        E_half, _, _ = poisson_E_caluc(f)
        f = shift_v_lagrangian(f, E_half, dt)
        f = shift_x_semi_lagrangian(f, v, dt * 0.5)

        t += dt
        step += 1

    # save
    np.savez(
        os.path.join(save_dir, "moments.npz"),
        t=np.array(t_list),
        n=np.array(n_list),
        u=np.array(u_list),
        p=np.array(p_list),
        dq_dx=np.array(dqdx_list),
    )

    np.savez(
        os.path.join(save_dir, "init_info.npz"),
        T=T,
        k=k,
        x=x,
        v=v,
        dt=dt,
        tmax=tmax,
        eps=0.1,
        vd=2.0,
        x_domain=np.array([x_left, x_right]),
        Vmax=Vmax,
        N=N,
        M=M,
    )

    if save_animation:
        np.savez_compressed(
            os.path.join(save_dir, "animation_data.npz"),
            t=np.array(t_anim_list),
            f=np.array(f_anim_list, dtype=animation_dtype),
            E=np.array(E_anim_list, dtype=animation_dtype),
            x=x.astype(animation_dtype),
            v=v.astype(animation_dtype),
            frame_stride=frame_stride,
            dt=dt,
            tmax=tmax,
        )

    if save_ml_distribution:
        np.savez_compressed(
            os.path.join(save_dir, "distribution_full.npz"),
            t=np.array(t_list),
            f=np.array(f_ml_list, dtype=ml_distribution_dtype),
            x=x.astype(ml_distribution_dtype),
            v=v.astype(ml_distribution_dtype),
            T=T,
            k=k,
            dt=dt,
            tmax=tmax,
        )

    print(f"[OK] T={T:.2f}, k={k:.2f} saved to {save_dir}  (Nt={len(t_list)}, Nframes={len(t_anim_list)})")

# =========================
# Parameter grid generator
# =========================
def generate_param_grid(out_root):
    os.makedirs(out_root, exist_ok=True)

    T_list = np.round(np.arange(T_min, T_max + 1e-12, dT), 2)
    k_list = np.round(np.arange(k_min, k_max + 1e-12, dk), 2)

    case_id = 0
    for T in T_list:
        for k in k_list:
            save_dir = os.path.join(out_root, f"T_{T:.2f}_k_{k:.2f}")
            run_vlasov_case_twostream(T, k, save_dir)
            case_id += 1

    print(f"==== All datasets generated: {case_id} cases ====")

if __name__ == "__main__":
    generate_param_grid(save_root)
