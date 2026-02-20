# LaSDI Burgers Baseline

Baseline tools to reproduce the 1D inviscid Burgers dataset and a minimal LaSDI pipeline (auto‑encoder + SINDy + kNN interpolation) used for reduced‑order modeling experiments.

**Key components**
- `Burgers/burgers_simulation.py` generates full‑order data for 1D inviscid Burgers on `[-3, 3]` with periodic BCs and Gaussian ICs.
- `Burgers/lasdi.py` trains a baseline LaSDI model and predicts new parameters.
- `Burgers/tests` contains unit and physics tests for the solver plus dataset utilities.
- `paper/main.tex` contains the reference description used to set the numerical defaults.

## Requirements
- Python 3.10+
- `numpy` (required)
- `pytest` (for tests)
- `matplotlib` (optional, only for plotting)
- `torch` (required for `Burgers/lasdi.py`)

## Quickstart

**1) Generate dataset**
```bash
python /Users/abetakuro/work/LaSDI/Burgers/burgers_simulation.py \
  --dataset-dir /Users/abetakuro/work/LaSDI/Burgers/dataset \
  --a-min 0.7 --a-max 0.9 --a-step 0.01 \
  --w-min 0.9 --w-max 1.1 --w-step 0.01
```
This writes one `.npz` per `(a, w)` plus `dataset_meta.npz` and `index.csv`.

**2) Train baseline LaSDI**
```bash
python /Users/abetakuro/work/LaSDI/Burgers/lasdi.py train \
  --dataset-dir /Users/abetakuro/work/LaSDI/Burgers/dataset \
  --model-dir /Users/abetakuro/work/LaSDI/Burgers/lasdi_model \
  --latent-dim 5 \
  --hidden-sizes 100 \
  --epochs 5000 \
  --sindy-degree 1
```
This trains an MLP auto‑encoder and SINDy coefficients per parameter.

**3) Predict a new parameter**
```bash
python /Users/abetakuro/work/LaSDI/Burgers/lasdi.py predict \
  --model-dir /Users/abetakuro/work/LaSDI/Burgers/lasdi_model \
  --a 0.8 --w 1.0 \
  --output /Users/abetakuro/work/LaSDI/Burgers/prediction.npz
```

**4) Run tests**
```bash
cd /Users/abetakuro/work/LaSDI/Burgers
pytest -q
```

## Notes
- The default solver settings match the paper: `dx=6e-3`, `dt=1e-3`, `t_end=1.0`, periodic BCs.
- By default, the dataset includes the periodic endpoint, giving 1001 spatial points. Use `--no-endpoint` in `burgers_simulation.py` or `--drop-endpoint` in `lasdi.py train` to remove the duplicate endpoint.
- The full 21×21 parameter grid with full time history is large (several GB). Reduce ranges or increase steps if needed.

## File Formats
**Dataset sample (`a_0.700000_w_0.900000.npz`)**
- `u`: shape `(Nt+1, Nx)` or `(Nt+1, Nx+1)` when endpoint included
- `x`: spatial grid
- `t`: time grid
- `a`, `w`: parameter values

**Model artifacts (`lasdi_model/`)**
- `ae.pt`: trained auto‑encoder
- `normalization.npz`: mean/std used for normalization
- `params.npy`: training parameter list
- `sindy_coeffs.npy`: per‑parameter SINDy coefficients
- `sindy_terms.json`: library term names
- `grid.npz`: `x`, `t`
- `config.json`: training settings
