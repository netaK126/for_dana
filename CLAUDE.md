# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

VHAGaR (Verifier of Hazardous Attacks for Global Robustness) is a neural network verification research framework. It computes minimal globally robust bounds using dual-network MIP (Mixed Integer Programming) encoding with Gurobi. The **primary, most-maintained implementation is `vaghar_org/`** — start there. See `vaghar_org/CLAUDE.md` for detailed architecture, execution flow, and internal state.

## Codebase Layout

| Directory | Purpose |
|---|---|
| `vaghar_org/` | **Main implementation** — dual-network MIP with transfer mode and perturbation intervals |
| `base_delta_diff_calculatio_fixingDeps/` | Delta-diff variant with dependency fixes |
| `delta_diff_base/` | Base delta-difference calculation |
| `lucid_org/` | LUCID-based verification variant |
| `lucid_delta_diff_with_perturbation/` | LUCID + delta-diff with perturbations |
| `lucid_for_vaghar_intervals/` | LUCID adapted for interval-based bounds |
| `vaghar_as_should_be_originally_no_c_target/` | Original VHAGaR without c_target logic |
| `code_deprecated_active_just_for_models/` | Deprecated code kept for model artifacts |
| `MNIST/` | Shared dataset directory |
| `gurobi.lic` | Gurobi license file (required at runtime) |

All variants follow the same structure: `run.jl` entry point + `utils/` containing vendored `MIPVerify.jl`, perturbation models, MIP config, models, and hyper-attack (Python PGD).

## Prerequisites

- **Julia >= 1.6** with packages: JuMP, Gurobi, MathOptInterface, IntervalArithmetic, CSV, DataFrames, PyCall
- **Gurobi** optimizer with valid `gurobi.lic` (referenced at repo root)
- **Python 3.8** with PyTorch, torchvision, numpy (for PGD attacks and model training)
- PyCall must point to the correct Python: `ENV["PYTHON"]="/usr/bin/python3.8"`

## Running

### Standard Mode (single network verification)

```bash
cd vaghar_org/
julia run.jl \
  --dataset mnist \
  --model_name 4x10 \
  --model_path ./models/model.p \
  --perturbation linf \
  --perturbation_size 0.02 \
  --ctag 1 \
  --ct "2,3,4,5,6,7,8,9" \
  --timout 4000 \
  --output_dir ./results/ \
  --name_to_save itr17 \
  --use_hyper_attack true \
  --activate_vaghgar_deps false \
  --use_perturbed_intervals false
```

### Transfer Mode (compare robustness of two networks)

```bash
cd vaghar_org/
julia run.jl \
  --mode transfer \
  --dataset mnist \
  --model_name 4x10 \
  --model_path ./models/n1_model.p \
  --model_path2 ./models/n2_model.p \
  --vaghar_results ./results/n1_vaghar_results.txt \
  --perturbation linf \
  --perturbation_size 0.02 \
  --ctag 1 \
  --ct "2,3,4,5,6,7,8,9" \
  --timout 4000 \
  --c_tag_mode false \
  --use_intervals false \
  --use_perturbed_intervals false
```

### Training a Model

```bash
cd vaghar_org/
python utils/train.py  # Trains on MNIST, saves .p (pickle) + .pth (PyTorch checkpoint)
```

### Combined Train + Verify Pipeline

```bash
cd vaghar_org/
python utils/train_and_run.py  # Orchestrates training then calls julia run.jl as subprocess
```

## Key Differences Between Variants

- **Standard mode** (`vaghar_org/`): verifies a single network N1 — finds minimum δ such that ∃ perturbed input causing misclassification
- **Transfer mode** (`vaghar_org/`, `--mode transfer`): encodes four networks (N1 original, N1 perturbed, N2 original, N2 perturbed) to compare robustness between two models. Reads `delta_1` from prior VHAGaR results via `get_delta1_vaghar()`
- **Delta-diff** (`delta_diff_base/`, `base_delta_diff_calculatio_fixingDeps/`): computes the difference in δ values between two models
- **LUCID variants**: use the LUCID algorithm for bound computation instead of the standard LP/MIP tightening

## Shared Conventions Across All Variants

- Julia class indices are 1-indexed; Python (hyper_attack) uses 0-indexed — conversions via `c_tag-1`
- Models stored as `.p` (pickle) files; loaded via `get_nn()` in `utils/models.jl`
- Results saved as CSV: `source,target,incumbent_obj,best_bound,solve_time`
- Layer type checks use `occursin("ReLU", string(typeof(l)))` pattern throughout
- Perturbation sizes are comma-separated strings parsed by `parse_numbers_to_Float64()`
- `network_version` global (`"org"` or `"perturbation"`) must be set before encoding each network copy
- Must call `mip_reset()` between class-pair iterations to clear global state

## Inter-Process Communication (Julia ↔ Python)

The hyper-attack PGD warmstart uses `/tmp/` files for data exchange:
- Julia calls Python `hyper_attack.py` via subprocess
- Python writes feasible solutions to `/tmp/booleans_<c_tag>_<c_target>_<token>.txt`, `/tmp/strings_...`, `/tmp/best_val_...`
- Julia reads these files and applies warm-start hints to the MIP via `hyper_attack_hints()`
- A `/tmp/fail_...` file indicates the PGD attack failed to find an adversarial example

## Supported Model Architectures

Defined in `utils/models.py` and loaded in `utils/models.jl`:
- Fully-connected: `3x10`, `3x50`, `4x10`, `5x10`, `5x50`, `10x10`, `3x100`
- Convolutional: `cnn0`, `cnn1`, `cnn2`, `cnn3`

## Supported Perturbation Types

Each has a dedicated `get_perturbation_specific_keys_*()` in `utils/perturbation_models.jl`:
`linf`, `brightness`, `contrast`, `patch`, `occ` (occlusion), `translation`, `rotation`
