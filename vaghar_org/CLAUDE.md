# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VHAGaR (Verifier of Hazardous Attacks for Global Robustness) is a neural network verification framework that computes minimal globally robust bounds using dual-network MIP (Mixed Integer Programming) encoding. It encodes two copies of a network (original input and perturbed input) into a single MIP, adds inter-network dependency constraints, and solves with Gurobi to find the minimum perturbation causing misclassification.

## Running

```bash
julia run.jl \
  --dataset mnist \
  --model_name 4x10 \
  --model_path /path/to/model.p \
  --perturbation linf \
  --perturbation_size 0.02 \
  --ctag 1 \
  --ct "2,3,4,5,6,7,8,9" \
  --timout 4000 \
  --output_dir ./results/ \
  --name_to_save itr17
```

Requires: Julia >= 1.6, Gurobi license, Python 3.8 with PyTorch (for hyper-attack).

## Architecture

### Execution Flow

```
run.jl main()
  parse_commandline()
  get_nn()                          # Load pretrained network from pickle
  for each (c_tag, c_target):
    hyper_attack()                  # Python PGD attack → warm-start hints
    get_model()                     # Build dual-network MIP formulation
      get_perturbation_specific_keys_*()
        v_in |> nn                  # Encode original network copy
        v_x0 |> nn                  # Encode perturbed network copy
    perturbation_dependencies()     # Add inter-network constraints via phi_dep
    mip_set_delta_property()        # Objective: max(confidence gap)
    optimize!()                     # Solve with Gurobi
```

### Key Files

- **`run.jl`** — Entry point. Orchestrates the full pipeline.
- **`utils/perturbation_models.jl`** — Creates the dual-network MIP. Each perturbation type (linf, brightness, contrast, occ, patch, translation, rotation) has its own `get_perturbation_specific_keys_*()` function defining how `v_in`, `v_x0`, and `v_e` relate.
- **`utils/perturbation_dependencies.jl`** — Dependency propagation between the two network copies. `phi_dep` matrix tracks per-neuron relationships (0=equal, 1=monotonic, -1=anti-monotonic, NaN=unknown). `encode_dependencies()` adds equality/inequality constraints or solves LP sub-problems for unknown cases.
- **`utils/mip.jl`** — MIP solver configuration, objective setup (`mip_set_delta_property`), result logging. Gurobi settings: MIPFocus=3, Threads=32, MIPGap=0.01.
- **`utils/models.jl`** — Loads pretrained PyTorch models (.p pickle files) into `Sequential` networks. Architectures: 3x10, 4x10, 10x10 (fully-connected), cnn0 (convolutional).
- **`utils/hyper_attack.jl`** / **`utils/hyper_attack.py`** — PGD-based attack that provides feasible solutions as MIP warm-start hints via `/tmp/` files.
- **`utils/help_functions.jl`** — Global mutable state: `layers_info_dict`, `neurons_names`, `reuse_bounds_conf`, `first_mip_solution`.

### MIPVerify Framework (`utils/MIPVerify.jl/`)

Vendored/modified version of [MIPVerify.jl](https://github.com/vtjeng/MIPVerify.jl). Key internals:

- **`src/net_components/core_ops.jl`** — ReLU big-M encoding, bound tightening (interval arithmetic / LP / MIP). The `relu(x, l, u)` function creates `x_rect` and binary `a` variables with four constraints. `layers_info_dict[(layer, neuron)] = (u, l, var_index)` stores bounds and variable positions for split neurons.
- **`src/net_components/layers/linear.jl`** — `Linear` layer: `matrix` shape is `(input_dim, output_dim)`, forward pass is `transpose(matrix) * x .+ bias`.
- **`src/net_components/layers/conv2d.jl`** — `Conv2d` layer: `filter` shape is `(height, width, in_channels, out_channels)`.
- **`src/net_components/layers/flatten.jl`** — `Flatten` with permutation, e.g. `Flatten([1, 3, 2, 4])`.
- **`src/net_components/nets/sequential.jl`** — `Sequential` network, supports piping: `x |> nn`.

### Global State

Defined in `utils/help_functions.jl`, used across files:

- `layers_info_dict::Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}}` — Maps `(layer, neuron)` to `(upper_bound, lower_bound, var_index)` for split ReLU neurons. For the dual encoding, original network uses layers 1..K, perturbed network uses layers K+1..2K.
- `neurons_names::NeuronsAssignNames` — Tracks current `layer` and `neuron` counters during encoding.
- `network_version::String` — Set to `"org"` or `"perturbation"` before encoding each network copy.
- `layer_counter`, `nueron_counter` — Additional counters for variable naming.
- `reuse_bounds_conf` — Caches bounds and dependencies across class-pair iterations.

### Variable Naming Convention

JuMP variables in the MIP are named: `{network_version}x_rect_layerCount{n}_neuronCount{n}_{layer}_{neuron}` and similarly for binary `a` variables. Access via `JuMP.all_variables(m)` with index offsets from `layers_info_dict`.

### Dual-Network Variable Mapping

In `get_perturbation_specific_keys_linf()`:
- `v_in` — Original input variables, bounded [0, 1]
- `v_x0` — Perturbed input variables, bounded [0, 1]
- `v_e` — Perturbation variables, bounded [-eps, eps]
- Constraint: `v_x0 == v_in + v_e`
- `v_in |> nn` encodes original network, `v_x0 |> nn` encodes perturbed network
- Returns dict with keys `:v_in`, `:v_in_p`, `:v_out`, `:v_out_p`, `:Perturbation`

## Conventions

- Layer type checks use `occursin("ReLU", string(typeof(l)))` pattern throughout.
- Class indices are 1-indexed in Julia, 0-indexed in Python (hyper_attack); conversions via `c_tag-1`.
- Perturbation sizes are comma-separated strings parsed by `parse_numbers_to_Float64()`.
- Results saved as CSV: `source,target,incumbent_obj,best_bound,solve_time`.
