# Interval bound propagation for the transfer proof.
#
# Same approach as lucid_delta_diff_with_perturbation:
# - Uses globals: I_z_prev_up/down, I_z_prev_up/down_perturbation,
#   all_bounds_of_original/perturbation
#
# compute_diff_and_comp_bounds() also populates the relaxation globals
# (relu_diff_*_bounds, relu_comp_*_bounds, n1_preact_*_bounds) used by
# the conditional-triangle relaxation in core_ops.jl.
# - Initialized in get_perturbation_specific_keys_linf_transfer()
# - Propagation (paper eq. 3): W2·(z_org - z_pre) + ΔW·z_pre, where ΔW = W2 - W1
# - Constraints: av[vec_n2] <= av[vec_n1] .+ I_z_prev_up_to_use
#                av[vec_n2] >= av[vec_n1] .+ I_z_prev_down_to_use

# ── helpers ──────────────────────────────────────────────────────────────

function interval_matrix_vector_multiplication(w_h_low, w_h_high, I_z_prev_min, I_z_prev_max)
    n = size(w_h_low, 1)
    m = size(w_h_low, 2)
    result_min = zeros(n)
    result_max = zeros(n)
    for i in 1:n
        min_val = 0.0
        max_val = 0.0
        for j in 1:m
            prod1 = w_h_low[i, j] * I_z_prev_min[j]
            prod2 = w_h_low[i, j] * I_z_prev_max[j]
            prod3 = w_h_high[i, j] * I_z_prev_min[j]
            prod4 = w_h_high[i, j] * I_z_prev_max[j]
            min_val += minimum([prod1, prod2, prod3, prod4])
            max_val += maximum([prod1, prod2, prod3, prod4])
        end
        result_min[i] = min_val
        result_max[i] = max_val
    end
    return result_min, result_max
end

"""
    interval_conv2d_bounds(F_low, F_high, x_low, x_high, bias, stride, padding)

Interval arithmetic for Conv2d: computes [output_min, output_max] given
filter interval [F_low, F_high] and input interval [x_low, x_high].

Filter shape: (fh, fw, in_channels, out_channels)
Input shape:  (batch, height, width, in_channels)
Output shape: (batch, out_height, out_width, out_channels)
"""
function interval_conv2d_bounds(F_low::Array{Float64,4}, F_high::Array{Float64,4},
                                x_low::Array{Float64,4}, x_high::Array{Float64,4},
                                bias::Vector{Float64}, stride::Int, padding)
    (batch, in_h, in_w, in_ch) = size(x_low)
    (fh, fw, _, out_ch) = size(F_low)

    ((out_h, out_w), (fh_off, fw_off)) = compute_output_parameters(in_h, in_w, fh, fw, stride, padding)

    result_min = zeros(Float64, batch, out_h, out_w, out_ch)
    result_max = zeros(Float64, batch, out_h, out_w, out_ch)

    for b_idx in 1:batch
        for oh in 1:out_h
            for ow in 1:out_w
                for oc in 1:out_ch
                    lo = bias[oc]
                    hi = bias[oc]
                    for fhi in 1:fh
                        for fwi in 1:fw
                            for ic in 1:in_ch
                                x_row = (oh - 1) * stride + fhi - fh_off
                                x_col = (ow - 1) * stride + fwi - fw_off
                                if x_row >= 1 && x_row <= in_h && x_col >= 1 && x_col <= in_w
                                    xl = x_low[b_idx, x_row, x_col, ic]
                                    xh = x_high[b_idx, x_row, x_col, ic]
                                    fl = F_low[fhi, fwi, ic, oc]
                                    fhv = F_high[fhi, fwi, ic, oc]
                                    p1 = fl * xl; p2 = fl * xh
                                    p3 = fhv * xl; p4 = fhv * xh
                                    lo += min(p1, p2, p3, p4)
                                    hi += max(p1, p2, p3, p4)
                                end
                            end
                        end
                    end
                    result_min[b_idx, oh, ow, oc] = lo
                    result_max[b_idx, oh, ow, oc] = hi
                end
            end
        end
    end
    return result_min, result_max
end

"""
    ensure_1d(x)

Flatten a multi-dimensional array to 1D for constraint addition.
Variables in JuMP are stored flat; interval arrays must match.
"""
function ensure_1d(x)
    ndims(x) > 1 ? vec(x) : x
end

"""
    extract_neuron_index(var_name)

Extract the structural neuron index (last integer) from a JuMP variable name.
Variable names follow: {prefix}x_rect_layerCount{n}_neuronCount{n}_{layer}_{neuron}
Returns the neuron index (1-based) used to index into interval bound arrays.
"""
function extract_neuron_index(var_name::String)
    # The neuron index is the last integer after the last underscore
    last_under = findlast('_', var_name)
    return parse(Int, var_name[last_under+1:end])
end

"""
    add_matched_interval_constraints!(m, av, vec_a, vec_b, I_up_flat, I_down_flat)

Add interval constraints only for neurons that have x_rect variables in BOTH
network copies. Neurons are matched by their structural neuron index (extracted
from JuMP variable names). This handles the case where bound tightening produces
different split/fixed decisions for the two copies (common in CNN layers).

Constraints added:  av[b_idx] <= av[a_idx] + I_up_flat[neuron]
                    av[b_idx] >= av[a_idx] + I_down_flat[neuron]
"""
function add_matched_interval_constraints!(m, av, vec_a, vec_b, I_up_flat, I_down_flat)
    # Build neuron_index → av_index maps for both copies
    map_a = Dict{Int,Int}()
    for idx in vec_a
        ni = extract_neuron_index(JuMP.name(av[idx]))
        map_a[ni] = idx
    end
    map_b = Dict{Int,Int}()
    for idx in vec_b
        ni = extract_neuron_index(JuMP.name(av[idx]))
        map_b[ni] = idx
    end

    # Find common neurons and add constraints
    common = sort(collect(intersect(keys(map_a), keys(map_b))))
    count = 0
    for ni in common
        @constraint(m, av[map_b[ni]] <= av[map_a[ni]] + I_up_flat[ni])
        @constraint(m, av[map_b[ni]] >= av[map_a[ni]] + I_down_flat[ni])
        count += 2
    end
    return count
end

# ── interval propagation (like HyperlinearDepsIn) ────────────────────────

function propagate_intervals(nn1, nn2, version)
    global I_z_prev_up
    global I_z_prev_down
    global I_z_prev_up_perturbation
    global I_z_prev_down_perturbation
    global all_bounds_of_original
    global all_bounds_of_perturbation

    layer_cnt = 0

    for (layer_idx, l) in enumerate(nn1.layers)
        if occursin("Flatten", string(typeof(l)))
            if version == "org"
                if ndims(I_z_prev_up) > 1
                    I_z_prev_up   = I_z_prev_up |> l
                    I_z_prev_down = I_z_prev_down |> l
                end
            elseif version == "perturbation"
                if ndims(I_z_prev_up_perturbation) > 1
                    I_z_prev_up_perturbation   = I_z_prev_up_perturbation |> l
                    I_z_prev_down_perturbation = I_z_prev_down_perturbation |> l
                end
            end

        elseif occursin("Linear", string(typeof(l)))
            l2 = nn2.layers[layer_idx]

            # Decomposition (paper eq. 3):
            #   W2·z_org - W1·z_pre = W2·(z_org - z_pre) + ΔW·z_pre
            # Term 1: ΔW·z_pre  (ΔW is constant → point interval [ΔW, ΔW])
            # Term 2: W2·diff   (W2 = W^N, diff = z_org - z_pre)
            w1       = Float64.(transpose(l.matrix))
            w2       = Float64.(transpose(l2.matrix))
            ΔW       = w2 - w1

            if version == "org"
                z_prev_up   = all_bounds_of_original[layer_cnt+1][1]
                z_prev_down = all_bounds_of_original[layer_cnt+1][2]
                I_z_prev_up_to_use   = I_z_prev_up
                I_z_prev_down_to_use = I_z_prev_down
            elseif version == "perturbation"
                z_prev_up   = all_bounds_of_perturbation[layer_cnt+1][1]
                z_prev_down = all_bounds_of_perturbation[layer_cnt+1][2]
                I_z_prev_up_to_use   = I_z_prev_up_perturbation
                I_z_prev_down_to_use = I_z_prev_down_perturbation
            end

            result_min_1, result_max_1 = interval_matrix_vector_multiplication(ΔW, ΔW, z_prev_down, z_prev_up)
            result_min_2, result_max_2 = interval_matrix_vector_multiplication(w2, w2, I_z_prev_down_to_use, I_z_prev_up_to_use)
            bias_diff = Float64.(l2.bias) .- Float64.(l.bias)
            I_z_down_in = result_min_1 .+ result_min_2 .+ bias_diff
            I_z_up_in   = result_max_1 .+ result_max_2 .+ bias_diff

            # Propagate activation bounds through W1
            new_up   = zeros(Float64, size(w1, 1))
            new_down = zeros(Float64, size(w1, 1))
            for i in 1:size(w1, 1)
                hi = 0.0
                lo = 0.0
                for j in 1:size(w1, 2)
                    p1 = w1[i,j] * z_prev_up[j]
                    p2 = w1[i,j] * z_prev_down[j]
                    hi += max(p1, p2)
                    lo += min(p1, p2)
                end
                new_up[i]   = hi + l.bias[i]
                new_down[i] = lo + l.bias[i]
            end

            # Store pre-ReLU interval and activation bounds
            if version == "org"
                I_z_prev_up   = I_z_up_in
                I_z_prev_down = I_z_down_in
            elseif version == "perturbation"
                I_z_prev_up_perturbation   = I_z_up_in
                I_z_prev_down_perturbation = I_z_down_in
            end

            if version == "org"
                push!(all_bounds_of_original, [new_up, new_down])
            elseif version == "perturbation"
                push!(all_bounds_of_perturbation, [new_up, new_down])
            end

        elseif occursin("Conv", string(typeof(l)))
            l2 = nn2.layers[layer_idx]

            F1 = Float64.(l.filter)
            F2 = Float64.(l2.filter)
            ΔF = F2 - F1
            zero_bias = zeros(Float64, length(l.bias))

            if version == "org"
                z_prev_up   = all_bounds_of_original[layer_cnt+1][1]
                z_prev_down = all_bounds_of_original[layer_cnt+1][2]
                I_z_prev_up_to_use   = I_z_prev_up
                I_z_prev_down_to_use = I_z_prev_down
            elseif version == "perturbation"
                z_prev_up   = all_bounds_of_perturbation[layer_cnt+1][1]
                z_prev_down = all_bounds_of_perturbation[layer_cnt+1][2]
                I_z_prev_up_to_use   = I_z_prev_up_perturbation
                I_z_prev_down_to_use = I_z_prev_down_perturbation
            end

            # Ensure 4D shape for conv operation
            z_prev_up_4d   = ndims(z_prev_up) == 4 ? z_prev_up : reshape(z_prev_up, 1, :, 1, 1)
            z_prev_down_4d = ndims(z_prev_down) == 4 ? z_prev_down : reshape(z_prev_down, 1, :, 1, 1)
            I_up_4d   = ndims(I_z_prev_up_to_use) == 4 ? I_z_prev_up_to_use : reshape(I_z_prev_up_to_use, 1, :, 1, 1)
            I_down_4d = ndims(I_z_prev_down_to_use) == 4 ? I_z_prev_down_to_use : reshape(I_z_prev_down_to_use, 1, :, 1, 1)

            # Term 1: ΔF·z_pre (ΔF is constant → point interval)
            result_min_1, result_max_1 = interval_conv2d_bounds(
                ΔF, ΔF, z_prev_down_4d, z_prev_up_4d, zero_bias, l.stride, l.padding)
            # Term 2: F2·diff (F2 = W^N filter)
            result_min_2, result_max_2 = interval_conv2d_bounds(
                F2, F2, I_down_4d, I_up_4d, zero_bias, l.stride, l.padding)

            # Bias difference: constant offset on the interval
            bias_diff = Float64.(l2.bias) .- Float64.(l.bias)

            I_z_down_in = result_min_1 .+ result_min_2
            I_z_up_in   = result_max_1 .+ result_max_2
            # Broadcast bias_diff into the spatial dimensions
            for oc in 1:length(bias_diff)
                I_z_down_in[:, :, :, oc] .+= bias_diff[oc]
                I_z_up_in[:, :, :, oc]   .+= bias_diff[oc]
            end

            # Activation bounds through F1 (with bias)
            new_down, new_up = interval_conv2d_bounds(
                F1, F1, z_prev_down_4d, z_prev_up_4d, Float64.(l.bias), l.stride, l.padding)

            if version == "org"
                I_z_prev_up   = I_z_up_in
                I_z_prev_down = I_z_down_in
                push!(all_bounds_of_original, [new_up, new_down])
            elseif version == "perturbation"
                I_z_prev_up_perturbation   = I_z_up_in
                I_z_prev_down_perturbation = I_z_down_in
                push!(all_bounds_of_perturbation, [new_up, new_down])
            end

        elseif occursin("ReLU", string(typeof(l)))
            layer_cnt += 1

            # ReLU on interval bounds (same as lucid)
            if version == "org"
                I_z_prev_up   = max.(0.0, I_z_prev_up)
                I_z_prev_down = .- max.(0.0, .- I_z_prev_down)
            elseif version == "perturbation"
                I_z_prev_up_perturbation   = max.(0.0, I_z_prev_up_perturbation)
                I_z_prev_down_perturbation = .- max.(0.0, .- I_z_prev_down_perturbation)
            end

            # ReLU on activation bounds
            if version == "org"
                all_bounds_of_original[end][1] = max.(0.0, all_bounds_of_original[end][1])
                all_bounds_of_original[end][2] = max.(0.0, all_bounds_of_original[end][2])
            elseif version == "perturbation"
                all_bounds_of_perturbation[end][1] = max.(0.0, all_bounds_of_perturbation[end][1])
                all_bounds_of_perturbation[end][2] = max.(0.0, all_bounds_of_perturbation[end][2])
            end
        end
    end
end

# ── constraint addition (like HyperlinearDeps) ──────────────────────────

function add_interval_constraints(m, nn1, nn2, version, n1_prefix, n2_prefix)
    global I_z_prev_up
    global I_z_prev_down
    global I_z_prev_up_perturbation
    global I_z_prev_down_perturbation
    global all_bounds_of_original
    global all_bounds_of_perturbation

    av = JuMP.all_variables(m)
    layer_cnt = 0
    constraints_added = 0

    # Re-propagate intervals layer by layer so we have per-layer values for constraints
    # Reset to initial values — preserve shape (4D for conv, 1D for FC)
    if version == "org"
        I_z_prev_up_local   = zeros(Float64, size(all_bounds_of_original[1][1]))
        I_z_prev_down_local = zeros(Float64, size(all_bounds_of_original[1][1]))
    elseif version == "perturbation"
        I_z_prev_up_local   = zeros(Float64, size(all_bounds_of_perturbation[1][1]))
        I_z_prev_down_local = zeros(Float64, size(all_bounds_of_perturbation[1][1]))
    end

    for (layer_idx, l) in enumerate(nn1.layers)
        if occursin("Flatten", string(typeof(l)))
            if ndims(I_z_prev_up_local) > 1
                I_z_prev_up_local   = I_z_prev_up_local |> l
                I_z_prev_down_local = I_z_prev_down_local |> l
            end

        elseif occursin("Linear", string(typeof(l)))
            l2 = nn2.layers[layer_idx]

            ΔW = Float64.(transpose(l2.matrix)) - Float64.(transpose(l.matrix))
            w2 = Float64.(transpose(l2.matrix))

            if version == "org"
                z_prev_up   = all_bounds_of_original[layer_cnt+1][1]
                z_prev_down = all_bounds_of_original[layer_cnt+1][2]
            elseif version == "perturbation"
                z_prev_up   = all_bounds_of_perturbation[layer_cnt+1][1]
                z_prev_down = all_bounds_of_perturbation[layer_cnt+1][2]
            end

            result_min_1, result_max_1 = interval_matrix_vector_multiplication(ΔW, ΔW, z_prev_down, z_prev_up)
            result_min_2, result_max_2 = interval_matrix_vector_multiplication(w2, w2, I_z_prev_down_local, I_z_prev_up_local)
            I_z_prev_down_local = result_min_1 .+ result_min_2
            I_z_prev_up_local   = result_max_1 .+ result_max_2

        elseif occursin("Conv", string(typeof(l)))
            l2 = nn2.layers[layer_idx]

            F1 = Float64.(l.filter)
            F2 = Float64.(l2.filter)
            ΔF = F2 - F1
            zero_bias = zeros(Float64, length(l.bias))

            if version == "org"
                z_prev_up   = all_bounds_of_original[layer_cnt+1][1]
                z_prev_down = all_bounds_of_original[layer_cnt+1][2]
            elseif version == "perturbation"
                z_prev_up   = all_bounds_of_perturbation[layer_cnt+1][1]
                z_prev_down = all_bounds_of_perturbation[layer_cnt+1][2]
            end

            z_prev_up_4d   = ndims(z_prev_up) == 4 ? z_prev_up : reshape(z_prev_up, 1, :, 1, 1)
            z_prev_down_4d = ndims(z_prev_down) == 4 ? z_prev_down : reshape(z_prev_down, 1, :, 1, 1)
            I_up_4d   = ndims(I_z_prev_up_local) == 4 ? I_z_prev_up_local : reshape(I_z_prev_up_local, 1, :, 1, 1)
            I_down_4d = ndims(I_z_prev_down_local) == 4 ? I_z_prev_down_local : reshape(I_z_prev_down_local, 1, :, 1, 1)

            result_min_1, result_max_1 = interval_conv2d_bounds(
                ΔF, ΔF, z_prev_down_4d, z_prev_up_4d, zero_bias, l.stride, l.padding)
            result_min_2, result_max_2 = interval_conv2d_bounds(
                F2, F2, I_down_4d, I_up_4d, zero_bias, l.stride, l.padding)

            I_z_prev_down_local = result_min_1 .+ result_min_2
            I_z_prev_up_local   = result_max_1 .+ result_max_2

        elseif occursin("ReLU", string(typeof(l)))
            layer_cnt += 1

            # ReLU on intervals
            I_z_prev_up_local   = max.(0.0, I_z_prev_up_local)
            I_z_prev_down_local = .- max.(0.0, .- I_z_prev_down_local)

            # Find N1 variables by name (like lucid HyperlinearDeps)
            vec_1 = []
            for n in eachindex(av)
                if occursin(n1_prefix * "x_rect" * "_" * "layerCount" * string(layer_cnt), JuMP.name(av[n]))
                    append!(vec_1, n)
                end
            end

            # Find N2 variables by name
            vec_2 = []
            for n in eachindex(av)
                if occursin(n2_prefix * "x_rect" * "_" * "layerCount" * string(layer_cnt), JuMP.name(av[n]))
                    append!(vec_2, n)
                end
            end

            if length(vec_1) > 0 && length(vec_2) > 0
                I_up_flat   = ensure_1d(I_z_prev_up_local)
                I_down_flat = ensure_1d(I_z_prev_down_local)
                constraints_added += add_matched_interval_constraints!(
                    m, av, vec_1, vec_2, I_up_flat, I_down_flat)
            end
        end
    end

    return constraints_added
end

# ── main entry point ─────────────────────────────────────────────────────

function transfer_interval_constraints(m, nn1, nn2, perturbation, perturbation_size, w, h, k)
    global all_bounds_of_original
    global all_bounds_of_perturbation

    # Phase 1: Propagate intervals (populates globals and all_bounds)
    propagate_intervals(nn1, nn2, "org")
    propagate_intervals(nn1, nn2, "perturbation")

    # Phase 2: Add constraints to MIP (re-propagates per-layer for correct values)
    # Clean: N1(x) prefix = "n1_org", N2(x) prefix = "n2_org"
    c1 = add_interval_constraints(m, nn1, nn2, "org", "n1_org", "n2_org")

    # Perturbed: N1(x_p) prefix = "n1_pert", N2(x_p) prefix = "n2_pert"
    c2 = add_interval_constraints(m, nn1, nn2, "perturbation", "n1_pert", "n2_pert")

    println("Interval constraints added: $(c1 + c2)")
end

# ═══════════════════════════════════════════════════════════════════════════
# Perturbation intervals: bounds on Δ = a^p - a (perturbed minus clean)
# for a SINGLE network, using the L∞ perturbation ε.
#
# Recursion (difference form):
#   Init: I^pert ∈ [-ε, ε]
#   Linear layer m: I^pert_z = interval_mul(W, W, I^pert_a_prev)
#   Conv2d layer m: I^pert_z = interval_conv(F, F, I^pert_a_prev)
#   ReLU:           I^pert_a_up  = max(0, I^pert_z_up)
#                   I^pert_a_down = -max(0, -I^pert_z_down)
#
# Constraint: x_rect_pert ∈ [x_rect_clean + I^pert_down, x_rect_clean + I^pert_up]
# ═══════════════════════════════════════════════════════════════════════════

"""
    perturbed_interval_constraints(m, nn, clean_prefix, pert_prefix)

Propagates perturbation intervals through network `nn` and adds constraints
between clean (`clean_prefix`) and perturbed (`pert_prefix`) x_rect variables.

Works for both standard mode (clean_prefix="org", pert_prefix="perturbation")
and transfer mode (e.g. clean_prefix="n1_org", pert_prefix="n1_pert").

Uses global `I_pert_prev_up/down` initialized to ±ε in the encoding function.
"""
# ═══════════════════════════════════════════════════════════════════════════
# Composed intervals: bounds on Δ^C = a^{N2}(x_p) - a^{N1}(x)
# This directly links the first and last network copies in transfer mode,
# "closing the triangle" N1(x) ↔ N2(x) ↔ N2(x_p).
#
# Propagation (difference form):
#   Init: I^C ∈ [-ε, ε]  (at input: x_p - x = e ∈ [-ε, ε])
#   Linear layer ℓ (W1 from nn1, W2 from nn2):
#     Δz^C_ℓ = (W2-W1)^T · a^{N1}_{ℓ-1} + W2^T · Δa^C_{ℓ-1}
#     Term 1: IA([0, W2-W1], [a^{N1}_down, a^{N1}_up])  — weight difference
#     Term 2: IA([W2, W2],   [I^C_down, I^C_up])         — prev composed interval
#   Conv2d layer ℓ (F1 from nn1, F2 from nn2):
#     Same formula but with interval_conv2d_bounds instead of interval_matrix_vector_multiplication
#   ReLU:
#     I^C_up  = max(0, I^C_up)
#     I^C_down = -max(0, -I^C_down)
#
# Constraint: n2_pert_x_rect ∈ [n1_org_x_rect + I^C_down, n1_org_x_rect + I^C_up]
# ═══════════════════════════════════════════════════════════════════════════

"""
    composed_interval_constraints(m, nn1, nn2, perturbation, perturbation_size, w, h, k)

Propagates composed intervals I^C through the networks and adds constraints
linking N1(x) (prefix "n1_org") directly to N2(x_p) (prefix "n2_pert").

I^C bounds the difference a^{N2}(x_p) - a^{N1}(x), combining both the
weight difference (W2 ≠ W1) and the input perturbation (x_p = x + e).

Uses `all_bounds_of_original` (must be populated by a prior call to
`transfer_interval_constraints` or `propagate_intervals`).
Uses `I_pert_prev_up/down` for the initial ±ε values.
"""
function composed_interval_constraints(m, nn1, nn2, perturbation, perturbation_size, w, h, k)
    global all_bounds_of_original
    global I_pert_prev_up
    global I_pert_prev_down

    # Ensure all_bounds_of_original is fully populated (may not be if
    # transfer_interval_constraints was not called before us)
    if length(all_bounds_of_original) <= 1
        println("  Propagating transfer intervals (needed for I^C activation bounds)...")
        propagate_intervals(nn1, nn2, "org")
    end

    av = JuMP.all_variables(m)
    layer_cnt = 0
    constraints_added = 0

    # Initialize composed interval to [-ε, ε] (same shape as perturbation interval)
    I_C_up_local   = copy(I_pert_prev_up)
    I_C_down_local = copy(I_pert_prev_down)

    for (layer_idx, l) in enumerate(nn1.layers)
        if occursin("Flatten", string(typeof(l)))
            if ndims(I_C_up_local) > 1
                I_C_up_local   = I_C_up_local |> l
                I_C_down_local = I_C_down_local |> l
            end

        elseif occursin("Linear", string(typeof(l)))
            l2 = nn2.layers[layer_idx]

            # Weight matrices
            W1 = Float64.(transpose(l.matrix))
            W2 = Float64.(transpose(l2.matrix))

            # Weight difference (constant → point interval)
            ΔW = W2 - W1

            # Activation bounds of N1(x) at this layer
            z_prev_up   = all_bounds_of_original[layer_cnt+1][1]
            z_prev_down = all_bounds_of_original[layer_cnt+1][2]

            # Term 1: ΔW * a^{N1}_{prev} — weight difference contribution
            result_min_1, result_max_1 = interval_matrix_vector_multiplication(
                ΔW, ΔW, z_prev_down, z_prev_up)

            # Term 2: W2 * Δa^C_{prev} — composed interval through N2's weights
            # Key difference from transfer: uses W2, not W1
            result_min_2, result_max_2 = interval_matrix_vector_multiplication(
                W2, W2, I_C_down_local, I_C_up_local)

            bias_diff = Float64.(l2.bias) .- Float64.(l.bias)
            I_C_down_local = result_min_1 .+ result_min_2 .+ bias_diff
            I_C_up_local   = result_max_1 .+ result_max_2 .+ bias_diff

        elseif occursin("Conv", string(typeof(l)))
            l2 = nn2.layers[layer_idx]

            F1 = Float64.(l.filter)
            F2 = Float64.(l2.filter)
            ΔF = F2 - F1
            zero_bias = zeros(Float64, length(l.bias))

            z_prev_up   = all_bounds_of_original[layer_cnt+1][1]
            z_prev_down = all_bounds_of_original[layer_cnt+1][2]

            z_prev_up_4d   = ndims(z_prev_up) == 4 ? z_prev_up : reshape(z_prev_up, 1, :, 1, 1)
            z_prev_down_4d = ndims(z_prev_down) == 4 ? z_prev_down : reshape(z_prev_down, 1, :, 1, 1)
            I_C_up_4d   = ndims(I_C_up_local) == 4 ? I_C_up_local : reshape(I_C_up_local, 1, :, 1, 1)
            I_C_down_4d = ndims(I_C_down_local) == 4 ? I_C_down_local : reshape(I_C_down_local, 1, :, 1, 1)

            # Term 1: ΔF * activation bounds (point interval)
            result_min_1, result_max_1 = interval_conv2d_bounds(
                ΔF, ΔF, z_prev_down_4d, z_prev_up_4d, zero_bias, l.stride, l.padding)
            # Term 2: N2 weights * composed interval
            result_min_2, result_max_2 = interval_conv2d_bounds(
                F2, F2, I_C_down_4d, I_C_up_4d, zero_bias, l.stride, l.padding)

            bias_diff = Float64.(l2.bias) .- Float64.(l.bias)

            I_C_down_local = result_min_1 .+ result_min_2
            I_C_up_local   = result_max_1 .+ result_max_2
            for oc in 1:length(bias_diff)
                I_C_down_local[:, :, :, oc] .+= bias_diff[oc]
                I_C_up_local[:, :, :, oc]   .+= bias_diff[oc]
            end

        elseif occursin("ReLU", string(typeof(l)))
            layer_cnt += 1

            # ReLU relaxation on composed difference
            I_C_up_local   = max.(0.0, I_C_up_local)
            I_C_down_local = .- max.(0.0, .- I_C_down_local)

            # Find N1(x) x_rect variables
            vec_n1 = []
            for n in eachindex(av)
                if occursin("n1_org" * "x_rect" * "_" * "layerCount" * string(layer_cnt), JuMP.name(av[n]))
                    append!(vec_n1, n)
                end
            end

            # Find N2(x_p) x_rect variables
            vec_n2p = []
            for n in eachindex(av)
                if occursin("n2_pert" * "x_rect" * "_" * "layerCount" * string(layer_cnt), JuMP.name(av[n]))
                    append!(vec_n2p, n)
                end
            end

            if length(vec_n1) > 0 && length(vec_n2p) > 0
                I_C_up_flat   = ensure_1d(I_C_up_local)
                I_C_down_flat = ensure_1d(I_C_down_local)
                constraints_added += add_matched_interval_constraints!(
                    m, av, vec_n1, vec_n2p, I_C_up_flat, I_C_down_flat)
            end
        end
    end

    println("Composed interval constraints (I^C) added: $constraints_added")
    return constraints_added
end


function perturbed_interval_constraints(m, nn, clean_prefix, pert_prefix)
    global I_pert_prev_up
    global I_pert_prev_down

    av = JuMP.all_variables(m)
    layer_cnt = 0
    constraints_added = 0

    # Local copy of I_pert for propagation
    I_pert_up_local   = copy(I_pert_prev_up)
    I_pert_down_local = copy(I_pert_prev_down)

    for (layer_idx, l) in enumerate(nn.layers)
        if occursin("Flatten", string(typeof(l)))
            if ndims(I_pert_up_local) > 1
                I_pert_up_local   = I_pert_up_local |> l
                I_pert_down_local = I_pert_down_local |> l
            end

        elseif occursin("Linear", string(typeof(l)))
            # W = W1^T (same network for both clean and perturbed)
            W = Float64.(transpose(l.matrix))

            # Δz_{m,k} = Σ w_{m,k,k'} · Δa_{m-1,k'}
            # Since weights are fixed: interval_mul(W, W, I_down, I_up)
            r_min, r_max = interval_matrix_vector_multiplication(W, W, I_pert_down_local, I_pert_up_local)
            I_pert_down_local = r_min
            I_pert_up_local   = r_max

        elseif occursin("Conv", string(typeof(l)))
            F = Float64.(l.filter)
            zero_bias = zeros(Float64, length(l.bias))

            I_up_4d   = ndims(I_pert_up_local) == 4 ? I_pert_up_local : reshape(I_pert_up_local, 1, :, 1, 1)
            I_down_4d = ndims(I_pert_down_local) == 4 ? I_pert_down_local : reshape(I_pert_down_local, 1, :, 1, 1)

            r_min, r_max = interval_conv2d_bounds(F, F, I_down_4d, I_up_4d, zero_bias, l.stride, l.padding)
            I_pert_down_local = r_min
            I_pert_up_local   = r_max

        elseif occursin("ReLU", string(typeof(l)))
            layer_cnt += 1

            # ReLU relaxation on difference:
            # σ(z+Δ) - σ(z) ∈ [min(0, Δ_down), max(0, Δ_up)]
            I_pert_up_local   = max.(0.0, I_pert_up_local)
            I_pert_down_local = .- max.(0.0, .- I_pert_down_local)

            # Find clean x_rect variables by name
            vec_clean = []
            for n in eachindex(av)
                if occursin(clean_prefix * "x_rect" * "_" * "layerCount" * string(layer_cnt), JuMP.name(av[n]))
                    append!(vec_clean, n)
                end
            end

            # Find perturbed x_rect variables by name
            vec_pert = []
            for n in eachindex(av)
                if occursin(pert_prefix * "x_rect" * "_" * "layerCount" * string(layer_cnt), JuMP.name(av[n]))
                    append!(vec_pert, n)
                end
            end

            if length(vec_clean) > 0 && length(vec_pert) > 0
                I_up_flat   = ensure_1d(I_pert_up_local)
                I_down_flat = ensure_1d(I_pert_down_local)
                constraints_added += add_matched_interval_constraints!(
                    m, av, vec_clean, vec_pert, I_up_flat, I_down_flat)
            end
        end
    end

    return constraints_added
end

# ═══════════════════════════════════════════════════════════════════════════
# Distillation interval constraints
#
# For transfer_distilation mode where N2 has twice as many layers as N1.
# Layer mapping: N1 ReLU layer k ↔ N2 ReLU layer 2k.
#
# Since architectures differ, coupled interval propagation (which requires
# matching weight matrices) cannot be used. Instead, we compute independent
# activation bounds for each network via interval arithmetic and constrain
# the difference at matched layers:
#   av[N2_layer_2k] ∈ [av[N1_layer_k] + diff_down, av[N1_layer_k] + diff_up]
# where diff_up = N2_ub - N1_lb, diff_down = N2_lb - N1_ub.
# ═══════════════════════════════════════════════════════════════════════════

"""
    compute_activation_bounds_ia(nn)

Compute per-ReLU-layer activation bounds via interval arithmetic.
Returns a list of (upper_bound, lower_bound) tuples, one per ReLU layer.
Input is assumed to be in [0, 1]^n. Supports both Linear and Conv2d layers.
"""
function compute_activation_bounds_ia(nn; input_h::Int=28, input_w::Int=28)
    bounds = []
    current_up = nothing
    current_down = nothing
    initialized = false

    for l in nn.layers
        if occursin("Flatten", string(typeof(l)))
            if initialized && ndims(current_up) > 1
                current_up = current_up |> l
                current_down = current_down |> l
            end

        elseif occursin("Linear", string(typeof(l)))
            W = Float64.(transpose(l.matrix))
            b = Float64.(l.bias)
            if !initialized
                n_in = size(W, 2)
                current_up = ones(Float64, n_in)
                current_down = zeros(Float64, n_in)
                initialized = true
            end
            new_up = zeros(Float64, size(W, 1))
            new_down = zeros(Float64, size(W, 1))
            for i in 1:size(W, 1)
                hi = b[i]
                lo = b[i]
                for j in 1:size(W, 2)
                    p1 = W[i,j] * current_up[j]
                    p2 = W[i,j] * current_down[j]
                    hi += max(p1, p2)
                    lo += min(p1, p2)
                end
                new_up[i] = hi
                new_down[i] = lo
            end
            current_up = new_up
            current_down = new_down

        elseif occursin("Conv", string(typeof(l)))
            F = Float64.(l.filter)
            b = Float64.(l.bias)
            if !initialized
                (fh, fw, in_ch, _) = size(F)
                current_up = ones(Float64, 1, input_h, input_w, in_ch)
                current_down = zeros(Float64, 1, input_h, input_w, in_ch)
                initialized = true
            end
            up_4d   = ndims(current_up) == 4 ? current_up : reshape(current_up, 1, :, 1, 1)
            down_4d = ndims(current_down) == 4 ? current_down : reshape(current_down, 1, :, 1, 1)
            new_down, new_up = interval_conv2d_bounds(F, F, down_4d, up_4d, b, l.stride, l.padding)
            current_up = new_up
            current_down = new_down

        elseif occursin("ReLU", string(typeof(l)))
            current_up = max.(0.0, current_up)
            current_down = max.(0.0, current_down)
            push!(bounds, (copy(current_up), copy(current_down)))
        end
    end
    return bounds
end

"""
    compute_activation_bounds_perturbed_ia(nn, I_pert_up, I_pert_down)

Compute per-ReLU-layer activation bounds for a network on perturbed input
x_p = x + e where e ∈ [I_pert_down, I_pert_up]. Input x ∈ [0,1].
So x_p ∈ [max(0, I_pert_down), min(1, 1 + I_pert_up)].
Supports both Linear and Conv2d layers.
"""
function compute_activation_bounds_perturbed_ia(nn, I_pert_up, I_pert_down; input_h::Int=28, input_w::Int=28)
    bounds = []
    current_up = nothing
    current_down = nothing
    initialized = false

    for l in nn.layers
        if occursin("Flatten", string(typeof(l)))
            if initialized && ndims(current_up) > 1
                current_up = current_up |> l
                current_down = current_down |> l
            end

        elseif occursin("Linear", string(typeof(l)))
            W = Float64.(transpose(l.matrix))
            b = Float64.(l.bias)
            if !initialized
                n_in = size(W, 2)
                pert_up_flat = reshape(I_pert_up, :)
                pert_down_flat = reshape(I_pert_down, :)
                n_use = min(n_in, length(pert_up_flat))
                current_up = min.(1.0, ones(Float64, n_in) .+ pert_up_flat[1:n_use])
                current_down = max.(0.0, zeros(Float64, n_in) .+ pert_down_flat[1:n_use])
                initialized = true
            end
            new_up = zeros(Float64, size(W, 1))
            new_down = zeros(Float64, size(W, 1))
            for i in 1:size(W, 1)
                hi = b[i]
                lo = b[i]
                for j in 1:size(W, 2)
                    p1 = W[i,j] * current_up[j]
                    p2 = W[i,j] * current_down[j]
                    hi += max(p1, p2)
                    lo += min(p1, p2)
                end
                new_up[i] = hi
                new_down[i] = lo
            end
            current_up = new_up
            current_down = new_down

        elseif occursin("Conv", string(typeof(l)))
            F = Float64.(l.filter)
            b = Float64.(l.bias)
            if !initialized
                # First layer is Conv: perturbed input x_p = x + e, x ∈ [0,1], e ∈ [I_pert_down, I_pert_up]
                (fh, fw, in_ch, _) = size(F)
                pert_up_4d = ndims(I_pert_up) == 4 ? I_pert_up : reshape(I_pert_up, 1, input_h, input_w, in_ch)
                pert_down_4d = ndims(I_pert_down) == 4 ? I_pert_down : reshape(I_pert_down, 1, input_h, input_w, in_ch)
                current_up = min.(1.0, ones(Float64, size(pert_up_4d)) .+ pert_up_4d)
                current_down = max.(0.0, zeros(Float64, size(pert_down_4d)) .+ pert_down_4d)
                initialized = true
            end
            up_4d   = ndims(current_up) == 4 ? current_up : reshape(current_up, 1, :, 1, 1)
            down_4d = ndims(current_down) == 4 ? current_down : reshape(current_down, 1, :, 1, 1)
            new_down, new_up = interval_conv2d_bounds(F, F, down_4d, up_4d, b, l.stride, l.padding)
            current_up = new_up
            current_down = new_down

        elseif occursin("ReLU", string(typeof(l)))
            current_up = max.(0.0, current_up)
            current_down = max.(0.0, current_down)
            push!(bounds, (copy(current_up), copy(current_down)))
        end
    end
    return bounds
end

"""
    add_interval_constraints_distilation(m, bounds_n1, bounds_n2, K1, K2, n1_prefix, n2_prefix)

Add interval constraints at matched layers: N1 ReLU layer k ↔ N2 ReLU layer 2k.
Uses independently-computed activation bounds from `compute_activation_bounds_ia`.
"""
function add_interval_constraints_distilation(m, bounds_n1, bounds_n2, K1, K2, n1_prefix, n2_prefix)
    av = JuMP.all_variables(m)
    constraints_added = 0

    for k in 1:K1
        n2_layer = 2 * k
        if n2_layer > K2
            break
        end

        # Find N1 x_rect variables for layerCount k
        vec_1 = []
        for n in eachindex(av)
            if occursin(n1_prefix * "x_rect" * "_" * "layerCount" * string(k), JuMP.name(av[n]))
                append!(vec_1, n)
            end
        end

        # Find N2 x_rect variables for layerCount 2k
        vec_2 = []
        for n in eachindex(av)
            if occursin(n2_prefix * "x_rect" * "_" * "layerCount" * string(n2_layer), JuMP.name(av[n]))
                append!(vec_2, n)
            end
        end

        if length(vec_1) > 0 && length(vec_2) > 0
            # Difference bounds: diff = N2_activation - N1_activation
            diff_up   = ensure_1d(bounds_n2[n2_layer][1]) .- ensure_1d(bounds_n1[k][2])    # N2_up - N1_down
            diff_down = ensure_1d(bounds_n2[n2_layer][2]) .- ensure_1d(bounds_n1[k][1])    # N2_down - N1_up
            added = add_matched_interval_constraints!(
                m, av, vec_1, vec_2, diff_up, diff_down)
            constraints_added += added
            println("  Distillation interval: N1 layer $k ↔ N2 layer $n2_layer, $(added ÷ 2) neurons")
        end
    end

    return constraints_added
end

"""
    transfer_interval_constraints_distilation(m, nn1, nn2)

Main entry point for distillation interval constraints.
Computes independent activation bounds and adds constraints at matched layers.
"""
function transfer_interval_constraints_distilation(m, nn1, nn2)
    bounds_n1 = compute_activation_bounds_ia(nn1)
    bounds_n2 = compute_activation_bounds_ia(nn2)
    K1 = length(bounds_n1)
    K2 = length(bounds_n2)

    # Clean copies: N1(x) ↔ N2(x)
    c1 = add_interval_constraints_distilation(m, bounds_n1, bounds_n2, K1, K2, "n1_org", "n2_org")
    # Perturbed copies: N1(x_p) ↔ N2(x_p)
    c2 = add_interval_constraints_distilation(m, bounds_n1, bounds_n2, K1, K2, "n1_pert", "n2_pert")

    println("Distillation interval constraints added: $(c1 + c2)")
end

"""
    composed_interval_constraints_distilation(m, nn1, nn2)

Composed interval constraints linking N1(x) directly to N2(x_p) in distillation mode.
Uses independent activation bounds at matched layers (N1 layer k ↔ N2 layer 2k).
"""
function composed_interval_constraints_distilation(m, nn1, nn2)
    global I_pert_prev_up
    global I_pert_prev_down

    bounds_n1 = compute_activation_bounds_ia(nn1)
    bounds_n2_pert = compute_activation_bounds_perturbed_ia(nn2, I_pert_prev_up, I_pert_prev_down)
    K1 = length(bounds_n1)
    K2 = length(bounds_n2_pert)

    av = JuMP.all_variables(m)
    constraints_added = 0

    for k in 1:K1
        n2_layer = 2 * k
        if n2_layer > K2
            break
        end

        # Find N1(x) x_rect variables
        vec_n1 = []
        for n in eachindex(av)
            if occursin("n1_org" * "x_rect" * "_" * "layerCount" * string(k), JuMP.name(av[n]))
                append!(vec_n1, n)
            end
        end

        # Find N2(x_p) x_rect variables
        vec_n2p = []
        for n in eachindex(av)
            if occursin("n2_pert" * "x_rect" * "_" * "layerCount" * string(n2_layer), JuMP.name(av[n]))
                append!(vec_n2p, n)
            end
        end

        if length(vec_n1) > 0 && length(vec_n2p) > 0
            diff_up   = ensure_1d(bounds_n2_pert[n2_layer][1]) .- ensure_1d(bounds_n1[k][2])
            diff_down = ensure_1d(bounds_n2_pert[n2_layer][2]) .- ensure_1d(bounds_n1[k][1])
            added = add_matched_interval_constraints!(
                m, av, vec_n1, vec_n2p, diff_up, diff_down)
            constraints_added += added
            println("  Composed distillation: N1(x) layer $k ↔ N2(x_p) layer $n2_layer, $(added ÷ 2) neurons")
        end
    end

    println("Composed distillation interval constraints (I^C) added: $constraints_added")
    return constraints_added
end

# ═══════════════════════════════════════════════════════════════════════════
# compute_diff_and_comp_bounds
#
# Analytically propagates, per ReLU layer:
#   • diff bounds   [l_diff, u_diff]  =  z_n2_org - z_n1_org
#   • pert bounds   [l_pert, u_pert]  =  z_n2_pert - z_n2_org
#   • comp bounds   [l_comp, u_comp]  =  l_diff+l_pert, u_diff+u_pert
#   • N1 pre-activation bounds        =  bounds on zˆ_n1 before ReLU
#
# Results are stored in the relaxation globals (see help_functions.jl) and
# consumed by the conditional-triangle relaxation in core_ops.jl's relu().
#
# Must be called BEFORE encoding n2_org / n2_pert so that relu() can skip
# binary variables for qualifying neurons.
#
# Requires all_bounds_of_original to be initialised (input-layer entry only
# is sufficient; the rest is built here in parallel with propagate_intervals).
# ═══════════════════════════════════════════════════════════════════════════
function compute_diff_and_comp_bounds(nn1, nn2, I_pert_up_init, I_pert_down_init; optimizing_intervals::Bool=true)
    # ── Activation conditional-triangle relaxation (n2_org) ──────────────────
    # diff bounds: z_n2_org - z_n1_org, used with a_n1_org + N1 preact bounds
    global relu_diff_up_bounds, relu_diff_down_bounds
    global n1_preact_up_bounds, n1_preact_down_bounds
    # ── Perturbation conditional-triangle relaxation (n2_pert) ───────────────
    # composed bounds: (z_n2_pert - z_n1_org) = diff + pert, used with a_n1_org + N1 preact bounds
    global relu_comp_up_bounds, relu_comp_down_bounds

    relu_diff_up_bounds   = Array{Float64}[]
    relu_diff_down_bounds = Array{Float64}[]
    n1_preact_up_bounds   = Array{Float64}[]
    n1_preact_down_bounds = Array{Float64}[]
    relu_comp_up_bounds   = Array{Float64}[]
    relu_comp_down_bounds = Array{Float64}[]

    # Running interval state (4D for CNNs, flattened to 1D at Flatten layer)
    diff_up   = zeros(Float64, size(I_pert_up_init))   # z_n2_org - z_n1_org
    diff_down = zeros(Float64, size(I_pert_down_init))
    pert_up   = copy(Float64.(I_pert_up_init))          # z_n2_pert - z_n2_org
    pert_down = copy(Float64.(I_pert_down_init))

    # N1 post-activation bounds (initialised to input domain [0,1])
    n1_act_up   = ones(Float64,  size(I_pert_up_init))
    n1_act_down = zeros(Float64, size(I_pert_down_init))

    # Will be set at each Linear layer, read at the following ReLU layer
    n1_pre_up_cur   = Float64[]
    n1_pre_down_cur = Float64[]

    for (layer_idx, l) in enumerate(nn1.layers)
        l2 = nn2.layers[layer_idx]

        if occursin("Flatten", string(typeof(l)))
            if ndims(diff_up) > 1
                diff_up   = vec(diff_up   |> l)
                diff_down = vec(diff_down |> l)
                pert_up   = vec(pert_up   |> l)
                pert_down = vec(pert_down |> l)
                n1_act_up   = vec(n1_act_up   |> l)
                n1_act_down = vec(n1_act_down |> l)
            end

        elseif occursin("Linear", string(typeof(l)))
            W1 = Float64.(transpose(l.matrix))
            W2 = Float64.(transpose(l2.matrix))
            b1 = Float64.(l.bias)
            b2 = Float64.(l2.bias)
            ΔW = W2 - W1
            Δb = b2 - b1

            # diff:  ΔW·z_n1 + W2·diff_prev + Δb  (paper eq. 3)
            r1_min, r1_max = interval_matrix_vector_multiplication(
                ΔW, ΔW, n1_act_down, n1_act_up)
            r2_min, r2_max = interval_matrix_vector_multiplication(
                W2, W2, diff_down, diff_up)
            diff_down = r1_min .+ r2_min .+ Δb
            diff_up   = r1_max .+ r2_max .+ Δb

            # pert:  W2·pert_prev  (bias cancels in the difference)
            rp_min, rp_max = interval_matrix_vector_multiplication(
                W2, W2, pert_down, pert_up)
            pert_down = rp_min
            pert_up   = rp_max

            # N1 pre-activation bounds (before ReLU)
            rn_min, rn_max = interval_matrix_vector_multiplication(
                W1, W1, n1_act_down, n1_act_up)
            n1_pre_up_cur   = rn_max .+ b1
            n1_pre_down_cur = rn_min .+ b1

        elseif occursin("Conv", string(typeof(l)))
            l2 = nn2.layers[layer_idx]
            F1 = Float64.(l.filter);  F2 = Float64.(l2.filter)
            b1 = Float64.(l.bias);    b2 = Float64.(l2.bias)
            ΔF = F2 - F1;             Δb = b2 - b1
            zero_bias = zeros(Float64, length(b1))

            # Ensure 4D shape for conv operation
            n1_4d_down  = ndims(n1_act_down) == 4 ? n1_act_down : reshape(n1_act_down, 1, :, 1, 1)
            n1_4d_up    = ndims(n1_act_up)   == 4 ? n1_act_up   : reshape(n1_act_up,   1, :, 1, 1)
            diff_4d_down = ndims(diff_down)  == 4 ? diff_down   : reshape(diff_down,   1, :, 1, 1)
            diff_4d_up   = ndims(diff_up)    == 4 ? diff_up     : reshape(diff_up,     1, :, 1, 1)
            pert_4d_down = ndims(pert_down)  == 4 ? pert_down   : reshape(pert_down,   1, :, 1, 1)
            pert_4d_up   = ndims(pert_up)    == 4 ? pert_up     : reshape(pert_up,     1, :, 1, 1)

            # diff = ΔF·a_n1 + F2·diff_prev + Δb
            r1_min, r1_max = interval_conv2d_bounds(ΔF, ΔF, n1_4d_down, n1_4d_up, zero_bias, l.stride, l.padding)
            r2_min, r2_max = interval_conv2d_bounds(F2, F2, diff_4d_down, diff_4d_up, zero_bias, l.stride, l.padding)
            diff_down = r1_min .+ r2_min
            diff_up   = r1_max .+ r2_max
            # Broadcast Δb into the spatial dimensions (4th dim = out_channels)
            for oc in 1:length(Δb)
                diff_down[:, :, :, oc] .+= Δb[oc]
                diff_up[:, :, :, oc]   .+= Δb[oc]
            end

            # pert = F2·pert_prev
            rp_min, rp_max = interval_conv2d_bounds(F2, F2, pert_4d_down, pert_4d_up, zero_bias, l.stride, l.padding)
            pert_down = rp_min
            pert_up   = rp_max

            # N1 preact bounds (F1·a_n1 + b1, bias handled inside interval_conv2d_bounds)
            rn_min, rn_max = interval_conv2d_bounds(F1, F1, n1_4d_down, n1_4d_up, Float64.(b1), l.stride, l.padding)
            n1_pre_up_cur   = rn_max
            n1_pre_down_cur = rn_min

        elseif occursin("ReLU", string(typeof(l)))
            # ── Activation conditional-triangle relaxation (n2_org) ──────────
            # diff bounds: z_n2_org - z_n1_org (pre-activation, before clipping)
            # N1 preact bounds: used to form conditional intervals in core_ops.jl
            push!(relu_diff_up_bounds,   copy(diff_up))
            push!(relu_diff_down_bounds, copy(diff_down))
            push!(n1_preact_up_bounds,   copy(n1_pre_up_cur))
            push!(n1_preact_down_bounds, copy(n1_pre_down_cur))

            # ── Perturbation conditional-triangle relaxation (n2_pert) ───────
            # composed bounds: z_n2_pert - z_n1_org = diff + pert (pre-activation)
            # Same a_n1_org binary and N1 preact bounds are reused (see paper eq. 6)
            push!(relu_comp_up_bounds,   diff_up   .+ pert_up)
            push!(relu_comp_down_bounds, diff_down .+ pert_down)

            # Clip intervals through ReLU
            if optimizing_intervals
                # Tighter per-neuron clipping using N1/N2 preact stability
                for i in eachindex(diff_up)
                    l_n1 = n1_pre_down_cur[i]
                    u_n1 = n1_pre_up_cur[i]
                    l_n2 = l_n1 + diff_down[i]
                    u_n2 = u_n1 + diff_up[i]

                    if l_n1 >= 0 && l_n2 >= 0
                        # both active: post-ReLU diff = pre-ReLU diff, keep as-is
                    elseif u_n1 <= 0 && u_n2 <= 0
                        # both inactive: post-ReLU diff = 0
                        diff_up[i] = 0.0
                        diff_down[i] = 0.0
                    else
                        # mixed: use conservative non-expansive clipping
                        diff_up[i] = max(0.0, diff_up[i])
                        diff_down[i] = -max(0.0, -diff_down[i])
                    end
                end
            else
                # Original conservative clipping (non-expansive)
                diff_up   = max.(0.0, diff_up)
                diff_down = .- max.(0.0, .- diff_down)
            end
            pert_up   = max.(0.0, pert_up)
            pert_down = .- max.(0.0, .- pert_down)

            # N1 post-activation bounds
            n1_act_up   = max.(0.0, n1_pre_up_cur)
            n1_act_down = max.(0.0, n1_pre_down_cur)
        end
    end

    # Save final diff bounds (after the last Linear layer, before any output activation).
    # These are the output-layer bounds: N2(x)[k] - N1(x)[k] ∈ [diff_down[k], diff_up[k]].
    # Used by --no_n1_encoding_at_all to replace the entire N1 encoding.
    global output_diff_up_bounds   = vec(copy(Float64.(diff_up)))
    global output_diff_down_bounds = vec(copy(Float64.(diff_down)))
    println("compute_diff_and_comp_bounds: output-layer diff bounds width = $(maximum(output_diff_up_bounds .- output_diff_down_bounds))")

    println("compute_diff_and_comp_bounds: populated $(length(relu_diff_up_bounds)) ReLU layers")
end

# ── N2-only perturbation relaxation bounds ──────────────────────────────
# Computes perturbation intervals through N2 alone (z_n2_pert - z_n2_org)
# and N2 pre-activation bounds, for conditioning N2(x_p) on N2(x).
# Used when --no_n1_binaries_and_relaxtions_only_on_n2 is active.
function compute_n2_pert_relaxation_bounds(nn2, I_pert_up_init, I_pert_down_init)
    global relu_n2pert_up_bounds, relu_n2pert_down_bounds
    global n2_preact_up_bounds, n2_preact_down_bounds

    relu_n2pert_up_bounds   = Array{Float64}[]
    relu_n2pert_down_bounds = Array{Float64}[]
    n2_preact_up_bounds     = Array{Float64}[]
    n2_preact_down_bounds   = Array{Float64}[]

    # Running perturbation interval: z_n2_pert - z_n2_org
    pert_up   = copy(Float64.(I_pert_up_init))
    pert_down = copy(Float64.(I_pert_down_init))

    # N2 post-activation bounds (initialised to input domain [0,1])
    n2_act_up   = ones(Float64,  size(I_pert_up_init))
    n2_act_down = zeros(Float64, size(I_pert_down_init))

    n2_pre_up_cur   = Float64[]
    n2_pre_down_cur = Float64[]

    for (layer_idx, l) in enumerate(nn2.layers)
        if occursin("Flatten", string(typeof(l)))
            if ndims(pert_up) > 1
                pert_up     = vec(pert_up   |> l)
                pert_down   = vec(pert_down |> l)
                n2_act_up   = vec(n2_act_up   |> l)
                n2_act_down = vec(n2_act_down |> l)
            end

        elseif occursin("Linear", string(typeof(l)))
            W2 = Float64.(transpose(l.matrix))
            b2 = Float64.(l.bias)

            # pert: W2 * pert_prev  (bias cancels in the difference)
            rp_min, rp_max = interval_matrix_vector_multiplication(W2, W2, pert_down, pert_up)
            pert_down = rp_min
            pert_up   = rp_max

            # N2 pre-activation bounds: W2 * a_n2 + b2
            rn_min, rn_max = interval_matrix_vector_multiplication(W2, W2, n2_act_down, n2_act_up)
            n2_pre_up_cur   = rn_max .+ b2
            n2_pre_down_cur = rn_min .+ b2

        elseif occursin("Conv", string(typeof(l)))
            F2 = Float64.(l.filter)
            b2 = Float64.(l.bias)
            zero_bias = zeros(Float64, length(b2))

            pert_4d_down = ndims(pert_down) == 4 ? pert_down : reshape(pert_down, 1, :, 1, 1)
            pert_4d_up   = ndims(pert_up)   == 4 ? pert_up   : reshape(pert_up,   1, :, 1, 1)
            n2_4d_down   = ndims(n2_act_down) == 4 ? n2_act_down : reshape(n2_act_down, 1, :, 1, 1)
            n2_4d_up     = ndims(n2_act_up)   == 4 ? n2_act_up   : reshape(n2_act_up,   1, :, 1, 1)

            # pert = F2 * pert_prev
            rp_min, rp_max = interval_conv2d_bounds(F2, F2, pert_4d_down, pert_4d_up, zero_bias, l.stride, l.padding)
            pert_down = rp_min
            pert_up   = rp_max

            # N2 preact bounds
            rn_min, rn_max = interval_conv2d_bounds(F2, F2, n2_4d_down, n2_4d_up, Float64.(b2), l.stride, l.padding)
            n2_pre_up_cur   = rn_max
            n2_pre_down_cur = rn_min

        elseif occursin("ReLU", string(typeof(l)))
            # Save pre-ReLU bounds for relaxation decisions in core_ops.jl
            push!(relu_n2pert_up_bounds,   copy(pert_up))
            push!(relu_n2pert_down_bounds, copy(pert_down))
            push!(n2_preact_up_bounds,     copy(n2_pre_up_cur))
            push!(n2_preact_down_bounds,   copy(n2_pre_down_cur))

            # Clip perturbation intervals through ReLU
            if optimizing_intervals
                # Tighter per-neuron clipping: if both N2(x) and N2(x') neurons
                # are in the same activation state, we can preserve or zero the pert interval.
                for i in eachindex(pert_up)
                    l_n2     = n2_pre_down_cur[i]             # N2(x) preact lower
                    u_n2     = n2_pre_up_cur[i]               # N2(x) preact upper
                    l_n2_p   = l_n2 + pert_down[i]            # N2(x') preact lower
                    u_n2_p   = u_n2 + pert_up[i]              # N2(x') preact upper

                    if l_n2 >= 0 && l_n2_p >= 0
                        # both active: post-ReLU pert = pre-ReLU pert, keep as-is
                    elseif u_n2 <= 0 && u_n2_p <= 0
                        # both inactive: post-ReLU pert = 0
                        pert_up[i]   = 0.0
                        pert_down[i] = 0.0
                    else
                        # mixed: conservative non-expansive clipping
                        pert_up[i]   = max(0.0, pert_up[i])
                        pert_down[i] = -max(0.0, -pert_down[i])
                    end
                end
            else
                # Original conservative clipping (non-expansive)
                pert_up   = max.(0.0, pert_up)
                pert_down = .- max.(0.0, .- pert_down)
            end

            # N2 post-activation bounds
            n2_act_up   = max.(0.0, n2_pre_up_cur)
            n2_act_down = max.(0.0, n2_pre_down_cur)
        end
    end

    println("compute_n2_pert_relaxation_bounds: populated $(length(relu_n2pert_up_bounds)) ReLU layers")
end

# ── Output-layer diff bounds only (lightweight) ─────────────────────────
# Computes N2(x)[k] - N1(x)[k] bounds at the output layer using interval
# arithmetic on weight differences. Does NOT compute per-ReLU bounds.
# Used by --no_n1_encoding_at_all when compute_diff_and_comp_bounds is not called.
function compute_output_diff_bounds_only(nn1, nn2)
    global output_diff_up_bounds, output_diff_down_bounds

    # Determine input shape from I_pert_prev_up (set by perturbation-specific function),
    # falling back to layer-based detection for standalone use.
    global I_pert_prev_up
    if length(I_pert_prev_up) > 0
        input_shape = size(I_pert_prev_up)
    else
        # Fallback: detect from first non-Flatten layer
        sample_layer = nn1.layers[1]
        if occursin("Flatten", string(typeof(sample_layer)))
            sample_layer = nn1.layers[2]
        end
        if occursin("Linear", string(typeof(sample_layer)))
            n_in = size(sample_layer.matrix, 1)
            input_shape = (n_in,)
        else
            # Conv: need (1, w, h, k) — estimate from filter's input channels
            # and assume square input (MNIST: 28x28, CIFAR: 32x32)
            F = sample_layer.filter
            k_in = size(F, 3)
            # Use common dataset sizes; for robustness, caller should set I_pert_prev_up
            w_guess = k_in == 1 ? 28 : 32
            input_shape = (1, w_guess, w_guess, k_in)
        end
    end

    act_up   = ones(Float64, input_shape)
    act_down = zeros(Float64, input_shape)
    diff_up   = zeros(Float64, input_shape)
    diff_down = zeros(Float64, input_shape)

    pre_up   = copy(act_up)
    pre_down = copy(act_down)

    for (layer_idx, l) in enumerate(nn1.layers)
        l2 = nn2.layers[layer_idx]

        if occursin("Flatten", string(typeof(l)))
            if ndims(diff_up) > 1
                diff_up   = vec(diff_up   |> l)
                diff_down = vec(diff_down |> l)
                act_up    = vec(act_up    |> l)
                act_down  = vec(act_down  |> l)
            end

        elseif occursin("Linear", string(typeof(l)))
            W1 = Float64.(transpose(l.matrix))
            W2 = Float64.(transpose(l2.matrix))
            b1 = Float64.(l.bias)
            b2 = Float64.(l2.bias)
            ΔW = W2 - W1
            Δb = b2 - b1

            r1_min, r1_max = interval_matrix_vector_multiplication(ΔW, ΔW, act_down, act_up)
            r2_min, r2_max = interval_matrix_vector_multiplication(W2, W2, diff_down, diff_up)
            diff_down = r1_min .+ r2_min .+ Δb
            diff_up   = r1_max .+ r2_max .+ Δb

            # N1 preact for post-activation bounds
            rn_min, rn_max = interval_matrix_vector_multiplication(W1, W1, act_down, act_up)
            pre_up   = rn_max .+ b1
            pre_down = rn_min .+ b1

        elseif occursin("Conv", string(typeof(l)))
            F1 = Float64.(l.filter);  F2 = Float64.(l2.filter)
            b1 = Float64.(l.bias);    b2 = Float64.(l2.bias)
            ΔF = F2 - F1;             Δb = b2 - b1
            zero_bias = zeros(Float64, length(b1))

            d4_down = ndims(diff_down) == 4 ? diff_down : reshape(diff_down, 1, :, 1, 1)
            d4_up   = ndims(diff_up)   == 4 ? diff_up   : reshape(diff_up,   1, :, 1, 1)
            a4_down = ndims(act_down)  == 4 ? act_down  : reshape(act_down,  1, :, 1, 1)
            a4_up   = ndims(act_up)    == 4 ? act_up    : reshape(act_up,    1, :, 1, 1)

            r1_min, r1_max = interval_conv2d_bounds(ΔF, ΔF, a4_down, a4_up, zero_bias, l.stride, l.padding)
            r2_min, r2_max = interval_conv2d_bounds(F2, F2, d4_down, d4_up, zero_bias, l.stride, l.padding)
            diff_down = r1_min .+ r2_min
            diff_up   = r1_max .+ r2_max
            for oc in 1:length(Δb)
                diff_down[:, :, :, oc] .+= Δb[oc]
                diff_up[:, :, :, oc]   .+= Δb[oc]
            end

            rn_min, rn_max = interval_conv2d_bounds(F1, F1, a4_down, a4_up, Float64.(b1), l.stride, l.padding)
            pre_up   = rn_max
            pre_down = rn_min

        elseif occursin("ReLU", string(typeof(l)))
            # Clip diff through ReLU (non-expansive)
            diff_up   = max.(0.0, diff_up)
            diff_down = .- max.(0.0, .- diff_down)
            # N1 post-activation bounds
            act_up   = max.(0.0, pre_up)
            act_down = max.(0.0, pre_down)
        end
    end

    output_diff_up_bounds   = vec(copy(Float64.(diff_up)))
    output_diff_down_bounds = vec(copy(Float64.(diff_down)))
    println("compute_output_diff_bounds_only: output-layer diff bounds width = $(maximum(output_diff_up_bounds .- output_diff_down_bounds))")
end
