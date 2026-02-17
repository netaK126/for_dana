# Interval bound propagation for the transfer proof.
#
# Same approach as lucid_delta_diff_with_perturbation:
# - Uses globals: I_z_prev_up/down, I_z_prev_up/down_perturbation,
#   all_bounds_of_original/perturbation
# - Initialized in get_perturbation_specific_keys_linf_transfer()
# - Propagation formula: I_w_up = W2^T - W1^T, I_w_down = 0, w1 = W1^T
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

            # Same formula as lucid HyperlinearDepsIn:
            # I_w_up   = W2^T - W1^T  (weight difference)
            # I_w_down = W1^T - W1^T = 0
            # w1       = W1^T
            I_w_up   = Float64.(transpose(l2.matrix)) - Float64.(transpose(l.matrix))
            I_w_down = Float64.(transpose(l.matrix))  - Float64.(transpose(l.matrix))
            w1       = Float64.(transpose(l.matrix))

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

            result_min_1, result_max_1 = interval_matrix_vector_multiplication(I_w_down, I_w_up, z_prev_down, z_prev_up)
            result_min_2, result_max_2 = interval_matrix_vector_multiplication(w1, w1, I_z_prev_down_to_use, I_z_prev_up_to_use)
            I_z_down_in = result_min_1 .+ result_min_2
            I_z_up_in   = result_max_1 .+ result_max_2

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
    # Reset to initial values
    if version == "org"
        I_z_prev_up_local   = zeros(Float64, length(all_bounds_of_original[1][1]))
        I_z_prev_down_local = zeros(Float64, length(all_bounds_of_original[1][1]))
    elseif version == "perturbation"
        I_z_prev_up_local   = zeros(Float64, length(all_bounds_of_perturbation[1][1]))
        I_z_prev_down_local = zeros(Float64, length(all_bounds_of_perturbation[1][1]))
    end

    for (layer_idx, l) in enumerate(nn1.layers)
        if occursin("Flatten", string(typeof(l)))
            if ndims(I_z_prev_up_local) > 1
                I_z_prev_up_local   = I_z_prev_up_local |> l
                I_z_prev_down_local = I_z_prev_down_local |> l
            end

        elseif occursin("Linear", string(typeof(l)))
            l2 = nn2.layers[layer_idx]

            I_w_up   = Float64.(transpose(l2.matrix)) - Float64.(transpose(l.matrix))
            I_w_down = Float64.(transpose(l.matrix))  - Float64.(transpose(l.matrix))
            w1       = Float64.(transpose(l.matrix))

            if version == "org"
                z_prev_up   = all_bounds_of_original[layer_cnt+1][1]
                z_prev_down = all_bounds_of_original[layer_cnt+1][2]
            elseif version == "perturbation"
                z_prev_up   = all_bounds_of_perturbation[layer_cnt+1][1]
                z_prev_down = all_bounds_of_perturbation[layer_cnt+1][2]
            end

            result_min_1, result_max_1 = interval_matrix_vector_multiplication(I_w_down, I_w_up, z_prev_down, z_prev_up)
            result_min_2, result_max_2 = interval_matrix_vector_multiplication(w1, w1, I_z_prev_down_local, I_z_prev_up_local)
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
                @constraint(m, av[vec_2] .<= av[vec_1] .+ I_z_prev_up_local)
                @constraint(m, av[vec_2] .>= av[vec_1] .+ I_z_prev_down_local)
                constraints_added += 2 * length(vec_1)
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
                @constraint(m, av[vec_pert] .<= av[vec_clean] .+ I_pert_up_local)
                @constraint(m, av[vec_pert] .>= av[vec_clean] .+ I_pert_down_local)
                constraints_added += 2 * length(vec_clean)
            end
        end
    end

    return constraints_added
end
