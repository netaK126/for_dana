
function mip_reset()
    neurons_names.neuron = 0
    neurons_names.layer = 0
    first_mip_solution.solution = -1.0
    first_mip_solution.time = 0.0
end

function mip_set_delta_property(m, perturbation, d)
    if perturbation != "max"
        set_max_indexes(m, d[:v_out_p], d[:TargetIndex])
    end
    (maximum_target_var, nontarget_vars) = get_vars_for_max_index(d[:v_out], d[:SourceIndex])
    maximum_nontarget_var = maximum_ge(nontarget_vars)
    delta = @variable(m)
    @constraint(m, delta == maximum_target_var - maximum_nontarget_var)
    @objective(m, Max, delta)
end

function mip_set_attr(m, perturbation, d, timout)
    if (perturbation == "contrast")
        set_optimizer_attribute(m, "NonConvex", 2)
    end
    set_optimizer_attribute(m, "MIPFocus", 3)
    set_optimizer_attribute(m, "Cutoff", d[:suboptimal_solution])
    global Threads_num
    set_optimizer_attribute(m, "Threads", Threads_num)
    set_optimizer_attribute(m, "TimeLimit", timout)
    set_optimizer_attribute(m, "MIPGap", 0.01)
end

function mip_log(m, d)
    d[:SolveStatus] = JuMP.termination_status(m)
    d[:SolveTime] = JuMP.solve_time(m)
    incumbent_obj = 0
    try
        incumbent_obj = JuMP.objective_value(m)
    catch e
        println("no incumbent_obj")
    end
    d[:incumbent_obj] = incumbent_obj
    best_bound = 0
    try
        best_bound = JuMP.objective_bound(m)
    catch e
        println("WARNING: could not query objective_bound (status: $(d[:SolveStatus]))")
    end
    d[:best_bound] = best_bound
    d[:solve_time] = JuMP.solve_time(m)
    d[:first_mip_solution] = first_mip_solution.solution
    d[:time_for_first_mip_solution] = first_mip_solution.time
    println(string(incumbent_obj)*"  "*string(d[:best_bound])*"  "*string(d[:solve_time]))
    try
        d[:v_in_p] = (JuMP.value.(d[:v_in_p]))
        d[:v_in] = (JuMP.value.(d[:v_in]))
        if d[:Perturbation] != "None"
            d[:Perturbation] = (JuMP.value.(d[:Perturbation]))
        end
    catch e
        d[:v_in_p] = 0
        d[:v_in] = 0
        d[:Perturbation] = 0
    end
end

function mip_reuse_bounds()
    reuse_bounds_conf.is_reuse_bounds_and_deps = true
    reuse_bounds_conf.reusable_indexes = 1
end

# ============================================================
# Transfer proof: define confidence margin C(N, x, c) as a
# JuMP variable using big-M encoding for the max over
# non-target classes.
# C(N, x, c) = N(x)[c] - max_{k≠c} N(x)[k]
# ============================================================
function define_conf!(m, d, c, key, name)
    max_num = 1e6
    conf = @variable(m, base_name=name)
    max_kk = @variable(m, base_name=name*"_max_kk")
    @constraint(m, conf == d[key][c] - max_kk)
    n_classes = length(d[key])
    a_conf = Dict()
    for i in 1:n_classes
        if i == c
            continue
        end
        a_conf[i] = @variable(m, binary = true, base_name=name*"_bin_"*string(i))
    end
    @constraint(m, sum(a_conf[i] for i in keys(a_conf)) == 1)
    for i in 1:n_classes
        if i == c
            continue
        end
        @constraint(m, max_kk >= d[key][i])
        @constraint(m, max_kk <= d[key][i] + max_num * (1 - a_conf[i]))
    end
    return conf
end

# ============================================================
# Transfer proof objective:
#   max delta_diff  s.t.
#     C(N1, x, c)  >=  delta_1 + 1e-8
#     C(N2, x, c) - C(N1, x, c)  >=  delta_diff
#     delta_diff  >=  0
#     C(N2, f(x,ε), c_pert) - C(N1, f(x,ε), c_pert)  <=  -1e-5
#
# c_tag_mode=true  → c_pert = c_tag  (untargeted)
# c_tag_mode=false → c_pert = c_target (targeted)
# ============================================================
function mip_set_transfer_property(m, d, delta_1, c_tag, c_target,
    c_tag_mode, n1_p_mode, n2_fewer_binars_encoding, delta_diff_positive)
    # Confidence margins on clean input (both measured for source class c_tag)
    conf_n1_x = define_conf!(m, d, c_tag, :v_out_n1, "conf_n1_x")
    conf_n2_x = define_conf!(m, d, c_tag, :v_out_n2, "conf_n2_x")

    # Confidence margins on perturbed input.
    # conf_n1_xp is only needed (and d[:v_out_n1_p] only encoded) when n1_p_mode is on.
    c_pert = c_tag_mode ? c_tag : c_target
    if n1_p_mode
        conf_n1_xp = define_conf!(m, d, c_pert, :v_out_n1_p, "conf_n1_xp")
    end

    # conf_n2_xp is always needed except when n2_fewer_binars_encoding handles
    # the perturbed constraint directly via per-class inequalities.
    if !n2_fewer_binars_encoding || n1_p_mode || c_tag_mode
        conf_n2_xp = define_conf!(m, d, c_pert, :v_out_n2_p, "conf_n2_xp")
    end

    # Constraint (1): N1 is confident on clean input
    @constraint(m, conf_n1_x >= delta_1 + 1e-3)

    # Constraint (2)+(3): delta_diff = C(N2,x,c) - C(N1,x,c) >= 0
    delta_diff = @variable(m, base_name="delta_diff")
    @constraint(m, delta_diff == conf_n2_x - conf_n1_x)
    if delta_diff_positive
        @constraint(m, delta_diff >= 0)
    else
        @constraint(m, conf_n2_x >= 0)
    end
    margin = 1e-3
    # Constraint (4): confidence gap flips under perturbation

    if c_tag_mode 
        if n1_p_mode
            @constraint(m, conf_n2_xp - conf_n1_xp <= -margin)
        else
            @constraint(m, conf_n2_xp <= -margin)
        end

    else # c_target is on
        if n1_p_mode
            @constraint(m, conf_n2_xp - conf_n1_xp >= margin)
        else
            if n2_fewer_binars_encoding
                for i in eachindex(d[:v_out_n2_p])
                    if i == c_target
                        continue
                    end
                    @constraint(m, d[:v_out_n2_p][c_target] - d[:v_out_n2_p][i] >= margin)
                end
            else
                @constraint(m, conf_n2_xp >= margin)
            end
        end
    end

    # Objective: maximize delta_diff
    @objective(m, Max, delta_diff)
end

# ============================================================
# Transfer proof WITHOUT encoding N1 (--no_n1_encoding_at_all)
#
# Replaces conf(N1,x,c) >= delta_1 with interval-bounded
# constraints on N2 outputs, and uses a lower bound on
# conf(N1,x,c) for the delta_diff objective.
#
# Output diff bounds: N2(x)[k] - N1(x)[k] ∈ [d_lo[k], d_hi[k]]
# stored in global output_diff_down_bounds / output_diff_up_bounds.
# ============================================================
function mip_set_transfer_property_no_n1(m, d, delta_1, c_tag, c_target,
    c_tag_mode, n2_fewer_binars_encoding, delta_diff_positive)

    global output_diff_up_bounds, output_diff_down_bounds
    d_hi = output_diff_up_bounds    # N2(x)[k] - N1(x)[k] upper bound
    d_lo = output_diff_down_bounds  # N2(x)[k] - N1(x)[k] lower bound

    n_classes = length(d[:v_out_n2])
    println("  output diff bounds: max width = $(maximum(d_hi .- d_lo))")
    println("  output diff bounds per class:")
    for k in 1:n_classes
        println("    class $k: d_lo=$(d_lo[k]), d_hi=$(d_hi[k]), width=$(d_hi[k]-d_lo[k])")
    end

    # ── Constraint (1): N1 confidence via interval bounds on N2 outputs ──
    # For all k ≠ c_tag:
    #   N2(x)[c_tag] - N2(x)[k] >= delta_1 - d_hi[k] + d_lo[c_tag]
    for k in 1:n_classes
        if k == c_tag
            continue
        end
        rhs = delta_1 + 1e-3 - d_hi[k] + d_lo[c_tag]
        println("    N1 conf constraint k=$k: N2[$c_tag]-N2[$k] >= $rhs")
        @constraint(m, d[:v_out_n2][c_tag] - d[:v_out_n2][k] >= rhs)
    end

    # ── Confidence of N2 on clean input (uses binary encoding for max) ──
    conf_n2_x = define_conf!(m, d, c_tag, :v_out_n2, "conf_n2_x")

    # ── delta_diff upper-bounded by interval-derived constraints ──
    # For each k ≠ c_tag, conf_n1_x >= expr_k where
    #   expr_k = (N2(x)[c_tag] - N2(x)[k]) + (d_lo[k] - d_hi[c_tag])
    # So delta_diff = conf_n2_x - conf_n1_x <= conf_n2_x - expr_k for each k.
    # These UPPER BOUND constraints prevent unboundedness.
    delta_diff = @variable(m, base_name="delta_diff")
    for k in 1:n_classes
        if k == c_tag
            continue
        end
        expr_k = (d[:v_out_n2][c_tag] - d[:v_out_n2][k]) + (d_lo[k] - d_hi[c_tag])
        @constraint(m, delta_diff <= conf_n2_x - expr_k)
    end
    if delta_diff_positive
        @constraint(m, delta_diff >= 0)
    else
        @constraint(m, conf_n2_x >= 0)
    end

    # ── Constraint (4): N2 is fooled on perturbed input ──
    c_pert = c_tag_mode ? c_tag : c_target
    margin = 1e-3

    if c_tag_mode
        conf_n2_xp = define_conf!(m, d, c_pert, :v_out_n2_p, "conf_n2_xp")
        @constraint(m, conf_n2_xp <= -margin)
    else
        if n2_fewer_binars_encoding
            for i in eachindex(d[:v_out_n2_p])
                if i == c_target
                    continue
                end
                @constraint(m, d[:v_out_n2_p][c_target] - d[:v_out_n2_p][i] >= margin)
            end
        else
            conf_n2_xp = define_conf!(m, d, c_pert, :v_out_n2_p, "conf_n2_xp")
            @constraint(m, conf_n2_xp >= margin)
        end
    end

    # Objective: maximize delta_diff (upper bound on true delta_diff)
    @objective(m, Max, delta_diff)
end

# ============================================================
# No-N1-encoding + encode N1's last layer:
#   Same as no_n1 mode, but creates interval-bounded variables
#   for N1's last hidden layer and encodes the final linear
#   layer exactly → exact conf_n1_x and delta_diff.
# ============================================================
function mip_set_transfer_property_n1_last_layer(m, d, delta_1, c_tag, c_target,
    c_tag_mode, n2_fewer_binars_encoding, delta_diff_positive, nn1)

    global n1_last_hidden_up, n1_last_hidden_down
    global last_hidden_diff_up, last_hidden_diff_down

    n_hidden = length(n1_last_hidden_up)
    println("  encode_n1_last_layer: creating $n_hidden bounded hidden vars")
    println("  N1 last hidden bounds width: max=$(maximum(n1_last_hidden_up .- n1_last_hidden_down))")

    # ── Create interval-bounded variables for N1's last hidden layer ──
    h_n1 = [@variable(m, lower_bound = n1_last_hidden_down[i],
                          upper_bound = n1_last_hidden_up[i],
                          base_name = "h_n1_last_$i") for i in 1:n_hidden]

    # ── Link h_n1 to N2's last hidden layer via difference bounds ──
    # Prevents h_n1 from being chosen independently of the input x.
    # For all i: Δh_down[i] <= h_n2[i] - h_n1[i] <= Δh_up[i]
    v_n2_hidden = d[:v_n2_last_hidden]
    if v_n2_hidden !== nothing
        println("  Adding $n_hidden linking constraints (h_n2 - h_n1 ∈ [Δh_dn, Δh_up])")
        println("  Hidden diff width: max=$(maximum(last_hidden_diff_up .- last_hidden_diff_down))")
        for i in 1:n_hidden
            @constraint(m, v_n2_hidden[i] - h_n1[i] >= last_hidden_diff_down[i])
            @constraint(m, v_n2_hidden[i] - h_n1[i] <= last_hidden_diff_up[i])
        end
    else
        println("  WARNING: v_n2_last_hidden not available, h_n1 is unlinked (loose bounds)")
    end

    # ── Find N1's last Linear layer and compute N1 output exactly ──
    last_linear = nothing
    for l in nn1.layers
        if occursin("Linear", string(typeof(l)))
            last_linear = l
        end
    end
    W = Float64.(transpose(last_linear.matrix))  # (output_dim × input_dim)
    b = Float64.(last_linear.bias)
    n_classes = length(b)
    println("  N1 last layer: $(size(last_linear.matrix,1)) -> $n_classes")

    # N1 output: v_out_n1[k] = W[k,:] * h_n1 + b[k]
    v_out_n1 = [@variable(m, base_name = "n1_out_$k") for k in 1:n_classes]
    for k in 1:n_classes
        @constraint(m, v_out_n1[k] == sum(W[k, j] * h_n1[j] for j in 1:n_hidden) + b[k])
    end

    # ── Confidence of N1 on clean input (exact, with binaries for max) ──
    # Store in d temporarily so define_conf! can access it
    d[:v_out_n1_last] = v_out_n1
    conf_n1_x = define_conf!(m, d, c_tag, :v_out_n1_last, "conf_n1_x")

    # ── N1 confidence >= delta_1 ──
    @constraint(m, conf_n1_x >= delta_1 + 1e-3)

    # ── Confidence of N2 on clean input (uses binary encoding for max) ──
    conf_n2_x = define_conf!(m, d, c_tag, :v_out_n2, "conf_n2_x")

    # ── delta_diff: exact (not just upper-bounded) ──
    delta_diff = @variable(m, base_name="delta_diff")
    @constraint(m, delta_diff == conf_n2_x - conf_n1_x)
    if delta_diff_positive
        @constraint(m, delta_diff >= 0)
    else
        @constraint(m, conf_n2_x >= 0)
    end

    # ── Constraint: N2 is fooled on perturbed input ──
    c_pert = c_tag_mode ? c_tag : c_target
    margin = 1e-3

    if c_tag_mode
        conf_n2_xp = define_conf!(m, d, c_pert, :v_out_n2_p, "conf_n2_xp")
        @constraint(m, conf_n2_xp <= -margin)
    else
        if n2_fewer_binars_encoding
            for i in eachindex(d[:v_out_n2_p])
                if i == c_target
                    continue
                end
                @constraint(m, d[:v_out_n2_p][c_target] - d[:v_out_n2_p][i] >= margin)
            end
        else
            conf_n2_xp = define_conf!(m, d, c_pert, :v_out_n2_p, "conf_n2_xp")
            @constraint(m, conf_n2_xp >= margin)
        end
    end

    # Objective: maximize delta_diff (now exact, not upper-bounded)
    @objective(m, Max, delta_diff)
end

function mip_set_attr_transfer(m, timout, suboptimal_solution=0, delta_diff_positive=false)
    set_optimizer_attribute(m, "MIPFocus", 3)
    if delta_diff_positive
        set_optimizer_attribute(m, "Cutoff", suboptimal_solution)
    elseif suboptimal_solution != 0
        set_optimizer_attribute(m, "Cutoff", suboptimal_solution)
    end
    set_optimizer_attribute(m, "Threads", 32)
    set_optimizer_attribute(m, "TimeLimit", timout)
    set_optimizer_attribute(m, "MIPGap", 0.01)
end
