
function mip_reset()
    neurons_names.neuron = 0
    neurons_names.layer = 0
    first_mip_solution.solution = -1.0
    first_mip_solution.time = 0.0
    # Technique 4 (SibGate): clear the per-neuron MIP-state cache so a
    # stale (preact, l, u, x_rect) from the previous c_target build can't
    # leak into the current one. apply_sibgate_constraints! reads from
    # this cache, so it must be repopulated freshly for each MIP build.
    clear_n2_relu_state!()
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

# ── TwoSafe confidence-based robustness: decision mode ────────────────────
# Chosen when the global `twosafe_property != "none"` (set from run.jl under
# --internet_nets_benchmarks). Instead of MAXIMIZING the confidence margin
# (mip_set_delta_property), we DECIDE the confidence-based robustness property
# of Athavale et al. (Def 2.10 asymmetric / Def 3.1 symmetric) at a fixed
# (ε, τ): a feasible point of the MIP is a counterexample (⇒ NOT robust); a
# proven-infeasible MIP means the property holds for this (source,target) pair
# (⇒ robust); a timeout with no feasible point is inconclusive.

function add_softmax_confidence_constraint(m, logits, class_index, tau_val; gmin = -1e4, tol = 1e-6)
    # conf_c = softmax_c(logits) > τ. Because `class_index` is fixed as the
    # argmax (set_max_indexes), every g_j = f_j - f_c ≤ 0, so e^{g_j} ∈ (0,1]
    # and the equivalent condition is overflow-free:
    #   softmax_c > τ  ⟺  e^{f_c}/Σ_j e^{f_j} > τ  ⟺  Σ_{j≠c} e^{f_j-f_c} < (1-τ)/τ.
    @assert 0.0 < tau_val < 1.0 "tau must be in (0,1); got $tau_val"
    n = length(logits)
    egs = Vector{JuMP.VariableRef}()
    for j in 1:n
        j == class_index && continue
        g = @variable(m, lower_bound = gmin, upper_bound = 0.0)
        @constraint(m, g == logits[j] - logits[class_index])
        eg = @variable(m, lower_bound = 0.0, upper_bound = 1.0)
        @constraint(m, eg == exp(g))   # Gurobi general function constraint (exact under FuncNonlinear=1)
        push!(egs, eg)
    end
    @constraint(m, sum(egs) <= (1.0 - tau_val) / tau_val - tol)
    return nothing
end

function mip_set_twosafe_property(m, d, tau_val, property)
    # Fix both classifications: original x ⇒ source class, perturbed x' ⇒ target.
    set_max_indexes(m, d[:v_out], d[:SourceIndex])
    set_max_indexes(m, d[:v_out_p], d[:TargetIndex])
    # conf(f(x)) > τ on the original copy (Def 2.10, asymmetric).
    add_softmax_confidence_constraint(m, d[:v_out], d[:SourceIndex][1], tau_val)
    if property == "symmetric"
        # Also require the perturbed point to be high-confidence (Def 3.1).
        add_softmax_confidence_constraint(m, d[:v_out_p], d[:TargetIndex][1], tau_val)
    elseif property != "asymmetric"
        error("twosafe_property must be \"asymmetric\" or \"symmetric\"; got \"$property\"")
    end
    # Feasibility problem: any feasible point is a counterexample.
    @objective(m, Max, 0)
    return nothing
end

function mip_set_attr_twosafe(m, timout)
    # Decision mode: stop at the first counterexample (feasible ⇒ NOT robust);
    # otherwise prove infeasibility (⇒ robust) or hit the time limit. No Cutoff
    # (that belongs to the optimization objective); exact softmax exponentials.
    global Threads_num
    global gurobi_seed
    set_optimizer_attribute(m, "MIPFocus", 1)        # prioritise finding feasible points
    set_optimizer_attribute(m, "SolutionLimit", 1)   # the first counterexample is enough
    set_optimizer_attribute(m, "FuncNonlinear", 1)   # exact exp (not piecewise)
    set_optimizer_attribute(m, "Threads", Threads_num)
    set_optimizer_attribute(m, "TimeLimit", timout)
    set_optimizer_attribute(m, "Seed", gurobi_seed)
    return nothing
end

function twosafe_verdict(m)
    st = JuMP.termination_status(m)
    if JuMP.primal_status(m) == MOI.FEASIBLE_POINT
        return "NOT_ROBUST"      # a counterexample satisfying every constraint exists
    elseif st == MOI.INFEASIBLE || st == MOI.INFEASIBLE_OR_UNBOUNDED
        return "ROBUST"          # property holds for this class pair
    else
        return "INCONCLUSIVE"    # e.g. TIME_LIMIT before any counterexample
    end
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
    global gurobi_seed
    set_optimizer_attribute(m, "Seed", gurobi_seed)
    # cnn2 only: the PGD partial warm-start triggers a completion sub-MIP whose
    # default node budget explodes on this large model (~1h/target, no incumbent).
    # Bound it so Gurobi stops repairing the start and gets to the real solve.
    # Every other architecture is left exactly as before (no attribute set).
    global model_name
    if model_name == "cnn2"
        set_optimizer_attribute(m, "StartNodeLimit", 100)
    end
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
    c_tag_mode, n1_p_mode, n2_fewer_binars_encoding)
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
    @constraint(m, conf_n2_x >= 0)
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
    c_tag_mode, n2_fewer_binars_encoding)

    global output_diff_up_bounds, output_diff_down_bounds, cap_delta_diff
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

    # ── delta_diff via conf_n1_est (sound upper bound on delta_diff_exact) ──
    # conf_n1_est = min_k margin_est_k via C-1 big-M argmax encoding.
    #
    # margin_est_k = (N2[c]-N2[k]) + (d_lo[k] - d_hi[c])
    #              <= N1[c] - N1[k]   (sound lower bound on per-class margin)
    # conf_n1_est  = min_{k≠c} margin_est_k <= conf_n1_true
    # => delta_diff = conf_n2 - conf_n1_est >= delta_diff_exact   (sound)
    max_M = 1e6
    conf_n1_est = @variable(m, base_name="conf_n1_est")
    b_n1 = Dict{Int, Any}()
    for k in 1:n_classes
        if k == c_tag; continue; end
        b_n1[k] = @variable(m, binary=true, base_name="conf_n1_est_bin_$k")
    end
    @constraint(m, sum(b_n1[k] for k in keys(b_n1)) == 1)
    for k in 1:n_classes
        if k == c_tag; continue; end
        margin_est_k = (d[:v_out_n2][c_tag] - d[:v_out_n2][k]) + (d_lo[k] - d_hi[c_tag])
        @constraint(m, conf_n1_est <= margin_est_k)
        @constraint(m, conf_n1_est >= margin_est_k - max_M * (1 - b_n1[k]))
    end
    delta_diff = @variable(m, base_name="delta_diff")
    @constraint(m, delta_diff == conf_n2_x - conf_n1_est)
    @constraint(m, conf_n2_x >= 0)

    # ── Optional scalar cap on delta_diff (tightens LP relaxation) ──
    if cap_delta_diff
        delta_up_scalar = -Inf
        for k in 1:n_classes
            if k == c_tag; continue; end
            delta_up_scalar = max(delta_up_scalar, d_hi[c_tag] - d_lo[k])
        end
        println("  cap_delta_diff: delta_diff <= $delta_up_scalar")
        @constraint(m, delta_diff <= delta_up_scalar)
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
# Transfer proof WITHOUT encoding N2(x') (--no_n2_xp_encoding)
#
# N1(x) and N2(x) are encoded exactly → exact conf_n1_x,
# conf_n2_x, and delta_diff.
# N2(x') is replaced by interval-bounded output variables:
#   N2(x')[k] ∈ [N2(x)[k] + p_lo[k], N2(x)[k] + p_up[k]]
# where p_lo/p_up are perturbation bounds through N2:
#   N2(x')[k] - N2(x)[k] ∈ [p_lo[k], p_up[k]]
# stored in global output_n2_pert_down / output_n2_pert_up.
#
# These bounds are computed by compute_diff_and_comp_bounds()
# or compute_diff_bounds_zonotope() (when --use_zonotope).
# ============================================================
function mip_set_transfer_property_no_n2_xp(m, d, delta_1, c_tag, c_target,
    c_tag_mode, n1_p_mode, n2_fewer_binars_encoding)

    global output_n2_pert_up, output_n2_pert_down
    p_up = output_n2_pert_up    # N2(x')[k] - N2(x)[k] upper bound
    p_lo = output_n2_pert_down  # N2(x')[k] - N2(x)[k] lower bound

    n_classes = length(d[:v_out_n2])
    println("  N2 pert bounds: max width = $(maximum(p_up .- p_lo))")
    println("  N2 pert bounds per class:")
    for k in 1:n_classes
        println("    class $k: p_lo=$(p_lo[k]), p_up=$(p_up[k]), width=$(p_up[k]-p_lo[k])")
    end

    # ── Confidence margins on clean input (exact binary encoding) ──
    conf_n1_x = define_conf!(m, d, c_tag, :v_out_n1, "conf_n1_x")
    conf_n2_x = define_conf!(m, d, c_tag, :v_out_n2, "conf_n2_x")

    # ── N1 confident on clean input ──
    @constraint(m, conf_n1_x >= delta_1 + 1e-3)

    # ── Exact delta_diff = C(N2,x,c) - C(N1,x,c) ──
    delta_diff = @variable(m, base_name="delta_diff")
    @constraint(m, delta_diff == conf_n2_x - conf_n1_x)
    @constraint(m, conf_n2_x >= 0)

    # ── N1(x') confidence (only when n1_p_mode is on) ──
    c_pert = c_tag_mode ? c_tag : c_target
    margin = 1e-3

    if n1_p_mode
        conf_n1_xp = define_conf!(m, d, c_pert, :v_out_n1_p, "conf_n1_xp")
    end

    # ── Create interval-bounded output variables for N2(x') ──
    # N2(x')[k] ∈ [N2(x)[k] + p_lo[k], N2(x)[k] + p_up[k]]
    v_n2_xp_out = [@variable(m, base_name="n2_xp_out_$k") for k in 1:n_classes]
    for k in 1:n_classes
        @constraint(m, v_n2_xp_out[k] >= d[:v_out_n2][k] + p_lo[k])
        @constraint(m, v_n2_xp_out[k] <= d[:v_out_n2][k] + p_up[k])
    end
    d[:v_out_n2_p_interval] = v_n2_xp_out

    # ── Confidence margin of N2 on perturbed input (using interval-bounded outputs) ──
    if !n2_fewer_binars_encoding || n1_p_mode || c_tag_mode
        conf_n2_xp = define_conf!(m, d, c_pert, :v_out_n2_p_interval, "conf_n2_xp")
    end

    # ── Constraint: confidence gap flips under perturbation ──
    if c_tag_mode
        if n1_p_mode
            @constraint(m, conf_n2_xp - conf_n1_xp <= -margin)
        else
            @constraint(m, conf_n2_xp <= -margin)
        end
    else
        if n1_p_mode
            @constraint(m, conf_n2_xp - conf_n1_xp >= margin)
        else
            if n2_fewer_binars_encoding
                for i in eachindex(v_n2_xp_out)
                    if i == c_target
                        continue
                    end
                    @constraint(m, v_n2_xp_out[c_target] - v_n2_xp_out[i] >= margin)
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
# No-N1-encoding + encode N1's last layer:
#   Creates interval-bounded variables for N1's last hidden layer,
#   linked to N2's encoded hidden layer via diff bounds.
#
#   Uses argmax encoding: conf_n1 = min_k m_k via C-1 binaries,
#   then delta_diff == conf_n2_x - conf_n1_x (equality, no objective cap).
# ============================================================
function mip_set_transfer_property_n1_last_layer(m, d, delta_1, c_tag, c_target,
    c_tag_mode, n2_fewer_binars_encoding, nn1, prune_tol::Float64=0.0,
    adaptive_prune_budget::Float64=0.0)

    global n1_last_hidden_up, n1_last_hidden_down
    global last_hidden_diff_up, last_hidden_diff_down

    n_hidden = length(n1_last_hidden_up)
    a_dn = n1_last_hidden_down
    a_up = n1_last_hidden_up

    # ── Get N1's last Linear layer weights (needed before pruning for adaptive mode) ──
    last_linear = nothing
    for l in nn1.layers
        if occursin("Linear", string(typeof(l)))
            last_linear = l
        end
    end
    W = Float64.(transpose(last_linear.matrix))  # (n_classes × n_hidden)
    b_vec = Float64.(last_linear.bias)
    n_classes = length(b_vec)

    # ── Constraint (1): N1 confidence via interval bounds on N2 outputs (necessary) ──
    # Same as in mip_set_transfer_property_no_n1: ensures S_exact ⊆ S.
    global output_diff_up_bounds, output_diff_down_bounds, cap_delta_diff
    d_hi_out = output_diff_up_bounds
    d_lo_out = output_diff_down_bounds
    for k in 1:n_classes
        if k == c_tag; continue; end
        rhs = delta_1 + 1e-3 - d_hi_out[k] + d_lo_out[c_tag]
        @constraint(m, d[:v_out_n2][c_tag] - d[:v_out_n2][k] >= rhs)
    end

    # ── Classify neurons: prune by fixed threshold or adaptive sensitivity budget ──
    is_pruned = falses(n_hidden)
    n_pruned = 0
    if adaptive_prune_budget > 0
        # Sensitivity-based pruning: score each neuron by worst-case margin error
        # loss_j = max_k |W[c,j] - W[k,j]| * (a_up[j] - a_dn[j])
        scores = zeros(n_hidden)
        for j in 1:n_hidden
            width_j = a_up[j] - a_dn[j]
            max_w = maximum([abs(W[c_tag, j] - W[k, j]) for k in 1:n_classes if k != c_tag])
            scores[j] = max_w * width_j
        end
        order = sortperm(scores)   # ascending: least important first
        cumulative = 0.0
        for idx in order
            if cumulative + scores[idx] > adaptive_prune_budget
                break
            end
            is_pruned[idx] = true
            cumulative += scores[idx]
            n_pruned += 1
        end
        n_active = n_hidden - n_pruned
        println("  adaptive_prune: H=$n_hidden, pruned $n_pruned (budget=$adaptive_prune_budget, used=$cumulative), $n_active MIP vars")
    else
        for i in 1:n_hidden
            if (a_up[i] - a_dn[i]) <= prune_tol
                is_pruned[i] = true
                n_pruned += 1
            end
        end
        n_active = n_hidden - n_pruned
        println("  encode_n1_last_layer: H=$n_hidden, pruned $n_pruned (width <= $prune_tol), $n_active MIP vars")
    end
    println("  N1 last hidden bounds width: max=$(maximum(a_up .- a_dn))")

    # ── Create interval-bounded variables only for non-pruned neurons ──
    # h_n1[i] is a JuMP variable for active neurons, unused for pruned ones.
    h_n1 = Vector{Any}(undef, n_hidden)
    for i in 1:n_hidden
        if !is_pruned[i]
            h_n1[i] = @variable(m, lower_bound = a_dn[i],
                                    upper_bound = a_up[i],
                                    base_name = "h_n1_last_$i")
        end
    end

    # ── Link non-pruned h_n1 to N2's last hidden layer via difference bounds ──
    v_n2_hidden = d[:v_n2_last_hidden]
    if v_n2_hidden !== nothing
        println("  Adding $n_active linking constraints (h_n2 - h_n1 ∈ [Δh_dn, Δh_up])")
        println("  Hidden diff width: max=$(maximum(last_hidden_diff_up .- last_hidden_diff_down))")
        for i in 1:n_hidden
            if !is_pruned[i]
                @constraint(m, v_n2_hidden[i] - h_n1[i] >= last_hidden_diff_down[i])
                @constraint(m, v_n2_hidden[i] - h_n1[i] <= last_hidden_diff_up[i])
            end
        end
    else
        println("  WARNING: v_n2_last_hidden not available, h_n1 is unlinked (loose bounds)")
    end

    println("  N1 last layer: $(size(last_linear.matrix,1)) -> $n_classes")

    # ── Helper: build margin expression m_k = (N1[c]-N1[k]) for a given k ──
    # For active neurons: uses the MIP variable h_n1[j].
    # For pruned neurons: uses the worst-case constant
    #   min(w_j * a_dn[j], w_j * a_up[j]) where w_j = W[c,j]-W[k,j].
    # This lower-bounds the real m_k, keeping delta_diff sound (>= exact).
    function build_margin_expr(k)
        # Variable part: sum over non-pruned neurons
        var_part = sum((W[c_tag, j] - W[k, j]) * h_n1[j]
                       for j in 1:n_hidden if !is_pruned[j]; init=0.0)
        # Constant part: worst-case sum over pruned neurons
        const_part = 0.0
        for j in 1:n_hidden
            if is_pruned[j]
                w_j = W[c_tag, j] - W[k, j]
                const_part += min(w_j * a_dn[j], w_j * a_up[j])
            end
        end
        return var_part + const_part + (b_vec[c_tag] - b_vec[k])
    end

    # ── Confidence of N2 on clean input (uses binary encoding for max) ──
    conf_n2_x = define_conf!(m, d, c_tag, :v_out_n2, "conf_n2_x")

    # ── delta_diff ──
    delta_diff = @variable(m, base_name="delta_diff")

    # ── Argmax encoding: conf_n1 = min_k m_k via C-1 binaries ──
    # Each m_k = (N1[c]-N1[k]) is a margin expression built by
    # build_margin_expr(k): active neurons contribute MIP variables,
    # pruned neurons contribute worst-case constants (lower-bounding
    # the real m_k). The argmax then computes the exact min of these
    # (possibly-loose) expressions.
    #
    # Soundness: each m_k^MIP <= m_k^real (worst-case on pruned terms),
    # so min_k m_k^MIP <= min_k m_k^real = conf_n1^real, hence
    # delta_diff = conf_n2 - conf_n1^MIP >= conf_n2 - conf_n1^real.

    # Create per-class margin variables and link to the margin expressions.
    v_margin = [@variable(m, base_name = "n1_margin_$k") for k in 1:n_classes]
    for k in 1:n_classes
        if k == c_tag
            continue
        end
        @constraint(m, v_margin[k] == build_margin_expr(k))
    end

    # conf_n1 = min_{k != c} v_margin[k], encoded with C-1 big-M binaries.
    max_M = 1e6
    conf_n1_x = @variable(m, base_name="conf_n1_x")
    a_conf = Dict{Int, Any}()
    for k in 1:n_classes
        if k == c_tag; continue; end
        a_conf[k] = @variable(m, binary = true, base_name="conf_n1_bin_$k")
    end
    @constraint(m, sum(a_conf[k] for k in keys(a_conf)) == 1)
    for k in 1:n_classes
        if k == c_tag; continue; end
        @constraint(m, conf_n1_x <= v_margin[k])
        @constraint(m, conf_n1_x >= v_margin[k] - max_M * (1 - a_conf[k]))
    end

    @constraint(m, delta_diff == conf_n2_x - conf_n1_x)
    @constraint(m, conf_n2_x >= 0)

    # ── Optional scalar cap on delta_diff (tightens LP relaxation) ──
    if cap_delta_diff
        delta_up_scalar = -Inf
        for k in 1:n_classes
            if k == c_tag; continue; end
            delta_up_scalar = max(delta_up_scalar, d_hi_out[c_tag] - d_lo_out[k])
        end
        println("  cap_delta_diff: delta_diff <= $delta_up_scalar")
        @constraint(m, delta_diff <= delta_up_scalar)
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

    @objective(m, Max, delta_diff)
end

# ============================================================
# Interval-based constraint: conf(N1, x', c_target) <= 0
# i.e., N1 does not classify x' as c_target.
# Sufficient condition: N1(x')[c_tag] >= N1(x')[c_target].
#
# Let δ_p[i] = N2(x')[i] - N1(x')[i] ∈ [dp_lo[i], dp_up[i]] where:
#   dp_lo[i] = d_lo[i] + n2_pert_lo[i] - n1_pert_up[i]
#   dp_up[i] = d_up[i] + n2_pert_up[i] - n1_pert_lo[i]
#
# N1(x')[c_tag] >= N1(x')[c_target]:
#   N2(x')[c_tag] - δ_p[c_tag] >= N2(x')[c_target] - δ_p[c_target]
#   N2(x')[c_target] - N2(x')[c_tag] <= δ_p[c_tag] - δ_p[c_target]
#                                      <= dp_up[c_tag] - dp_lo[c_target]
# ============================================================
function add_n1_xp_confidence_constraint!(m, d, c_tag, c_target)
    global output_diff_up_bounds, output_diff_down_bounds
    global output_n2_pert_up, output_n2_pert_down
    global output_n1_pert_up, output_n1_pert_down

    # Bounds on N2(x')[k] - N1(x')[k]
    dp_lo = output_diff_down_bounds .+ output_n2_pert_down .- output_n1_pert_up
    dp_up = output_diff_up_bounds   .+ output_n2_pert_up   .- output_n1_pert_down

    rhs = dp_up[c_tag] - dp_lo[c_target]
    println("  constrain_n1_xp: N2(x')[$c_target] - N2(x')[$c_tag] <= $rhs")
    println("    dp_up[$c_tag] = $(dp_up[c_tag]), dp_lo[$c_target] = $(dp_lo[c_target])")

    if rhs < 0
        println("    → rhs < 0: would conflict with N2-fooled constraint, skipping")
    elseif rhs < 1e6
        @constraint(m, d[:v_out_n2_p][c_target] - d[:v_out_n2_p][c_tag] <= rhs)
        println("    → Active constraint added (rhs=$rhs)")
    else
        println("    → Constraint trivially satisfied (rhs=$rhs too large), skipping")
    end
end

function mip_set_attr_transfer(m, timout, suboptimal_solution=0)
    set_optimizer_attribute(m, "MIPFocus", 3)
    set_optimizer_attribute(m, "Cutoff", suboptimal_solution)
    set_optimizer_attribute(m, "Threads", 32)
    set_optimizer_attribute(m, "TimeLimit", timout)
    set_optimizer_attribute(m, "MIPGap", 0.01)
    global gurobi_seed
    set_optimizer_attribute(m, "Seed", gurobi_seed)
    # cnn2 only: the PGD partial warm-start triggers a completion sub-MIP whose
    # default node budget explodes on this large model (~1h/target, no incumbent).
    # Bound it so Gurobi stops repairing the start and gets to the real solve.
    # Every other architecture is left exactly as before (no attribute set).
    global model_name
    if model_name == "cnn2"
        set_optimizer_attribute(m, "StartNodeLimit", 100)
    end
    if model_name == "cnn5"
        set_optimizer_attribute(m, "StartNodeLimit", 70)
    end
    if model_name == "cnn4"
        set_optimizer_attribute(m, "StartNodeLimit", 70)
    end
end
