
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

