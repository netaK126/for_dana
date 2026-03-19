
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
    set_optimizer_attribute(m, "Threads", 32)
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
    d[:best_bound] = JuMP.objective_bound(m)
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

function mip_set_attr_transfer(m, timout, suboptimal_solution=0)
    set_optimizer_attribute(m, "MIPFocus", 3)
    if suboptimal_solution != 0
        set_optimizer_attribute(m, "Cutoff", suboptimal_solution)
    end
    set_optimizer_attribute(m, "Threads", 32)
    set_optimizer_attribute(m, "TimeLimit", timout)
    set_optimizer_attribute(m, "MIPGap", 0.01)
end
