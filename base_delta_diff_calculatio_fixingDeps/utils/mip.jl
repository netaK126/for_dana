
function mip_reset()
    neurons_names.neuron = 0
    neurons_names.layer = 0
    first_mip_solution.solution = -1.0
    first_mip_solution.time = 0.0
end

function define_conf!(m, d, c_tag, key, name)
    max_num = 1e6 # Big number
    conf = @variable(m, base_name=name)
    max_kk = @variable(m)
    @constraint(m, conf == d[key][c_tag] - max_kk)
    a_conf = Dict()
    for i in 1:10
        if i == c_tag
            continue  # Skip this iteration
        end
        a_conf[i] = @variable(m, binary = true)
    end
    @constraint(m, sum(a_conf[i] for i in keys(a_conf)) == 1)
    for i in 1:10
        if i == c_tag
            continue  # Skip this iteration
        end
        @constraint(m, max_kk >= d[key][i])
        @constraint(m, max_kk <= d[key][i]+max_num*(1-a_conf[i]))
    end
    return conf
end

function mip_set_delta_diff_property(m, d,delta1_vaghar, c_tag)
    conf2 = define_conf!(m,d,c_tag, :v_out_2, "conf2")
    conf2_p = define_conf!(m,d,c_tag, :v_out_p_2, "conf2_p")
    conf1 = define_conf!(m,d,c_tag, :v_out_1, "conf1")
    conf1_p = define_conf!(m,d,c_tag, :v_out_p_1, "conf1_p")
    margin = 0.001
    # the objective and problem definition
    diff = @variable(m, base_name="diff_obj")
    @constraint(m, conf1>=delta1_vaghar + margin)
    @constraint(m, conf2 - conf1==diff)
    @constraint(m, diff>=0)
    @constraint(m,conf2_p-conf1_p<=-margin)

    @objective(m, Max, diff)
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

    # To prioritize finding a feasible solution quickly
    # set_optimizer_attribute(m, "MIPFocus", 1) 
    # set_optimizer_attribute(m, "Heuristics", 0.5) 
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
