
function mip_reset()
    neurons_names.neuron = 0
    neurons_names.layer = 0
    first_mip_solution.solution = -1.0
    first_mip_solution.time = 0.0
end

function mip_set_delta_property(m, perturbation, d,c_tag)
    if perturbation != "max" && perturbation != "min"
        # in that case, we should encode delta_p
        max_num = 1e6 # Big number
        delta_p = @variable(m, base_name="delta_p")
        @variable(m, max_kk_2) # this is would be max_{j \ne c_tag}N(fp(x,eps))[j]
        @constraint(m, delta_p == d[:v_out_p][c_tag] - max_kk_2) # delta_p encoding
        a_delta_p = Dict() # contains binary variables. if a_delta_p[j]==1 then j=arg {max_{j \ne c_tag}N(fp(x,eps))[j]}
        for i in 1:10
            if i == c_tag
                continue  # Skip this iteration
            end
            a_delta_p[i] = @variable(m, binary = true, base_name = "a$(i)_delta_p")
        end
        @constraint(m, sum(a_delta_p[i] for i in keys(a_delta_p) if i != d[:SourceIndex]) == 1) # exactly one variable equals 1
        for i in 1:10
            if i == c_tag
                continue  # Skip this iteration
            end
            # encoding max_kk_2 to be the maximum value: max_{j \ne c_tag}N(fp(x,eps))[j]
            @constraint(m, max_kk_2 >= d[:v_out_p][i])
            @constraint(m, max_kk_2 <= d[:v_out_p][i]+max_num*(1-a_delta_p[i]))
        end
        @constraint(m, delta_p <= d[:max_abs_value])
    end

    # encoding delta
    (maximum_target_var, nontarget_vars) = get_vars_for_max_index(d[:v_out], d[:SourceIndex])
    if  perturbation == "min"
        maximum_nontarget_var = maximum_ge_2(nontarget_vars)
    else
        maximum_nontarget_var = maximum_ge(nontarget_vars)
    end
    delta = @variable(m, base_name = "delta_vaghar")
    @constraint(m, delta == maximum_target_var - maximum_nontarget_var)

    if  perturbation == "min"
        @objective(m, Min, delta)
    else
        @objective(m, Max, delta)
    end
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
