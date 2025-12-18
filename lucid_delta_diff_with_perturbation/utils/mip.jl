
function mip_reset()
    neurons_names.neuron = 0
    neurons_names.layer = 0
    first_mip_solution.solution = -1.0
    first_mip_solution.time = 0.0
end

function mip_set_delta_diff_propery(m,d,c_tag)
    max_num = 1e6 # Big number
    delta2 = @variable(m, base_name="delta2")
    @variable(m, max_kk_2)
    @constraint(m, delta2 == d[:v_out_p][c_tag] - max_kk_2)
    a_delta2 = Dict()
    for i in 1:10
        if i == c_tag
            continue  # Skip this iteration
        end
        a_delta2[i] = @variable(m, binary = true, base_name = "a$(i)_delta2")
    end
    # Add constraint: exactly one variable equals 1
    @constraint(m, sum(a_delta2[i] for i in keys(a_delta2) if i != c_tag) == 1)
    for i in 1:10
        if i == c_tag
            continue  # Skip this iteration
        end
        @constraint(m, max_kk_2 >= d[:v_out_p][i])
        @constraint(m, max_kk_2 <= d[:v_out_p][i]+max_num*(1-a_delta2[i]))
    end
    

    delta1 = @variable(m, base_name="delta1")
    @variable(m, max_kk_1)
    @constraint(m, delta1 == d[:v_out][c_tag] - max_kk_1)
    a_delta1 = Dict()
    for i in 1:10
        if i == c_tag
            continue  # Skip this iteration
        end
        a_delta1[i] = @variable(m, binary = true, base_name = "a$(i)_delta1")
    end
    # Add constraint: exactly one variable equals 1
    @constraint(m, sum(a_delta1[i] for i in keys(a_delta1) if i != c_tag) == 1)
    for i in 1:10
        if i == c_tag
            continue  # Skip this iteration
        end
        @constraint(m, max_kk_1 >= d[:v_out][i])
        @constraint(m, max_kk_1 <= d[:v_out][i]+max_num*(1-a_delta1[i]))
    end

    diff = @variable(m, base_name="diff_obj")
    # @constraint(m, diff == delta1 - delta2)
    # if occursin("Max",problem_type_str)
    #     @objective(m, Max, diff)
    # else # Min
    #     @objective(m, Min, diff)
    # end
    @constraint(m, diff == delta1 - delta2)
    # @constraint(m, diff >= delta2 - delta1)
    @variable(m, z, Bin)  # z ∈ {0,1}
    M = 1e6  # large constant (big-M method)

    # @constraint(m, diff ≤ delta1 - delta2 + M * z)
    # @constraint(m, diff ≤ delta2 - delta1 + M * (1 - z))

    @constraint(m, delta1>=10)
    @constraint(m, delta2>=10)

    @objective(m, Max, diff)
end

# function mip_set_delta_property_diff_neta(m, d)
#     (maximum_target_var1, nontarget_vars1) = get_vars_for_max_index(d[:v_out], d[:SourceIndex])
#     maximum_nontarget_var1 = maximum_ge(nontarget_vars1)
#     delta1 = @variable(m)
#     @constraint(m, delta1 == maximum_target_var1 - maximum_nontarget_var1)
#     (maximum_target_var2, nontarget_vars2) = get_vars_for_max_index(d[:v_out_p], d[:SourceIndex])
#     maximum_nontarget_var2 = maximum_ge(nontarget_vars2)
#     a2_delta2 = @variable(m)
#     @constraint(m, delta2 == maximum_target_var2 - maximum_nontarget_var2)
    
# end

function mip_set_delta_property(m, d)
    set_max_indexes(m, d[:v_out_p], d[:TargetIndex])
    (maximum_target_var, nontarget_vars) = get_vars_for_max_index(d[:v_out], d[:SourceIndex])
    maximum_nontarget_var = maximum_ge(nontarget_vars)
    delta = @variable(m)
    @constraint(m, delta == maximum_target_var - maximum_nontarget_var)
    @objective(m, Max, delta)
end

function mip_set_attr(m, d, timout)
    set_optimizer_attribute(m, "MIPFocus", 3)
    set_optimizer_attribute(m, "Cutoff", 0)
    set_optimizer_attribute(m, "Threads", 8)
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
