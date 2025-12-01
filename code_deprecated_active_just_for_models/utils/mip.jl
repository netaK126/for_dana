
function mip_reset()
    neurons_names.neuron = 0
    neurons_names.layer = 0
    first_mip_solution.solution = -1.0
    first_mip_solution.time = 0.0
end

function mip_set_delta_property(m, perturbation, d, problem_type_str)
#     (maximum_target_var2, nontarget_vars2) = get_vars_for_max_index(d[:v_output_nn2], d[:SourceIndex])
#     maximum_nontarget_var2 = maximum_ge(nontarget_vars2)
#     delta2 = @variable(m, base_name="delta2")
#     @constraint(m, delta2 == maximum_target_var2 - maximum_nontarget_var2)
#
#     (maximum_target_var, nontarget_vars) = get_vars_for_max_index(d[:v_output_nn1], d[:SourceIndex])
#     maximum_nontarget_var = maximum_ge(nontarget_vars)
#     delta1 = @variable(m, base_name="delta1")
#     @constraint(m, delta1 == maximum_target_var - maximum_nontarget_var)


    delta2 = @variable(m, base_name="delta2")
    @variable(m, max_kk_2)
    for k_2 in 2:1:10
        @constraint(m, max_kk_2 >= d[:v_output_nn2][k_2])
    end
    @constraint(m, delta2 == d[:v_output_nn2][1] - max_kk_2)
    a2_delta2 = @variable(m, binary = true)
    a3_delta2 = @variable(m, binary = true)
    a4_delta2 = @variable(m, binary = true)
    a5_delta2 = @variable(m, binary = true)
    a6_delta2 = @variable(m, binary = true)
    a7_delta2 = @variable(m, binary = true)
    a8_delta2 = @variable(m, binary = true)
    a9_delta2 = @variable(m, binary = true)
    a10_delta2 = @variable(m, binary = true)
    @constraint(m, a2_delta2+a3_delta2+a4_delta2+a5_delta2+a6_delta2+a7_delta2+a8_delta2+a9_delta2+a10_delta2==1)
    @constraint(m, max_kk_2 == a2_delta2*d[:v_output_nn2][2]+a3_delta2*d[:v_output_nn2][3]+a4_delta2*d[:v_output_nn2][4]+a5_delta2*d[:v_output_nn2][5]+a6_delta2*d[:v_output_nn2][6]+a7_delta2*d[:v_output_nn2][7]+a8_delta2*d[:v_output_nn2][8]+a9_delta2*d[:v_output_nn2][9]+a10_delta2*d[:v_output_nn2][10])


    delta1 = @variable(m, base_name="delta1")
    @variable(m, max_kk)
    for k in 2:1:10
        @constraint(m, max_kk >= d[:v_output_nn1][k])
    end
    @constraint(m, delta1 == d[:v_output_nn1][1] - max_kk)
    a2 = @variable(m, binary = true)
    a3 = @variable(m, binary = true)
    a4 = @variable(m, binary = true)
    a5 = @variable(m, binary = true)
    a6 = @variable(m, binary = true)
    a7 = @variable(m, binary = true)
    a8 = @variable(m, binary = true)
    a9 = @variable(m, binary = true)
    a10 = @variable(m, binary = true)
    @constraint(m, a2+a3+a4+a5+a6+a7+a8+a9+a10==1)
    @constraint(m, max_kk == a2*d[:v_output_nn1][2]+a3*d[:v_output_nn1][3]+a4*d[:v_output_nn1][4]+a5*d[:v_output_nn1][5]+a6*d[:v_output_nn1][6]+a7*d[:v_output_nn1][7]+a8*d[:v_output_nn1][8]+a9*d[:v_output_nn1][9]+a10*d[:v_output_nn1][10])

    diff = @variable(m)
    @constraint(m, diff == delta1 - delta2)
    if occursin("Max",problem_type_str)
        @objective(m, Max, diff)
    else # Min
        @objective(m, Min, diff)
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
