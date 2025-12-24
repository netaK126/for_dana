function get_model(
    perturbation,
    perturbation_size,
    nn::NeuralNet,
    nn_hyper::NeuralNet,
    input::Array{<:Real},
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
)::Dict{Symbol,Any}
    notice(
        MIPVerify.LOGGER,
        "Determining upper and lower bounds for the input to each non-linear unit.",
    )
    m = Model(optimizer_with_attributes(optimizer, tightening_options...))
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)
    d_common = Dict(
        :Model => m,
        :TighteningApproach => string(tightening_algorithm),
    )
    println("Encoding the two copies...")
    if perturbation == "linf"
        return merge(d_common, get_perturbation_specific_keys_linf(perturbation_size, nn, nn_hyper, input, m))
    end
end


function get_perturbation_specific_keys_linf(perturbation_size, nn::NeuralNet, nn_hyper::NeuralNet,input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    global network_version
    global layer_counter
    global nueron_counter
    global all_bounds_of_original
    global all_bounds_of_perturbation
    global I_z_prev_up
    global I_z_prev_up_perturbation
    global I_z_prev_down
    global I_z_prev_down_perturbation

    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e = map(_ -> @variable(m, lower_bound = -p_size, upper_bound = p_size), input_range,)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    @constraint(m, v_x0 .== v_in + v_e)

    if size(input)[4]>1
        # org
	    append!(all_bounds_of_original,[[ones(Float64, size(input)[4], 1),zeros(Float64, size(input)[4], 1)]])
	    I_z_prev_up =  zeros(Float64, size(input)[4], 1)
        I_z_prev_down =  zeros(Float64, size(input)[4], 1)
        # perturbation
        append!(all_bounds_of_perturbation,[[ones(Float64, size(input)[4], 1),zeros(Float64, size(input)[4], 1)]])
	    I_z_prev_up_perturbation =  zeros(Float64, size(input)[4], 1)
        I_z_prev_down_perturbation =  zeros(Float64, size(input)[4], 1)
    else
        # org
	    append!(all_bounds_of_original,[[ones(Float64, size(input)),zeros(Float64, size(input))]])
	    I_z_prev_up =  zeros(Float64, size(input))
        I_z_prev_down =  zeros(Float64, size(input))
        # perturbation
        append!(all_bounds_of_perturbation,[[ones(Float64, size(input)),zeros(Float64, size(input))]])
	    I_z_prev_up_perturbation =  zeros(Float64, size(input))
        I_z_prev_down_perturbation =  zeros(Float64, size(input))
    end


    println("regular nns")
    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn
    layer_counter = 0
    nueron_counter = 0
    network_version = "hyper"
    v_output = v_in |> nn_hyper

    println("perturbed nns")
    layer_counter = 0
    nueron_counter = 0
    network_version = "orgP"
    v_in_output_p = v_x0 |> nn
    layer_counter = 0
    nueron_counter = 0
    network_version = "hyperP"
    v_output_p = v_x0 |> nn_hyper


    return Dict(:v_in_hyper => v_in,
                :Perturbation => "None",
                :v_out_hyper => v_output,
                :v_in_nn => v_in,
                :v_out_nn => v_in_output,
                :v_in_hyper_perturbation => v_x0,
                :v_out_hyper_perturbation => v_output_p,
                :v_in_nn_perturbation => v_x0,
                :v_out_nn_perturbation => v_in_output_p)
end


