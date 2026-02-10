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
    if perturbation=="linf"
        return merge(d_common, get_perturbation_specific_keys_privacy(perturbation_size, nn, nn_hyper, input, m))

    end
end


function get_perturbation_specific_keys_privacy(perturbation_size, nn::NeuralNet, nn_hyper::NeuralNet,input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    global network_version
    global layer_counter
    global nueron_counter
    global all_bounds_of_original
    global I_z_prev_up
    global I_z_prev_down

    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]

    v_e = map(_ -> @variable(m, lower_bound = -p_size, upper_bound = p_size), input_range,)
    v_in = map( i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)



    if size(input)[4]>1
	    append!(all_bounds_of_original,[[ones(Float64, size(input)[4], 1),zeros(Float64, size(input)[4], 1)]])
	    I_z_prev_up =  zeros(Float64, size(input)[4], 1)
        I_z_prev_down =  zeros(Float64, size(input)[4], 1)
    else
	    append!(all_bounds_of_original,[[ones(Float64, size(input)),zeros(Float64, size(input))]])
	    I_z_prev_up =  zeros(Float64, size(input))
        I_z_prev_down =  zeros(Float64, size(input))
    end



    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn

    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn_hyper
    return Dict(:v_in_p => v_in, :Perturbation => "linf", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end




