function get_model(
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
    return merge(d_common, get_perturbation_specific_keys_privacy(nn, nn_hyper, input, m))
end


function get_perturbation_specific_keys_privacy(  nn::NeuralNet, nn_hyper::NeuralNet,input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    global network_version
    global layer_counter
    global nueron_counter
    global upper_bound_prev
    global lower_bound_prev
    global all_bounds_of_original
    global I_z_prev_up
    global I_z_prev_down

    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)



    if size(input)[4]>1
	    upper_bound_prev =  ones(Float64, size(input)[4], 1)
	    lower_bound_prev =  zeros(Float64, size(input)[4], 1)
	    append!(all_bounds_of_original,[[ones(Float64, size(input)[4], 1),zeros(Float64, size(input)[4], 1)]])
	    I_z_prev_up =  zeros(Float64, size(input)[4], 1)
        I_z_prev_down =  zeros(Float64, size(input)[4], 1)
    else
	    upper_bound_prev =  ones(Float64, size(input))
	    lower_bound_prev =  zeros(Float64, size(input))
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
    network_version = "hyper"
    v_output = v_in |> nn_hyper
    return Dict(:v_in_p => v_in, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end




