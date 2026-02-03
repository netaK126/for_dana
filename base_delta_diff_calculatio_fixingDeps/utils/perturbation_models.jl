function get_model(w_, h_, k_,
    perturbation,
    perturbation_size,
    nn1::NeuralNet,
    nn2::NeuralNet,
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
    if perturbation == "contrast"
        set_optimizer_attribute(m, "NonConvex", 2)
    end
    m.ext[:MIPVerify] = MIPVerifyExt(tightening_algorithm)
    d_common = Dict(
        :Model => m,
        :TighteningApproach => string(tightening_algorithm),
    )
    println("Encoding the two copies...")
    if perturbation == "delta_diff"
        return merge(d_common, get_delta_diff_keys(perturbation_size, nn1, nn2, input, m))
    end
end

function get_delta_diff_keys(perturbation_size, nn_hyper::NeuralNet, nn::NeuralNet,input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e = map(_ -> @variable(m, lower_bound = -p_size, upper_bound = p_size), input_range,)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    @constraint(m, v_x0 .== v_in + v_e)

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