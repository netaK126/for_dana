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
    if perturbation == "linf"
        return merge(d_common, get_perturbation_specific_keys_linf_transfer(perturbation_size, nn1, nn2, input, m))
    else
        error("Transfer mode currently only supports linf perturbation. Got: $perturbation")
    end
end

function get_perturbation_specific_keys_linf_transfer(perturbation_size, nn1::NeuralNet, nn2::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]

    # Shared input variables: x (clean) and x_p (perturbed)
    v_e = map(_ -> @variable(m, lower_bound = -p_size, upper_bound = p_size), input_range)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    @constraint(m, v_x0 .== v_in + v_e)


    # Encode N1 on clean input x → layers 1..K in layers_info_dict
    println("Encoding N1(x)...")
    network_version = "n1_org"
    v_out_n1 = v_in |> nn1

    # Encode N2 on clean input x → layers K+1..2K
    println("Encoding N2(x)...")
    network_version = "n2_org"
    v_out_n2 = v_in |> nn2

    # Encode N2 on perturbed input x_p → layers 3K+1..4K
    println("Encoding N2(x_p)...")
    layer_counter = 0
    nueron_counter = 0
    network_version = "n2_pert"
    v_out_n2_p = v_x0 |> nn2

    return Dict(
        :v_in => v_in,
        :v_in_p => v_x0,
        :Perturbation => v_e,
        :v_out_n1 => v_out_n1,
        :v_out_n2 => v_out_n2,
        :v_out_n2_p => v_out_n2_p,
    )
end
