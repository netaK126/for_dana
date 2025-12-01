function get_model(w_, h_, k_,
    perturbation,
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
        return merge(d_common, get_delta_diff_keys(nn1, nn2, input, m))
    end
end

function get_delta_diff_keys(nn1, nn2, input, m)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map( i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_output_nn1 = v_in |> nn1
    v_output_nn2 = v_in |> nn2
    return Dict(:Perturbation => "None", :v_out_p => v_output_nn2, :v_in => v_in, :v_out => v_output_nn1)
end