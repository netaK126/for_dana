function set_intervals(w_, h_, k_,
    perturbation,
    perturbation_size,
    nn::NeuralNet,
    input::Array{<:Real},
    optimizer,
    d)
    global network_version
    global layer_counter
    global nueron_counter

    v_p = d[:v_in_p]

    set_input_interval(w_, h_, k_,d,perturbation,perturbation_size,input)
    layer_counter = 0
    # TODO - HOW TO GET W AND B??
    for l in nn.layers
        
    end

end

function set_input_interval(w_, h_, k_,
    d,
    perturbation,
    perturbation_size,
    input::Array{<:Real})

    if perturbation=="linf"
        v_p = d[:v_in_p]
        v_in = d[:v_in ]
        m = d[:Model]
        @constraint(m, v_p .<= v_in + perturbation_size)
        @constraint(m, v_p .>= v_in - perturbation_size)
    end

end