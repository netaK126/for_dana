function get_model(w_, h_, k_,
    perturbation,
    perturbation_size,
    nn::NeuralNet,
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
    if perturbation == "brightness"
        return merge(d_common, get_perturbation_specific_keys_brightness(perturbation_size,nn, input, m))
    elseif perturbation == "linf"
        return merge(d_common, get_perturbation_specific_keys_linf(perturbation_size,nn, input, m))
    elseif perturbation == "max"
        return merge(d_common, get_perturbation_specific_keys_max(perturbation_size,nn, input, m))
    elseif perturbation == "contrast"
        return merge(d_common, get_perturbation_specific_keys_contrast(perturbation_size,nn, input, m))
    elseif perturbation == "occ"
        return merge(d_common, get_perturbation_specific_keys_occ(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "patch"
        return merge(d_common, get_perturbation_specific_keys_patch(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "patchM"
        return merge(d_common, get_perturbation_specific_keys_patchM(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "occM"
        return merge(d_common, get_perturbation_specific_keys_occM(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "translation"
        return merge(d_common, get_perturbation_specific_keys_translation(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "rotation"
        return merge(d_common, get_perturbation_specific_keys_rotate(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "filterv"
        return merge(d_common, get_perturbation_specific_keys_filter_v(perturbation_size,nn, input, m))
     elseif perturbation == "Privacy"
        return merge(d_common, get_perturbation_specific_keys_privacy(w_, h_, k_, perturbation_size, nn, nn_second, input, m))
    else
        return merge(d_common, get_perturbation_specific_keys(perturbation_size,nn, input, m))
    end
end


function get_perturbation_specific_keys_linf(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    global layer_counter
    global nueron_counter
    global network_version
    global I_pert_prev_up
    global I_pert_prev_down

    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e = map(_ -> @variable(m, lower_bound = -p_size, upper_bound = p_size), input_range,)
    v_in = map( i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    @constraint(m, v_x0 .== v_in + v_e)

    # Initialize perturbation interval globals: Δ_{0,k} = e_k ∈ [-ε, ε]
    if size(input)[4] > 1
        I_pert_prev_up = p_size .* ones(Float64, size(input)[4], 1)
        I_pert_prev_down = -p_size .* ones(Float64, size(input)[4], 1)
    else
        I_pert_prev_up = p_size .* ones(Float64, size(input))
        I_pert_prev_down = -p_size .* ones(Float64, size(input))
    end

    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn

    # Pre-compute perturbation interval bounds for the conditional-triangle
    # relaxation (used by core_ops.jl's relu() when use_relaxations=true).
    # With nn1==nn2 (same network), diff=0 and composed=pert, so
    # relu_comp_up/down_bounds will hold the perturbation interval bounds.
    if use_relaxations
        compute_diff_and_comp_bounds(nn, nn, I_pert_prev_up, I_pert_prev_down; optimizing_intervals=optimizing_intervals)
    end

    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => v_e, :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_privacy(w_, h_, k_, perturbation_size, nn::NeuralNet, nn_hyper::NeuralNet,input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_in_output = v_in |> nn
    v_output = v_in |> nn_hyper
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_brightness(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    global layer_counter
    global nueron_counter
    global network_version
    global I_pert_prev_up, I_pert_prev_down
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e = @variable(m, lower_bound = 0, upper_bound = p_size)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1),input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1+p_size), input_range,)
    @constraint(m, v_x0 .== v_in .+ v_e)

    # Perturbation interval: Δ = x' - x = e, e ∈ [0, ε] → Δ ∈ [0, ε]
    if size(input)[4] > 1
        I_pert_prev_up = p_size .* ones(Float64, size(input)[4], 1)
        I_pert_prev_down = zeros(Float64, size(input)[4], 1)
    else
        I_pert_prev_up = p_size .* ones(Float64, size(input))
        I_pert_prev_down = zeros(Float64, size(input))
    end

    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn
    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => v_e, :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_max(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    global layer_counter, nueron_counter, network_version
    println("HERE")
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_output = v_in |> nn
    return Dict(:v_in_p => v_in, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_output)
end

#contrast
function get_perturbation_specific_keys_contrast(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    global layer_counter, nueron_counter, network_version
    global I_pert_prev_up, I_pert_prev_down
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e = @variable(m, lower_bound = 1.0, upper_bound = 1+p_size)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1+p_size), input_range,)
    @constraint(m, v_x0 .== v_e*v_in)

    # Perturbation interval: Δ = x' - x = (e-1)*x, e-1 ∈ [0, ε], x ∈ [0,1] → Δ ∈ [0, ε]
    if size(input)[4] > 1
        I_pert_prev_up = p_size .* ones(Float64, size(input)[4], 1)
        I_pert_prev_down = zeros(Float64, size(input)[4], 1)
    else
        I_pert_prev_up = p_size .* ones(Float64, size(input))
        I_pert_prev_down = zeros(Float64, size(input))
    end

    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn
    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => v_e, :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_occ(w_, h_, k_, perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    l = []
    ind1 = Int(perturbation_size[1])
    ind2 = Int(perturbation_size[2])
    w = w_
    h = h_
    k = k_
    res_ = w*h
    ind = ind1 + (ind2-1)*w
    for i_ in 0:Int(perturbation_size[3])-1
        for j_ in 0:Int(perturbation_size[3])-1
            append!(l, ind+j_+w*i_)
            if k == 3
                append!(l, res_+ind+j_+w*i_)
                append!(l, 2*res_+ind+j_+w*i_)
            end
        end
    end
    res = []
    for tt in 1:Int(w_*h_*k_)
        if tt in l
            continue
        end
        append!(res, tt)
    end
    @constraint(m, c1[i=l],v_x0[i] == 0.0)
    @constraint(m, c2[i=res],v_x0[i] == v_in[i])

    # Perturbation interval bounds: occluded Δ ∈ [-1, 0], non-occluded Δ = 0
    global I_pert_prev_up, I_pert_prev_down
    I_pert_prev_up = zeros(Float64, size(input))
    I_pert_prev_down = zeros(Float64, size(input))
    for i in l
        I_pert_prev_down[i] = -1.0
    end

    global layer_counter, nueron_counter, network_version
    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn
    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_patch(w_, h_, k_, perturbation_size, nn::NeuralNet,input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    l = []
    eps  = perturbation_size[1]
    ind1 = Int(perturbation_size[2])
    ind2 = Int(perturbation_size[3])
    w = w_
    h = h_
    k = k_
    res_ = w*h

    ind = ind1 + (ind2-1)*w
    for i_ in 0:Int(perturbation_size[4])-1
        for j_ in 0:Int(perturbation_size[4])-1
            append!(l, ind+j_+w*i_)
            if k == 3
                append!(l, res_+ind+j_+w*i_)
                append!(l, 2*res_+ind+j_+w*i_)
            end
        end
    end
    res = []
    for tt in 1:Int(w_*h_*k_)
        if tt in l
            continue
        end
        append!(res, tt)
    end
    @constraint(m, c0[i=l],v_x0[i] <= v_in[i]+eps)
    @constraint(m, c1[i=l],v_x0[i] >= v_in[i]-eps)
    @constraint(m, c2[i=res],v_x0[i] == v_in[i])

    # Perturbation interval bounds: patch Δ ∈ [-eps, eps], non-patch Δ = 0
    global I_pert_prev_up, I_pert_prev_down
    I_pert_prev_up = zeros(Float64, size(input))
    I_pert_prev_down = zeros(Float64, size(input))
    for i in l
        I_pert_prev_up[i] = eps
        I_pert_prev_down[i] = -eps
    end

    global layer_counter, nueron_counter, network_version
    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn
    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_patchM(w_, h_, k_, perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
   #TPD
end

function get_perturbation_specific_keys_occM(w_, h_, k_, perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    #TPD
end

function get_perturbation_specific_keys_translation(w_, h_, k_,perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    t_down = Int(perturbation_size[1])
    t_right = Int(perturbation_size[2])
    k = k_
    w = w_
    h = h_
    res = w*h
    m_ind = t_down
    n_ind = t_right
    for i2 = 1:w-t_right
        for i1 = 1:h-t_down
            i = i1 + h *(i2-1)
            @constraint(m,v_x0[i+m_ind+w*n_ind] == v_in[i])
            if k == 3
                i = res+i1 + h *(i2-1)
                @constraint(m,v_x0[i+m_ind+w*n_ind] == v_in[i])
                i = 2*res+i1 + h *(i2-1)
                @constraint(m,v_x0[i+m_ind+w*n_ind] == v_in[i])
            end
        end
    end

    for j = 1:t_down
        @constraint(m,[i=j:w:res],v_x0[i] == 0)
        if k == 3
            @constraint(m,[i=j+res:w:2*res],v_x0[i] == 0)
            @constraint(m,[i=j+2*res:w:3*res],v_x0[i] == 0)
        end
    end
    for j = 1:t_right
        @constraint(m,[i=1+w*(j-1):1:w*j],v_x0[i] == 0)
        if k == 3
            @constraint(m,[i=res+1+w*(j-1):1:res+w*j],v_x0[i] == 0)
            @constraint(m,[i=res*2+1+w*(j-1):1:res*2+w*j],v_x0[i] == 0)
        end
    end

    # Perturbation interval bounds:
    #   border (zeroed) pixels: Δ ∈ [-1, 0]
    #   interior (shifted) pixels: Δ = x[src] - x[dst] ∈ [-1, 1]
    global I_pert_prev_up, I_pert_prev_down
    I_pert_prev_up = ones(Float64, size(input))
    I_pert_prev_down = -ones(Float64, size(input))

    global layer_counter, nueron_counter, network_version
    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn
    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_filter_v(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)

    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    for j=1:27
        @constraint(m,[i=28*j+1:28*(j+1)-1],v_x0[i]==0.01*v_in[i-1]+0.99*v_in[i]+0.01*v_in[i+1])
    end
    @constraint(m,[i=28:28:784],v_x0[i]== 0.1*v_in[i-1]+0.8*v_in[i])
    @constraint(m,[i=1:28:756],v_x0[i]== 0.8*v_in[i]+0.1*v_in[i+1])
    global layer_counter, nueron_counter, network_version
    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn
    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end


# ============================================================
# Transfer proof: four-network MIP encoding
# N1(x), N2(x), N1(x_p), N2(x_p)
# ============================================================

function get_model_transfer(w_, h_, k_,
    perturbation,
    perturbation_size,
    nn1::NeuralNet,
    nn2::NeuralNet,
    input::Array{<:Real},
    optimizer,
    tightening_options::Dict,
    tightening_algorithm::TighteningAlgorithm,
    n1_p_mode::Bool,
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
    # Contrast uses a bilinear constraint (x' = e*x), which requires
    # Gurobi's NonConvex=2 mode to handle products of two variables.
    if perturbation == "contrast"
        set_optimizer_attribute(m, "NonConvex", 2)
    end
    println("Encoding four network copies for transfer proof...")
    if perturbation == "linf"
        return merge(d_common, get_perturbation_specific_keys_linf_transfer(perturbation_size, nn1, nn2, input, m, n1_p_mode))
    elseif perturbation == "brightness"
        return merge(d_common, get_perturbation_specific_keys_brightness_transfer(perturbation_size, nn1, nn2, input, m, n1_p_mode))
    elseif perturbation == "contrast"
        return merge(d_common, get_perturbation_specific_keys_contrast_transfer(perturbation_size, nn1, nn2, input, m, n1_p_mode))
    elseif perturbation == "translation"
        return merge(d_common, get_perturbation_specific_keys_translation_transfer(w_, h_, k_, perturbation_size, nn1, nn2, input, m, n1_p_mode))
    elseif perturbation == "patch"
        return merge(d_common, get_perturbation_specific_keys_patch_transfer(w_, h_, k_, perturbation_size, nn1, nn2, input, m, n1_p_mode))
    elseif perturbation == "occ"
        return merge(d_common, get_perturbation_specific_keys_occ_transfer(w_, h_, k_, perturbation_size, nn1, nn2, input, m, n1_p_mode))
    elseif perturbation == "rotation"
        return merge(d_common, get_perturbation_specific_keys_rotate_transfer(w_, h_, k_, perturbation_size, nn1, nn2, input, m, n1_p_mode))
    else
        error("Transfer mode does not support perturbation type: $perturbation")
    end
end

function get_perturbation_specific_keys_linf_transfer(perturbation_size, nn1::NeuralNet, nn2::NeuralNet, input::Array{<:Real}, m::Model, n1_p_mode::Bool)::Dict{Symbol,Any}
    global layer_counter
    global nueron_counter
    global network_version
    global I_pert_prev_up
    global I_pert_prev_down
    global all_bounds_of_original
    global all_bounds_of_perturbation
    global I_z_prev_up
    global I_z_prev_up_perturbation
    global I_z_prev_down
    global I_z_prev_down_perturbation

    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]

    # Shared input variables: x (clean) and x_p (perturbed)
    v_e = map(_ -> @variable(m, lower_bound = -p_size, upper_bound = p_size), input_range)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    @constraint(m, v_x0 .== v_in + v_e)

    # Initialize interval globals (same pattern as lucid: append! on untyped [])
    all_bounds_of_original = []
    all_bounds_of_perturbation = []
    if size(input)[4] > 1
        # conv input
        append!(all_bounds_of_original, [[ones(Float64, size(input)[4], 1), zeros(Float64, size(input)[4], 1)]])
        I_z_prev_up = zeros(Float64, size(input)[4], 1)
        I_z_prev_down = zeros(Float64, size(input)[4], 1)
        append!(all_bounds_of_perturbation, [[ones(Float64, size(input)[4], 1), zeros(Float64, size(input)[4], 1)]])
        I_z_prev_up_perturbation = zeros(Float64, size(input)[4], 1)
        I_z_prev_down_perturbation = zeros(Float64, size(input)[4], 1)
    else
        # FC input
        append!(all_bounds_of_original, [[ones(Float64, size(input)), zeros(Float64, size(input))]])
        I_z_prev_up = zeros(Float64, size(input))
        I_z_prev_down = zeros(Float64, size(input))
        append!(all_bounds_of_perturbation, [[ones(Float64, size(input)), zeros(Float64, size(input))]])
        I_z_prev_up_perturbation = zeros(Float64, size(input))
        I_z_prev_down_perturbation = zeros(Float64, size(input))
    end

    # I_pert_prev_up/down — initial perturbation interval at the input layer:
    #   Δ[i] = x'[i] - x[i] = e[i],  e[i] ∈ [-ε, ε]  (per-pixel, independent)
    #   → I_pert_prev_up[i]   =  ε  for all i
    #   → I_pert_prev_down[i] = -ε  for all i
    # Used by perturbed_interval_constraints() to propagate Δ through layers.
    if size(input)[4] > 1
        I_pert_prev_up = p_size .* ones(Float64, size(input)[4], 1)
        I_pert_prev_down = -p_size .* ones(Float64, size(input)[4], 1)
    else
        I_pert_prev_up = p_size .* ones(Float64, size(input))
        I_pert_prev_down = -p_size .* ones(Float64, size(input))
    end

    # Pre-compute diff/composed interval bounds.
    # Used by: (a) old T_relax relaxation (per-ReLU bounds), (b) no_n1_encoding (output-layer bounds),
    #          (c) tighten_n2_bounds (derive N2 preact bounds from N1 + diff).
    if (use_relaxations && !no_n1_binaries_and_relaxtions_only_on_n2 && !no_n1_encoding_at_all) || no_n1_encoding_at_all || tighten_n2_bounds || no_n2_xp_encoding
        if diff_bounds_cache.valid
            restore_diff_bounds_from_cache!()
        else
            if use_zonotope
                compute_diff_bounds_zonotope(nn1, nn2, I_pert_prev_up, I_pert_prev_down; optimizing_intervals=optimizing_intervals)
            else
                compute_diff_and_comp_bounds(nn1, nn2, I_pert_prev_up, I_pert_prev_down; optimizing_intervals=optimizing_intervals)
            end
            save_diff_bounds_to_cache!()
        end
    end

    # Derive tighter N2 pre-activation bounds from N1 + diff
    if tighten_n2_bounds && !isempty(n1_preact_up_bounds) && !isempty(relu_diff_up_bounds)
        global n2_derived_preact_up_bounds   = Array{Float64}[]
        global n2_derived_preact_down_bounds = Array{Float64}[]
        n_tightened = 0
        n_total_split = 0
        for r in 1:length(n1_preact_up_bounds)
            derived_up   = vec(n1_preact_up_bounds[r])   .+ vec(relu_diff_up_bounds[r])
            derived_down = vec(n1_preact_down_bounds[r]) .+ vec(relu_diff_down_bounds[r])
            push!(n2_derived_preact_up_bounds, derived_up)
            push!(n2_derived_preact_down_bounds, derived_down)
            # Count how many neurons would flip from split to stable
            for i in eachindex(derived_up)
                if derived_up[i] > 0 && derived_down[i] < 0
                    n_total_split += 1
                end
                if derived_up[i] <= 0 || derived_down[i] >= 0
                    n_tightened += 1
                end
            end
        end
        println("tighten_n2_bounds: derived N2 preact bounds for $(length(n1_preact_up_bounds)) ReLU layers, $n_tightened neurons stable by derived bounds")
    end

    # Skip N1(x) encoding when no_n1_encoding_at_all is active
    if !no_n1_encoding_at_all
        println("Encoding N1(x)...")
        layer_counter = 0
        nueron_counter = 0
        network_version = "n1_org"
        v_out_n1 = v_in |> nn1
    else
        println("Skipping N1(x) encoding (--no_n1_encoding_at_all)")
        v_out_n1 = nothing
    end

    # Encode N2 on clean input x → layers K+1..2K
    println("Encoding N2(x)...")
    layer_counter = 0
    nueron_counter = 0
    network_version = "n2_org"
    v_n2_last_hidden = nothing
    if encode_n1_last_layer
        # Split forward pass to capture N2's last hidden layer variables.
        # Pipe through all layers up to (but not including) the last Linear,
        # save the intermediate result, then apply the last Linear.
        # This produces identical MIP variables/constraints as the full pipe.
        n2_layers = nn2.layers
        last_linear_idx = findlast(l -> occursin("Linear", string(typeof(l))), n2_layers)
        v_temp = v_in
        for i in 1:(last_linear_idx - 1)
            v_temp = v_temp |> n2_layers[i]
        end
        v_n2_last_hidden = v_temp
        v_out_n2 = v_temp |> n2_layers[last_linear_idx]
    else
        v_out_n2 = v_in |> nn2
    end

    # Only encode N1(x') when n1_p_mode is on and N1 is encoded
    if n1_p_mode && !no_n1_encoding_at_all
        println("Encoding N1(x_p)...")
        layer_counter = 0
        nueron_counter = 0
        network_version = "n1_pert"
        v_out_n1_p = v_x0 |> nn1
    else
        v_out_n1_p = nothing
    end

    # Pre-compute N2 perturbation bounds for N2(x_p)->N2(x) relaxation
    if no_n1_binaries_and_relaxtions_only_on_n2
        compute_n2_pert_relaxation_bounds(nn2, I_pert_prev_up, I_pert_prev_down)
    end

    # Encode N2 on perturbed input x_p → layers 3K+1..4K
    if !no_n2_xp_encoding
        println("Encoding N2(x_p)...")
        layer_counter = 0
        nueron_counter = 0
        network_version = "n2_pert"
        v_out_n2_p = v_x0 |> nn2
    else
        println("Skipping N2(x_p) encoding (--no_n2_xp_encoding)")
        v_out_n2_p = nothing
    end

    return Dict(
        :v_in => v_in,
        :v_in_p => v_x0,
        :Perturbation => v_e,
        :v_out_n1 => v_out_n1,
        :v_out_n2 => v_out_n2,
        :v_out_n1_p => v_out_n1_p,
        :v_out_n2_p => v_out_n2_p,
        :v_n2_last_hidden => v_n2_last_hidden,
    )
end

# ============================================================
# Shared helper: initialise interval globals and run the four
# forward passes (N1/N2 × clean/perturbed) for transfer mode.
#
# I_pert_up / I_pert_down (passed in, same shape as `input`) are
# the INITIAL perturbation interval at the INPUT LAYER:
#
#   I_pert_prev_up[i]   = upper bound of  Δ[i] = x'[i] - x[i]
#   I_pert_prev_down[i] = lower bound of  Δ[i] = x'[i] - x[i]
#
# These globals are read by perturbed_interval_constraints()
# (perturbation_intervals.jl) which propagates them layer-by-layer
# through each Linear and ReLU, producing per-layer intervals on
# the DIFFERENCE between perturbed and clean activations, and
# then adds MIP constraints:
#
#   a_pert[l][i] ∈ [ a_clean[l][i] + I_pert_down[l][i],
#                    a_clean[l][i] + I_pert_up[l][i]  ]
#
# Tighter initial intervals → tighter propagated bounds → stronger
# tightening constraints → faster MIP solving.
# ============================================================
function _four_network_passes_transfer!(nn1, nn2, v_in, v_x0, input, I_pert_up, I_pert_down, n1_p_mode)
    global layer_counter, nueron_counter, network_version
    global I_pert_prev_up, I_pert_prev_down
    global all_bounds_of_original, all_bounds_of_perturbation
    global I_z_prev_up, I_z_prev_up_perturbation
    global I_z_prev_down, I_z_prev_down_perturbation

    # all_bounds_of_original[l]    = [upper, lower] activation bounds for N1(x)  at layer l
    # all_bounds_of_perturbation[l] = [upper, lower] activation bounds for N1(x') at layer l
    # Used by transfer_interval_constraints() to tighten N1 vs N2 interval constraints.
    all_bounds_of_original    = []
    all_bounds_of_perturbation = []
    if size(input)[4] > 1
        append!(all_bounds_of_original,    [[ones(Float64, size(input)[4], 1), zeros(Float64, size(input)[4], 1)]])
        I_z_prev_up                   = zeros(Float64, size(input)[4], 1)
        I_z_prev_down                 = zeros(Float64, size(input)[4], 1)
        append!(all_bounds_of_perturbation, [[ones(Float64, size(input)[4], 1), zeros(Float64, size(input)[4], 1)]])
        I_z_prev_up_perturbation      = zeros(Float64, size(input)[4], 1)
        I_z_prev_down_perturbation    = zeros(Float64, size(input)[4], 1)
    else
        append!(all_bounds_of_original,    [[ones(Float64, size(input)), zeros(Float64, size(input))]])
        I_z_prev_up                   = zeros(Float64, size(input))
        I_z_prev_down                 = zeros(Float64, size(input))
        append!(all_bounds_of_perturbation, [[ones(Float64, size(input)), zeros(Float64, size(input))]])
        I_z_prev_up_perturbation      = zeros(Float64, size(input))
        I_z_prev_down_perturbation    = zeros(Float64, size(input))
    end

    # Store input-layer perturbation interval for perturbed_interval_constraints()
    I_pert_prev_up   = I_pert_up
    I_pert_prev_down = I_pert_down

    # Pre-compute diff/composed interval bounds.
    # Used by: (a) old T_relax relaxation (per-ReLU bounds), (b) no_n1_encoding (output-layer bounds),
    #          (c) tighten_n2_bounds (derive N2 preact bounds from N1 + diff),
    #          (d) no_n2_xp_encoding (output-layer pert bounds for N2(x')).
    if (use_relaxations && !no_n1_binaries_and_relaxtions_only_on_n2 && !no_n1_encoding_at_all) || no_n1_encoding_at_all || tighten_n2_bounds || no_n2_xp_encoding
        if diff_bounds_cache.valid
            restore_diff_bounds_from_cache!()
        else
            if use_zonotope
                compute_diff_bounds_zonotope(nn1, nn2, I_pert_prev_up, I_pert_prev_down; optimizing_intervals=optimizing_intervals)
            else
                compute_diff_and_comp_bounds(nn1, nn2, I_pert_prev_up, I_pert_prev_down; optimizing_intervals=optimizing_intervals)
            end
            save_diff_bounds_to_cache!()
        end
    end

    # Derive tighter N2 pre-activation bounds from N1 + diff
    if tighten_n2_bounds && !isempty(n1_preact_up_bounds) && !isempty(relu_diff_up_bounds)
        global n2_derived_preact_up_bounds   = Array{Float64}[]
        global n2_derived_preact_down_bounds = Array{Float64}[]
        n_tightened = 0
        for r in 1:length(n1_preact_up_bounds)
            derived_up   = vec(n1_preact_up_bounds[r])   .+ vec(relu_diff_up_bounds[r])
            derived_down = vec(n1_preact_down_bounds[r]) .+ vec(relu_diff_down_bounds[r])
            push!(n2_derived_preact_up_bounds, derived_up)
            push!(n2_derived_preact_down_bounds, derived_down)
            for i in eachindex(derived_up)
                if derived_up[i] <= 0 || derived_down[i] >= 0
                    n_tightened += 1
                end
            end
        end
        println("tighten_n2_bounds: derived N2 preact bounds for $(length(n1_preact_up_bounds)) ReLU layers, $n_tightened neurons stable by derived bounds")
    end

    # Skip N1(x) encoding when no_n1_encoding_at_all is active
    if !no_n1_encoding_at_all
        println("Encoding N1(x)...")
        layer_counter = 0; nueron_counter = 0; network_version = "n1_org"
        v_out_n1 = v_in |> nn1
    else
        println("Skipping N1(x) encoding (--no_n1_encoding_at_all)")
        v_out_n1 = nothing
    end

    println("Encoding N2(x)...")
    layer_counter = 0; nueron_counter = 0; network_version = "n2_org"
    v_n2_last_hidden = nothing
    if encode_n1_last_layer
        n2_layers = nn2.layers
        last_linear_idx = findlast(l -> occursin("Linear", string(typeof(l))), n2_layers)
        v_temp = v_in
        for i in 1:(last_linear_idx - 1)
            v_temp = v_temp |> n2_layers[i]
        end
        v_n2_last_hidden = v_temp
        v_out_n2 = v_temp |> n2_layers[last_linear_idx]
    else
        v_out_n2 = v_in |> nn2
    end

    # Only encode N1(x') when n1_p_mode is on and N1 is encoded
    if n1_p_mode && !no_n1_encoding_at_all
        println("Encoding N1(x_p)...")
        layer_counter = 0; nueron_counter = 0; network_version = "n1_pert"
        v_out_n1_p = v_x0 |> nn1
    else
        v_out_n1_p = nothing
    end

    # Pre-compute N2 perturbation bounds for N2(x_p)->N2(x) relaxation
    if no_n1_binaries_and_relaxtions_only_on_n2
        compute_n2_pert_relaxation_bounds(nn2, I_pert_prev_up, I_pert_prev_down)
    end

    # Encode N2 on perturbed input x_p
    if !no_n2_xp_encoding
        println("Encoding N2(x_p)...")
        layer_counter = 0; nueron_counter = 0; network_version = "n2_pert"
        v_out_n2_p = v_x0 |> nn2
    else
        println("Skipping N2(x_p) encoding (--no_n2_xp_encoding)")
        v_out_n2_p = nothing
    end

    return v_out_n1, v_out_n2, v_out_n1_p, v_out_n2_p, v_n2_last_hidden
end

# ============================================================
# Transfer: brightness  x'[i] = x[i] + e,  e ∈ [0, ε]  (scalar, same for all pixels)
#
# I_pert_prev_up/down explanation:
#   Δ[i] = x'[i] - x[i] = e
#   Since e ∈ [0, ε]:  Δ[i] ∈ [0, ε]  for every pixel i.
#   → I_pert_up   = ε  (uniform upper bound)
#   → I_pert_down = 0  (uniform lower bound — brightness only adds light)
# ============================================================
function get_perturbation_specific_keys_brightness_transfer(perturbation_size, nn1::NeuralNet, nn2::NeuralNet, input::Array{<:Real}, m::Model, n1_p_mode::Bool)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e  = @variable(m, lower_bound = 0, upper_bound = p_size)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1),       input_range)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1+p_size), input_range)
    @constraint(m, v_x0 .== v_in .+ v_e)

    # Δ = e ∈ [0, ε] for every pixel — uniform, non-negative shift
    I_pert_up   = p_size .* ones(Float64, size(input))
    I_pert_down = zeros(Float64, size(input))

    v_out_n1, v_out_n2, v_out_n1_p, v_out_n2_p, v_n2_last_hidden =
        _four_network_passes_transfer!(nn1, nn2, v_in, v_x0, input, I_pert_up, I_pert_down, n1_p_mode)
    return Dict(:v_in => v_in, :v_in_p => v_x0, :Perturbation => v_e,
                :v_out_n1 => v_out_n1, :v_out_n2 => v_out_n2,
                :v_out_n1_p => v_out_n1_p, :v_out_n2_p => v_out_n2_p,
                :v_n2_last_hidden => v_n2_last_hidden)
end

# ============================================================
# Transfer: contrast  x'[i] = e * x[i],  e ∈ [1, 1+ε]  (scalar multiplier)
# Requires NonConvex=2 (bilinear constraint — set in get_model_transfer).
#
# I_pert_prev_up/down explanation:
#   Δ[i] = x'[i] - x[i] = (e - 1) * x[i]
#   Since e-1 ∈ [0, ε] and x[i] ∈ [0, 1]:
#     Δ[i] ∈ [0,  ε * 1] = [0, ε]  (over-approximation; tighter per-pixel
#     bound would be [0, ε*x[i]], but x[i] is a MIP variable, not a constant)
#   → I_pert_up   = ε  (uniform upper bound — contrast only brightens)
#   → I_pert_down = 0  (uniform lower bound — multiplier ≥ 1, so x'[i] ≥ x[i])
# ============================================================
function get_perturbation_specific_keys_contrast_transfer(perturbation_size, nn1::NeuralNet, nn2::NeuralNet, input::Array{<:Real}, m::Model, n1_p_mode::Bool)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e  = @variable(m, lower_bound = 1.0, upper_bound = 1+p_size)
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1),       input_range)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1+p_size), input_range)
    @constraint(m, v_x0 .== v_e .* v_in)

    # Δ = (e-1)*x[i] ∈ [0, ε] — over-approx since x[i] ≤ 1 and e-1 ≤ ε
    I_pert_up   = p_size .* ones(Float64, size(input))
    I_pert_down = zeros(Float64, size(input))

    v_out_n1, v_out_n2, v_out_n1_p, v_out_n2_p, v_n2_last_hidden =
        _four_network_passes_transfer!(nn1, nn2, v_in, v_x0, input, I_pert_up, I_pert_down, n1_p_mode)
    return Dict(:v_in => v_in, :v_in_p => v_x0, :Perturbation => v_e,
                :v_out_n1 => v_out_n1, :v_out_n2 => v_out_n2,
                :v_out_n1_p => v_out_n1_p, :v_out_n2_p => v_out_n2_p,
                :v_n2_last_hidden => v_n2_last_hidden)
end

# ============================================================
# Transfer: translation  x'[j] = x[j - offset]  (rigid shift by (t_down, t_right))
#                         x'[j] = 0              (zero-padded border pixels)
#
# I_pert_prev_up/down explanation:
#   Δ[j] = x'[j] - x[j]
#
#   Interior pixels (valid after shift):
#     x'[j] = x[j - offset]  where both x[j], x[j-offset] ∈ [0, 1]
#     → Δ[j] ∈ [-1, 1]  (difference of two values in [0,1])
#
#   Border pixels (zero-padded):
#     x'[j] = 0, x[j] ∈ [0, 1]
#     → Δ[j] = -x[j] ∈ [-1, 0]
#
#   Using [-1, 1] uniformly is a conservative (but sound) over-approximation
#   that covers both cases.  A tighter per-pixel bound is possible but would
#   require knowing which pixels are interior vs border, adding complexity.
#   → I_pert_up   =  1  (uniform upper bound)
#   → I_pert_down = -1  (uniform lower bound)
# ============================================================
function get_perturbation_specific_keys_translation_transfer(w_, h_, k_, perturbation_size, nn1::NeuralNet, nn2::NeuralNet, input::Array{<:Real}, m::Model, n1_p_mode::Bool)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    t_down  = Int(perturbation_size[1])
    t_right = Int(perturbation_size[2])
    k = k_; w = w_; h = h_; res = w * h
    # Interior pixels: x'[j+offset] = x[j]
    for i2 = 1:w-t_right
        for i1 = 1:h-t_down
            i = i1 + h*(i2-1)
            @constraint(m, v_x0[i+t_down+w*t_right] == v_in[i])
            if k == 3
                @constraint(m, v_x0[res+i+t_down+w*t_right] == v_in[res+i])
                @constraint(m, v_x0[2*res+i+t_down+w*t_right] == v_in[2*res+i])
            end
        end
    end
    # Border pixels: zero-padded
    for j = 1:t_down
        @constraint(m, [i=j:w:res], v_x0[i] == 0)
        if k == 3
            @constraint(m, [i=j+res:w:2*res], v_x0[i] == 0)
            @constraint(m, [i=j+2*res:w:3*res], v_x0[i] == 0)
        end
    end
    for j = 1:t_right
        @constraint(m, [i=1+w*(j-1):1:w*j], v_x0[i] == 0)
        if k == 3
            @constraint(m, [i=res+1+w*(j-1):1:res+w*j], v_x0[i] == 0)
            @constraint(m, [i=2*res+1+w*(j-1):1:2*res+w*j], v_x0[i] == 0)
        end
    end

    # Δ[j] ∈ [-1, 1] — conservative uniform bound covering all pixel cases
    I_pert_up   =  ones(Float64, size(input))
    I_pert_down = -ones(Float64, size(input))

    v_out_n1, v_out_n2, v_out_n1_p, v_out_n2_p, v_n2_last_hidden =
        _four_network_passes_transfer!(nn1, nn2, v_in, v_x0, input, I_pert_up, I_pert_down, n1_p_mode)
    return Dict(:v_in => v_in, :v_in_p => v_x0, :Perturbation => "None",
                :v_out_n1 => v_out_n1, :v_out_n2 => v_out_n2,
                :v_out_n1_p => v_out_n1_p, :v_out_n2_p => v_out_n2_p,
                :v_n2_last_hidden => v_n2_last_hidden)
end

# ============================================================
# Transfer: patch  x'[i] ∈ [x[i]-ε, x[i]+ε]  for i in the patch region
#                  x'[i] = x[i]                 for i outside the patch
#
# I_pert_prev_up/down explanation:
#   Δ[i] = x'[i] - x[i]
#
#   Patch pixels (indices in `l`):
#     x'[i] - x[i] ∈ [-ε, ε]  (free additive noise bounded by ε)
#     → I_pert_up[i]   =  ε
#     → I_pert_down[i] = -ε
#
#   Non-patch pixels:
#     x'[i] = x[i]  ⟹  Δ[i] = 0  (no perturbation outside patch)
#     → I_pert_up[i]   = 0
#     → I_pert_down[i] = 0
#
#   This gives the tightest possible initial interval: only the patch
#   pixels carry uncertainty; everything else is fixed.
# ============================================================
function get_perturbation_specific_keys_patch_transfer(w_, h_, k_, perturbation_size, nn1::NeuralNet, nn2::NeuralNet, input::Array{<:Real}, m::Model, n1_p_mode::Bool)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    eps  = perturbation_size[1]
    ind1 = Int(perturbation_size[2])
    ind2 = Int(perturbation_size[3])
    w = w_; h = h_; k = k_; res_ = w * h
    ind  = ind1 + (ind2-1)*w
    l = Int[]  # flat pixel indices inside the patch
    for i_ in 0:Int(perturbation_size[4])-1
        for j_ in 0:Int(perturbation_size[4])-1
            push!(l, ind+j_+w*i_)
            if k == 3
                push!(l, res_+ind+j_+w*i_)
                push!(l, 2*res_+ind+j_+w*i_)
            end
        end
    end
    res = [tt for tt in 1:Int(w_*h_*k_) if !(tt in l)]
    @constraint(m, c0[i=l], v_x0[i] <= v_in[i] + eps)
    @constraint(m, c1[i=l], v_x0[i] >= v_in[i] - eps)
    @constraint(m, c2[i=res], v_x0[i] == v_in[i])

    # Per-pixel interval: ±ε inside patch, 0 outside
    flat_up   = zeros(Float64, w_*h_*k_)
    flat_down = zeros(Float64, w_*h_*k_)
    for i in l; flat_up[i] = eps; flat_down[i] = -eps; end
    I_pert_up   = reshape(flat_up,   size(input))
    I_pert_down = reshape(flat_down, size(input))

    v_out_n1, v_out_n2, v_out_n1_p, v_out_n2_p, v_n2_last_hidden =
        _four_network_passes_transfer!(nn1, nn2, v_in, v_x0, input, I_pert_up, I_pert_down, n1_p_mode)
    return Dict(:v_in => v_in, :v_in_p => v_x0, :Perturbation => "None",
                :v_out_n1 => v_out_n1, :v_out_n2 => v_out_n2,
                :v_out_n1_p => v_out_n1_p, :v_out_n2_p => v_out_n2_p,
                :v_n2_last_hidden => v_n2_last_hidden)
end

# ============================================================
# Transfer: occ  x'[i] = 0          for i in occluded patch
#                x'[i] = x[i]        for i outside patch
#
# I_pert_prev_up/down explanation:
#   Δ[i] = x'[i] - x[i]
#
#   Occluded pixels (indices in `l`):
#     x'[i] = 0, x[i] ∈ [0, 1]
#     → Δ[i] = 0 - x[i] = -x[i] ∈ [-1, 0]
#     → I_pert_up[i]   = 0   (occlusion can only darken, never brighten)
#     → I_pert_down[i] = -1  (worst case: x[i]=1 was fully white, now 0)
#
#   Non-occluded pixels:
#     x'[i] = x[i]  ⟹  Δ[i] = 0
#     → I_pert_up[i]   = 0
#     → I_pert_down[i] = 0
#
#   This is the tightest possible sound bound: occluded pixels can only
#   decrease (Δ ≤ 0), so the upper interval is exactly 0.
# ============================================================
function get_perturbation_specific_keys_occ_transfer(w_, h_, k_, perturbation_size, nn1::NeuralNet, nn2::NeuralNet, input::Array{<:Real}, m::Model, n1_p_mode::Bool)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    ind1 = Int(perturbation_size[1])
    ind2 = Int(perturbation_size[2])
    w = w_; h = h_; k = k_; res_ = w * h
    ind  = ind1 + (ind2-1)*w
    l = Int[]  # flat pixel indices inside the occluded patch
    for i_ in 0:Int(perturbation_size[3])-1
        for j_ in 0:Int(perturbation_size[3])-1
            push!(l, ind+j_+w*i_)
            if k == 3
                push!(l, res_+ind+j_+w*i_)
                push!(l, 2*res_+ind+j_+w*i_)
            end
        end
    end
    res = [tt for tt in 1:Int(w_*h_*k_) if !(tt in l)]
    @constraint(m, c1[i=l],   v_x0[i] == 0.0)
    @constraint(m, c2[i=res], v_x0[i] == v_in[i])

    # Per-pixel interval: occluded → Δ ∈ [-1, 0]; non-occluded → Δ = 0
    flat_up   = zeros(Float64, w_*h_*k_)
    flat_down = zeros(Float64, w_*h_*k_)
    for i in l; flat_down[i] = -1.0; end  # upper stays 0: occlusion only removes signal
    I_pert_up   = reshape(flat_up,   size(input))
    I_pert_down = reshape(flat_down, size(input))

    v_out_n1, v_out_n2, v_out_n1_p, v_out_n2_p, v_n2_last_hidden =
        _four_network_passes_transfer!(nn1, nn2, v_in, v_x0, input, I_pert_up, I_pert_down, n1_p_mode)
    return Dict(:v_in => v_in, :v_in_p => v_x0, :Perturbation => "None",
                :v_out_n1 => v_out_n1, :v_out_n2 => v_out_n2,
                :v_out_n1_p => v_out_n1_p, :v_out_n2_p => v_out_n2_p,
                :v_n2_last_hidden => v_n2_last_hidden)
end


# ============================================================
# Transfer: rotation  x' = rotate(x, angle)  using bilinear interpolation
#
# I_pert_prev_up/down explanation:
#   Δ[i] = x'[i] - x[i]
#
#   Mapped pixels (rotation lands inside image):
#     x'[i] is a bilinear combination of x values ∈ [0,1], and x[i] ∈ [0,1],
#     so Δ[i] ∈ [-1, 1].  Conservative uniform bound.
#
#   Zero-padded pixels (rotation lands outside image):
#     x'[i] = 0, x[i] ∈ [0,1], so Δ[i] = -x[i] ∈ [-1, 0].
# ============================================================
function get_perturbation_specific_keys_rotate_transfer(w_, h_, k_, perturbation_size, nn1::NeuralNet, nn2::NeuralNet, input::Array{<:Real}, m::Model, n1_p_mode::Bool)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range)
    angle = perturbation_size[1]
    k = k_
    height = h_
    width = w_
    res_ = h_ * w_
    center = [width / 2, height / 2]
    mapped = Int[]  # flat pixel indices that map inside the image

    for i = 1:height
        for j = 1:width
            j_c = j - center[1]
            i_c = i - center[2]
            j_r = j_c * cos(angle * pi / 180) - i_c * sin(angle * pi / 180) + center[1]
            i_r = j_c * sin(angle * pi / 180) + i_c * cos(angle * pi / 180) + center[2]
            if floor(Int, j_r) >= 1 && ceil(Int, j_r) <= width && floor(Int, i_r) >= 1 && ceil(Int, i_r) <= height
                di = i_r - floor(i_r)
                dj = j_r - floor(j_r)
                # Bilinear interpolation constraint for channel 1
                @constraint(m, v_x0[i+(j-1)*height] ==
                    (1-di)*(1-dj)*v_in[floor(Int,i_r)+(floor(Int,j_r)-1)*height] +
                    (di)*(1-dj)*v_in[ceil(Int,i_r)+(floor(Int,j_r)-1)*height] +
                    (1-di)*(dj)*v_in[floor(Int,i_r)+(ceil(Int,j_r)-1)*height] +
                    (di)*(dj)*v_in[ceil(Int,i_r)+(ceil(Int,j_r)-1)*height])
                push!(mapped, i+(j-1)*height)
                if k == 3
                    @constraint(m, v_x0[i+(j-1)*height+res_] ==
                        (1-di)*(1-dj)*v_in[floor(Int,i_r)+(floor(Int,j_r)-1)*height+res_] +
                        (di)*(1-dj)*v_in[ceil(Int,i_r)+(floor(Int,j_r)-1)*height+res_] +
                        (1-di)*(dj)*v_in[floor(Int,i_r)+(ceil(Int,j_r)-1)*height+res_] +
                        (di)*(dj)*v_in[ceil(Int,i_r)+(ceil(Int,j_r)-1)*height+res_])
                    push!(mapped, i+(j-1)*height+res_)
                    @constraint(m, v_x0[i+(j-1)*height+2*res_] ==
                        (1-di)*(1-dj)*v_in[floor(Int,i_r)+(floor(Int,j_r)-1)*height+2*res_] +
                        (di)*(1-dj)*v_in[ceil(Int,i_r)+(floor(Int,j_r)-1)*height+2*res_] +
                        (1-di)*(dj)*v_in[floor(Int,i_r)+(ceil(Int,j_r)-1)*height+2*res_] +
                        (di)*(dj)*v_in[ceil(Int,i_r)+(ceil(Int,j_r)-1)*height+2*res_])
                    push!(mapped, i+(j-1)*height+2*res_)
                end
            end
        end
    end
    # Zero-pad pixels whose rotated source falls outside the image
    for tt in 1:res_
        if !(tt in mapped)
            @constraint(m, v_x0[tt] == 0)
        end
    end

    # Per-pixel perturbation intervals
    flat_up   = ones(Float64, Int(w_*h_*k_))   # mapped pixels: Δ ∈ [-1, 1]
    flat_down = -ones(Float64, Int(w_*h_*k_))
    # Zero-padded pixels: x' = 0, so Δ = -x ∈ [-1, 0] (tighter upper bound)
    for tt in 1:Int(w_*h_*k_)
        if !(tt in mapped)
            flat_up[tt] = 0.0    # Δ_up = 0 (x' is fixed at 0, can't exceed x)
        end
    end
    I_pert_up   = reshape(flat_up,   size(input))
    I_pert_down = reshape(flat_down, size(input))

    v_out_n1, v_out_n2, v_out_n1_p, v_out_n2_p, v_n2_last_hidden =
        _four_network_passes_transfer!(nn1, nn2, v_in, v_x0, input, I_pert_up, I_pert_down, n1_p_mode)
    return Dict(:v_in => v_in, :v_in_p => v_x0, :Perturbation => "None",
                :v_out_n1 => v_out_n1, :v_out_n2 => v_out_n2,
                :v_out_n1_p => v_out_n1_p, :v_out_n2_p => v_out_n2_p,
                :v_n2_last_hidden => v_n2_last_hidden)
end


function get_perturbation_specific_keys_rotate(w_, h_, k_, perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    input_range = CartesianIndices(size(input))
    v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    angle = perturbation_size[1]
    k = k_
    height = h_
    width = w_
    res_ = h_*w_
    center = [width/2, height/2]
    l = []
    for i = 1:height
        for j = 1:width
            j_c = j - center[1]
            i_c = i - center[2]
            j_r = (j_c*cos(angle*pi/180) - i_c*sin(angle*pi/180) + center[1])
            i_r = (j_c*sin(angle*pi/180) + i_c*cos(angle*pi/180) + center[2])
            if floor(Int,j_r) >= 1 && ceil(Int,j_r) <= width && floor(Int,i_r) >= 1 && ceil(Int,i_r) <= height
                di = i_r-floor(i_r)
                dj = j_r-floor(j_r)
                @constraint(m,v_x0[i+(j-1)*height] == (1-di)*(1-dj)*v_in[floor(Int,i_r)+(floor(Int,j_r)-1)*height]+
                (di)*(1-dj)*v_in[ceil(Int,i_r)+(floor(Int,j_r)-1)*height]+(1-di)*(dj)*v_in[floor(Int,i_r)+(ceil(Int,j_r)-1)*height]+
                (di)*(dj)*v_in[ceil(Int,i_r)+(ceil(Int,j_r)-1)*height])
                append!(l, i+(j-1)*height)
                if k==3
                    @constraint(m,v_x0[i+(j-1)*height+res_] == (1-di)*(1-dj)*v_in[floor(Int,i_r)+(floor(Int,j_r)-1)*height+res_]+
                    (di)*(1-dj)*v_in[ceil(Int,i_r)+(floor(Int,j_r)-1)*height+res_]+(1-di)*(dj)*v_in[floor(Int,i_r)+(ceil(Int,j_r)-1)*height+res_]+
                    (di)*(dj)*v_in[ceil(Int,i_r)+(ceil(Int,j_r)-1)*height+res_])
                    append!(l, i+(j-1)*height+res_)
                    @constraint(m,v_x0[i+(j-1)*height+2*res_] == (1-di)*(1-dj)*v_in[floor(Int,i_r)+(floor(Int,j_r)-1)*height+2*res_]+
                    (di)*(1-dj)*v_in[ceil(Int,i_r)+(floor(Int,j_r)-1)*height+2*res_]+(1-di)*(dj)*v_in[floor(Int,i_r)+(ceil(Int,j_r)-1)*height+2*res_]+
                    (di)*(dj)*v_in[ceil(Int,i_r)+(ceil(Int,j_r)-1)*height+2*res_])
                    append!(l, i+(j-1)*height+2*res_)
                end
            end
        end
    end
    for tt in 1:res_
        if tt in l
            continue
        end
        @constraint(m,v_x0[tt] == 0)
    end

    # Perturbation interval bounds:
    #   rotated (interior) pixels: bilinear interp, Δ ∈ [-1, 1]
    #   border (zeroed) pixels: Δ ∈ [-1, 0]
    global I_pert_prev_up, I_pert_prev_down
    I_pert_prev_up = ones(Float64, size(input))
    I_pert_prev_down = -ones(Float64, size(input))

    global layer_counter, nueron_counter, network_version
    layer_counter = 0
    nueron_counter = 0
    network_version = "org"
    v_in_output = v_in |> nn
    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => "None", :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end


