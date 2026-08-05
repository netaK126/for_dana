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
    global geometric_diff_map = nothing   # cleared each build; set by the translation/rotation encoder when the flag is on
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
    elseif perturbation == "translation"
        return merge(d_common, get_perturbation_specific_keys_translation(w_, h_, k_, perturbation_size,nn, input, m))
    elseif perturbation == "rotation"
        return merge(d_common, get_perturbation_specific_keys_rotate(w_, h_, k_, perturbation_size,nn, input, m))
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
    # x' is box-bounded here, so the interval passes may clip to the domain.
    global perturbed_input_in_domain = true
    if internet_nets_benchmarks
        # ACAS/HAR: use the per-coordinate input box. A missing box under the
        # flag is a misconfiguration (e.g. absent <model>_box.txt sidecar), so
        # fail loudly rather than silently reverting to [0,1].
        (input_box_lo === nothing || input_box_hi === nothing) &&
            error("--internet_nets_benchmarks is on but no input box was loaded; " *
                  "check the <model>_box.txt sidecar next to model_path.")
        v_in = map(i -> @variable(m, lower_bound = input_box_lo[i], upper_bound = input_box_hi[i]), input_range,)
        v_x0 = map(i -> @variable(m, lower_bound = input_box_lo[i], upper_bound = input_box_hi[i]), input_range,)
    else
        v_in = map( i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
        v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    end
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

    layer_counter = 0
    nueron_counter = 0
    network_version = "perturbation"
    v_output = v_x0 |> nn
    return Dict(:v_in_p => v_x0, :Perturbation => v_e, :v_out_p => v_output, :v_in => v_in, :v_out => v_in_output)
end

function get_perturbation_specific_keys_brightness(perturbation_size, nn::NeuralNet, input::Array{<:Real}, m::Model,)::Dict{Symbol,Any}
    global layer_counter
    global nueron_counter
    global network_version
    global I_pert_prev_up, I_pert_prev_down
    input_range = CartesianIndices(size(input))
    p_size = perturbation_size[1]
    v_e = @variable(m, lower_bound = 0, upper_bound = p_size)
    global perturbed_input_in_domain = true
    if internet_nets_benchmarks
        (input_box_lo === nothing || input_box_hi === nothing) &&
            error("--internet_nets_benchmarks is on but no input box was loaded; " *
                  "check the <model>_box.txt sidecar next to model_path.")
        v_in = map(i -> @variable(m, lower_bound = input_box_lo[i], upper_bound = input_box_hi[i]), input_range,)
        v_x0 = map(i -> @variable(m, lower_bound = input_box_lo[i], upper_bound = input_box_hi[i]), input_range,)
    else
        v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1),input_range,)
        v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    end
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
    # delta_max optimises over the whole input region. For ACAS [0,1] is almost
    # disjoint from the real box (coordinate 5 lives in [-0.5,-0.45]), so
    # without this the normaliser would be computed outside the net's domain.
    if internet_nets_benchmarks
        (input_box_lo === nothing || input_box_hi === nothing) &&
            error("--internet_nets_benchmarks is on but no input box was loaded; " *
                  "check the <model>_box.txt sidecar next to model_path.")
        v_in = map(i -> @variable(m, lower_bound = input_box_lo[i], upper_bound = input_box_hi[i]), input_range,)
    else
        v_in = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
    end
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
    v_x0 = map(i -> @variable(m, lower_bound = 0, upper_bound = 1), input_range,)
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

    # Per-pixel perturbation intervals:
    #   interior (shifted) pixels: Δ[dst] = x[src] - x[dst] ∈ [-1, 1]
    #     (both source and destination pixel values are unknown in [0,1],
    #      so the trivial bound is tight per-pixel)
    #   border (zero-padded) pixels: x'[j] = 0 ⟹ Δ[j] = -x[j] ∈ [-1, 0]
    #     (upper bound tightens from 1 to 0; sound because the @constraint
    #      blocks above pin exactly these indices to v_x0[j] = 0)
    flat_up   = ones(Float64,  Int(w*h*k))
    flat_down = -ones(Float64, Int(w*h*k))
    channel_offsets = (k == 3) ? (0, res, 2*res) : (0,)
    for ch_off in channel_offsets
        # Top t_down rows of each channel (pinned by the j=1:t_down loop above)
        for jj = 1:t_down
            for ii in jj:w:res
                flat_up[ii + ch_off] = 0.0
            end
        end
        # Left t_right columns of each channel (pinned by the j=1:t_right loop)
        for jj = 1:t_right
            for ii in (1+w*(jj-1)):(w*jj)
                flat_up[ii + ch_off] = 0.0
            end
        end
    end
    global I_pert_prev_up, I_pert_prev_down
    I_pert_prev_up   = reshape(flat_up,   size(input))
    I_pert_prev_down = reshape(flat_down, size(input))

    # --geometric_intervals: hand the move's exact (T-I) map to perturbed_interval_constraints
    # (geometric_interval_diff_bounds handles Flatten/Linear/Conv2d nets, FC and conv alike).
    global geometric_intervals, geometric_diff_map, geometric_input_shape
    if geometric_intervals && !isempty(nn.layers)
        geometric_diff_map = geometric_diff_map_translation(t_down, t_right, w, h, k)
        geometric_input_shape = size(input)
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
        if k==3
            @constraint(m,v_x0[tt+res_] == 0)
            @constraint(m,v_x0[tt+2*res_] == 0)
        end
    end

    # Per-pixel perturbation intervals:
    #   rotated (interior) pixels: Δ ∈ [-1, 1] (bilinear interp of unknowns)
    #   zero-padded pixels: x' = 0 ⟹ Δ = -x ∈ [-1, 0] (tighter up bound)
    # The zero-padding @constraint loop above pins every channel of a padded
    # pixel, so the tighter upper bound applies to all channels.
    flat_up   = ones(Float64,  Int(w_*h_*k_))
    flat_down = -ones(Float64, Int(w_*h_*k_))
    for tt in 1:res_
        if !(tt in l)
            flat_up[tt] = 0.0
            if k==3
                flat_up[tt+res_] = 0.0
                flat_up[tt+2*res_] = 0.0
            end
        end
    end
    global I_pert_prev_up, I_pert_prev_down
    I_pert_prev_up   = reshape(flat_up,   size(input))
    I_pert_prev_down = reshape(flat_down, size(input))

    # --geometric_intervals: hand the single angle's exact bilinear (T-I) map to perturbed_interval_constraints.
    # The map is block-diagonal over channels (the encoder pins all k channels), and
    # geometric_interval_diff_bounds handles Flatten/Linear/Conv2d nets alike.
    global geometric_intervals, geometric_diff_map, geometric_input_shape
    if geometric_intervals && !isempty(nn.layers)
        geometric_diff_map = geometric_diff_map_rotation(angle, w_, h_, k_)
        geometric_input_shape = size(input)
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


