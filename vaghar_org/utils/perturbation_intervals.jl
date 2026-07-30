# Interval bound propagation for the transfer proof.
#
# Same approach as lucid_delta_diff_with_perturbation:
# - Uses globals: I_z_prev_up/down, I_z_prev_up/down_perturbation,
#   all_bounds_of_original/perturbation
#
# compute_diff_bounds_zonotope() also populates the relaxation globals
# (relu_diff_*_bounds, n1_preact_*_bounds) used by
# the conditional-triangle relaxation in core_ops.jl.
# - Initialized in get_perturbation_specific_keys_linf_transfer()
# - Propagation (paper eq. 3): W2·(z_org - z_pre) + ΔW·z_pre, where ΔW = W2 - W1
# - Constraints: av[vec_n2] <= av[vec_n1] .+ I_z_prev_up_to_use
#                av[vec_n2] >= av[vec_n1] .+ I_z_prev_down_to_use

# ── helpers ──────────────────────────────────────────────────────────────

function interval_matrix_vector_multiplication(w_h_low, w_h_high, I_z_prev_min, I_z_prev_max)
    n = size(w_h_low, 1)
    m = size(w_h_low, 2)
    result_min = zeros(n)
    result_max = zeros(n)
    for i in 1:n
        min_val = 0.0
        max_val = 0.0
        for j in 1:m
            prod1 = w_h_low[i, j] * I_z_prev_min[j]
            prod2 = w_h_low[i, j] * I_z_prev_max[j]
            prod3 = w_h_high[i, j] * I_z_prev_min[j]
            prod4 = w_h_high[i, j] * I_z_prev_max[j]
            min_val += minimum([prod1, prod2, prod3, prod4])
            max_val += maximum([prod1, prod2, prod3, prod4])
        end
        result_min[i] = min_val
        result_max[i] = max_val
    end
    return result_min, result_max
end

"""
    interval_conv2d_bounds(F_low, F_high, x_low, x_high, bias, stride, padding)

Interval arithmetic for Conv2d: computes [output_min, output_max] given
filter interval [F_low, F_high] and input interval [x_low, x_high].

Filter shape: (fh, fw, in_channels, out_channels)
Input shape:  (batch, height, width, in_channels)
Output shape: (batch, out_height, out_width, out_channels)
"""
function interval_conv2d_bounds(F_low::Array{Float64,4}, F_high::Array{Float64,4},
                                x_low::Array{Float64,4}, x_high::Array{Float64,4},
                                bias::Vector{Float64}, stride::Int, padding)
    (batch, in_h, in_w, in_ch) = size(x_low)
    (fh, fw, _, out_ch) = size(F_low)

    ((out_h, out_w), (fh_off, fw_off)) = compute_output_parameters(in_h, in_w, fh, fw, stride, padding)

    result_min = zeros(Float64, batch, out_h, out_w, out_ch)
    result_max = zeros(Float64, batch, out_h, out_w, out_ch)

    for b_idx in 1:batch
        for oh in 1:out_h
            for ow in 1:out_w
                for oc in 1:out_ch
                    lo = bias[oc]
                    hi = bias[oc]
                    for fhi in 1:fh
                        for fwi in 1:fw
                            for ic in 1:in_ch
                                x_row = (oh - 1) * stride + fhi - fh_off
                                x_col = (ow - 1) * stride + fwi - fw_off
                                if x_row >= 1 && x_row <= in_h && x_col >= 1 && x_col <= in_w
                                    xl = x_low[b_idx, x_row, x_col, ic]
                                    xh = x_high[b_idx, x_row, x_col, ic]
                                    fl = F_low[fhi, fwi, ic, oc]
                                    fhv = F_high[fhi, fwi, ic, oc]
                                    p1 = fl * xl; p2 = fl * xh
                                    p3 = fhv * xl; p4 = fhv * xh
                                    lo += min(p1, p2, p3, p4)
                                    hi += max(p1, p2, p3, p4)
                                end
                            end
                        end
                    end
                    result_min[b_idx, oh, ow, oc] = lo
                    result_max[b_idx, oh, ow, oc] = hi
                end
            end
        end
    end
    return result_min, result_max
end

"""
    ensure_1d(x)

Flatten a multi-dimensional array to 1D for constraint addition.
Variables in JuMP are stored flat; interval arrays must match.
"""
function ensure_1d(x)
    ndims(x) > 1 ? vec(x) : x
end

"""
    extract_neuron_index(var_name)

Extract the structural neuron index (last integer) from a JuMP variable name.
Variable names follow: {prefix}x_rect_layerCount{n}_neuronCount{n}_{layer}_{neuron}
Returns the neuron index (1-based) used to index into interval bound arrays.
"""
function extract_neuron_index(var_name::String)
    # The neuron index is the last integer after the last underscore
    last_under = findlast('_', var_name)
    return parse(Int, var_name[last_under+1:end])
end

"""
    add_matched_interval_constraints!(m, av, vec_a, vec_b, I_up_flat, I_down_flat)

Add interval constraints only for neurons that have x_rect variables in BOTH
network copies. Neurons are matched by their structural neuron index (extracted
from JuMP variable names). This handles the case where bound tightening produces
different split/fixed decisions for the two copies (common in CNN layers).

Constraints added:  av[b_idx] <= av[a_idx] + I_up_flat[neuron]
                    av[b_idx] >= av[a_idx] + I_down_flat[neuron]
"""
function add_matched_interval_constraints!(m, av, vec_a, vec_b, I_up_flat, I_down_flat)
    # Build neuron_index → av_index maps for both copies
    map_a = Dict{Int,Int}()
    for idx in vec_a
        ni = extract_neuron_index(JuMP.name(av[idx]))
        map_a[ni] = idx
    end
    map_b = Dict{Int,Int}()
    for idx in vec_b
        ni = extract_neuron_index(JuMP.name(av[idx]))
        map_b[ni] = idx
    end

    # Find common neurons and add constraints
    common = sort(collect(intersect(keys(map_a), keys(map_b))))
    count = 0
    for ni in common
        @constraint(m, av[map_b[ni]] <= av[map_a[ni]] + I_up_flat[ni])
        @constraint(m, av[map_b[ni]] >= av[map_a[ni]] + I_down_flat[ni])
        count += 2
    end
    return count
end



# ════════════════════════════════════════════════════════════════════════════════════════════════
# --geometric_intervals : relocation-aware INTERVAL bounds for translation/rotation (no zonotope).
# All gated behind the geometric_intervals flag; geometric_diff_map !== nothing only for translation/rotation.
# Supports Flatten/Linear/Conv2d nets (FC and conv alike) and both k==1 and k==3 (the encoders pin
# every channel of mapped and zero-padded pixels). δ is unchanged — this is a sound bound tightening.
# ════════════════════════════════════════════════════════════════════════════════════════════════

# (T-I) for translation: covered dst +1 at src, -1 at dst; padded dst -1 at dst. Exact transcription of the encoder.
function geometric_diff_map_translation(sd, sr, w, h, k)
    n = w*h*k; res = w*h
    channel_offsets = (k == 3) ? (0, res, 2*res) : (0,)
    S = zeros(Float64, n, n); covered = falses(n)
    for i2 = 1:w, i1 = 1:h
        di1 = i1 + sd; di2 = i2 + sr
        if 1 <= di1 <= h && 1 <= di2 <= w
            for off in channel_offsets
                src = off + i1 + h*(i2-1); dst = off + di1 + h*(di2-1)
                S[dst, src] += 1.0; S[dst, dst] -= 1.0; covered[dst] = true
            end
        end
    end
    for dst in 1:n; if !covered[dst]; S[dst, dst] = -1.0; end; end
    return S
end

# (T-I) for rotation: accumulate the 4 bilinear weights at the 4 neighbours, then subtract identity.
# The same spatial map applies to every channel (block-diagonal over k), matching the encoder, which
# pins all k channels of mapped and zero-padded pixels alike.
function geometric_diff_map_rotation(angle, w, h, k)
    n = w*h*k; res = w*h; center = [w/2, h/2]
    channel_offsets = (k == 3) ? (0, res, 2*res) : (0,)
    T = zeros(Float64, n, n)
    for i = 1:h, j = 1:w
        j_c = j - center[1]; i_c = i - center[2]
        j_r = j_c*cos(angle*pi/180) - i_c*sin(angle*pi/180) + center[1]
        i_r = j_c*sin(angle*pi/180) + i_c*cos(angle*pi/180) + center[2]
        if floor(Int,j_r)>=1 && ceil(Int,j_r)<=w && floor(Int,i_r)>=1 && ceil(Int,i_r)<=h
            di = i_r-floor(i_r); dj = j_r-floor(j_r)
            fi=floor(Int,i_r); ci=ceil(Int,i_r); fj=floor(Int,j_r); cj=ceil(Int,j_r)
            for off in channel_offsets
                dst = off + i + (j-1)*h
                T[dst, off+fi+(fj-1)*h] += (1-di)*(1-dj);  T[dst, off+ci+(fj-1)*h] += (di)*(1-dj)
                T[dst, off+fi+(cj-1)*h] += (1-di)*(dj);    T[dst, off+ci+(cj-1)*h] += (di)*(dj)
            end
        end
    end
    for dst in 1:n; T[dst, dst] -= 1.0; end
    return T
end

# Push one (T-I) column through the affine prefix (the layers before the first ReLU) WITHOUT biases:
# biases cancel in the difference N(x') - N(x), so the diff map is the prefix's linear part alone.
# Conv2d reuses interval_conv2d_bounds with a degenerate interval (low == high == col), which is an
# exact convolution; Linear is a bias-free matvec; Flatten is the layer's own numeric forward.
function _geometric_prefix_linear_part(prefix, col)
    x = col
    for l in prefix
        t = string(typeof(l))
        if occursin("Flatten", t)
            x = x |> l
        elseif occursin("Linear", t)
            x = Float64.(transpose(l.matrix)) * x
        elseif occursin("Conv", t)
            F = Float64.(l.filter)
            zero_bias = zeros(Float64, length(l.bias))
            x4 = ndims(x) == 4 ? x : reshape(x, 1, :, 1, 1)
            lo, _ = interval_conv2d_bounds(F, F, x4, x4, zero_bias, l.stride, l.padding)
            x = lo
        else
            error("geometric_interval_diff_bounds: unsupported prefix layer $(t) (Flatten/Linear/Conv2d only)")
        end
    end
    return x
end

# Per-ReLU PRE-activation diff bounds for a geometric (T-I) move. Up to the first ReLU the diff is an
# exact linear map of the shared input, M = A*(T-I) with A the net's affine prefix; each (T-I) column
# streams through the prefix and the exact box min/max over v_in ∈ [0,1] accumulates per output neuron
# (positive/negative parts of M's rows). Past the first ReLU the diff is no longer linear, so the
# bounds interval-propagate through the remaining layers (bias-free: biases cancel in a difference)
# with the standard ReLU diff clip. Supports Flatten/Linear/Conv2d nets, FC and conv alike.
function geometric_interval_diff_bounds(nn, TmI, input_shape)
    relu_up = Array{Float64}[]; relu_dn = Array{Float64}[]
    first_relu = findfirst(l -> occursin("ReLU", string(typeof(l))), nn.layers)
    first_relu === nothing && return relu_up, relu_dn
    prefix = nn.layers[1:first_relu-1]

    diff_up = nothing; diff_dn = nothing
    for i in 1:size(TmI, 2)
        out = _geometric_prefix_linear_part(prefix, reshape(Float64.(TmI[:, i]), input_shape))
        if diff_up === nothing
            diff_up = max.(0.0, out); diff_dn = min.(0.0, out)
        else
            diff_up .+= max.(0.0, out); diff_dn .+= min.(0.0, out)
        end
    end

    # First ReLU: record the PRE-activation diff bounds, then clip for propagation.
    push!(relu_up, copy(diff_up)); push!(relu_dn, copy(diff_dn))
    diff_up = max.(0.0, diff_up); diff_dn = .- max.(0.0, .- diff_dn)

    for l in nn.layers[first_relu+1:end]
        t = string(typeof(l))
        if occursin("Flatten", t)
            if ndims(diff_up) > 1
                diff_up = diff_up |> l; diff_dn = diff_dn |> l
            end
        elseif occursin("Linear", t)
            W = Float64.(transpose(l.matrix))
            diff_dn, diff_up = interval_matrix_vector_multiplication(W, W, diff_dn, diff_up)
        elseif occursin("Conv", t)
            F = Float64.(l.filter)
            zero_bias = zeros(Float64, length(l.bias))
            up4 = ndims(diff_up) == 4 ? diff_up : reshape(diff_up, 1, :, 1, 1)
            dn4 = ndims(diff_dn) == 4 ? diff_dn : reshape(diff_dn, 1, :, 1, 1)
            diff_dn, diff_up = interval_conv2d_bounds(F, F, dn4, up4, zero_bias, l.stride, l.padding)
        elseif occursin("ReLU", t)
            push!(relu_up, copy(diff_up)); push!(relu_dn, copy(diff_dn))
            diff_up = max.(0.0, diff_up); diff_dn = .- max.(0.0, .- diff_dn)
        else
            error("geometric_interval_diff_bounds: unsupported layer $(t) (Flatten/Linear/Conv2d/ReLU only)")
        end
    end
    return relu_up, relu_dn
end

# Couple clean vs perturbed post-ReLU x_rect from the per-ReLU PRE-activation diff bounds (post-ReLU relaxation
# [min(0,dn), max(0,up)]). Mirrors the coupling perturbed_interval_constraints adds in the plain path.
function add_geometric_coupling(m, nn, clean_prefix, pert_prefix, relu_up, relu_dn)
    av = JuMP.all_variables(m); layer_cnt = 0; constraints_added = 0
    for l in nn.layers
        if occursin("ReLU", string(typeof(l)))
            layer_cnt += 1
            (layer_cnt <= length(relu_up)) || continue
            post_up = max.(0.0, relu_up[layer_cnt]); post_dn = .- max.(0.0, .- relu_dn[layer_cnt])
            vec_clean = Int[]; vec_pert = Int[]
            for nidx in eachindex(av)
                nm = JuMP.name(av[nidx])
                if occursin(clean_prefix*"x_rect_layerCount"*string(layer_cnt)*"_", nm); push!(vec_clean, nidx)
                elseif occursin(pert_prefix*"x_rect_layerCount"*string(layer_cnt)*"_", nm); push!(vec_pert, nidx) end
            end
            if !isempty(vec_clean) && !isempty(vec_pert)
                constraints_added += add_matched_interval_constraints!(m, av, vec_clean, vec_pert,
                                                                       ensure_1d(post_up), ensure_1d(post_dn))
            end
        end
    end
    println("  geometric-interval coupling: $constraints_added constraints ($(clean_prefix)/$(pert_prefix))")
    return constraints_added
end


function perturbed_interval_constraints(m, nn, clean_prefix, pert_prefix)
    global geometric_intervals, geometric_diff_map, geometric_input_shape
    if geometric_intervals && geometric_diff_map !== nothing
        gup, gdn = geometric_interval_diff_bounds(nn, geometric_diff_map, geometric_input_shape)
        return add_geometric_coupling(m, nn, clean_prefix, pert_prefix, gup, gdn)
    end
    return perturbed_interval_constraints_plain(m, nn, clean_prefix, pert_prefix)
end

function perturbed_interval_constraints_plain(m, nn, clean_prefix, pert_prefix)
    global I_pert_prev_up
    global I_pert_prev_down

    av = JuMP.all_variables(m)
    layer_cnt = 0
    constraints_added = 0

    # Local copy of I_pert for propagation
    I_pert_up_local   = copy(I_pert_prev_up)
    I_pert_down_local = copy(I_pert_prev_down)

    for (layer_idx, l) in enumerate(nn.layers)
        if occursin("Flatten", string(typeof(l)))
            if ndims(I_pert_up_local) > 1
                I_pert_up_local   = I_pert_up_local |> l
                I_pert_down_local = I_pert_down_local |> l
            end

        elseif occursin("Linear", string(typeof(l)))
            # W = W1^T (same network for both clean and perturbed)
            W = Float64.(transpose(l.matrix))

            # Δz_{m,k} = Σ w_{m,k,k'} · Δa_{m-1,k'}
            # Since weights are fixed: interval_mul(W, W, I_down, I_up)
            r_min, r_max = interval_matrix_vector_multiplication(W, W, I_pert_down_local, I_pert_up_local)
            I_pert_down_local = r_min
            I_pert_up_local   = r_max

        elseif occursin("Conv", string(typeof(l)))
            F = Float64.(l.filter)
            zero_bias = zeros(Float64, length(l.bias))

            I_up_4d   = ndims(I_pert_up_local) == 4 ? I_pert_up_local : reshape(I_pert_up_local, 1, :, 1, 1)
            I_down_4d = ndims(I_pert_down_local) == 4 ? I_pert_down_local : reshape(I_pert_down_local, 1, :, 1, 1)

            r_min, r_max = interval_conv2d_bounds(F, F, I_down_4d, I_up_4d, zero_bias, l.stride, l.padding)
            I_pert_down_local = r_min
            I_pert_up_local   = r_max

        elseif occursin("ReLU", string(typeof(l)))
            layer_cnt += 1

            # ReLU relaxation on difference:
            # σ(z+Δ) - σ(z) ∈ [min(0, Δ_down), max(0, Δ_up)]
            I_pert_up_local   = max.(0.0, I_pert_up_local)
            I_pert_down_local = .- max.(0.0, .- I_pert_down_local)

            # Find clean x_rect variables by name
            vec_clean = []
            for n in eachindex(av)
                if occursin(clean_prefix * "x_rect" * "_" * "layerCount" * string(layer_cnt) * "_", JuMP.name(av[n]))
                    append!(vec_clean, n)
                end
            end

            # Find perturbed x_rect variables by name
            vec_pert = []
            for n in eachindex(av)
                if occursin(pert_prefix * "x_rect" * "_" * "layerCount" * string(layer_cnt) * "_", JuMP.name(av[n]))
                    append!(vec_pert, n)
                end
            end

            if length(vec_clean) > 0 && length(vec_pert) > 0
                I_up_flat   = ensure_1d(I_pert_up_local)
                I_down_flat = ensure_1d(I_pert_down_local)
                constraints_added += add_matched_interval_constraints!(
                    m, av, vec_clean, vec_pert, I_up_flat, I_down_flat)
            end
        end
    end

    return constraints_added
end


# ── Zonotope-based diff bound propagation ──────────────────────────────
# Computes the diff bounds with a zonotope (affine
# arithmetic) for the diff propagation through FC layers. Zonotope activates
# at Flatten (conv layers are propagated with interval arithmetic).
#
# A zonotope represents each neuron as: z_i = c_i + Σ_j g_{i,j} * ε_j
# where ε_j ∈ [-1,1]. Bounds: z_i ∈ [c_i - Σ|g_{i,j}|, c_i + Σ|g_{i,j}|].
# Linear layers preserve correlations: W*z has generators W*G.
# ReLU: stable neurons pass through; split neurons use DeepZ relaxation.
function compute_diff_bounds_zonotope(nn1, nn2, I_pert_up_init, I_pert_down_init; optimizing_intervals::Bool=true)
    # Same globals as the retired interval pass
    global relu_diff_up_bounds, relu_diff_down_bounds
    global n1_preact_up_bounds, n1_preact_down_bounds

    relu_diff_up_bounds   = Array{Float64}[]
    relu_diff_down_bounds = Array{Float64}[]
    n1_preact_up_bounds   = Array{Float64}[]
    n1_preact_down_bounds = Array{Float64}[]

    # Interval state (used for conv layers, pert, n1_act)
    diff_up   = zeros(Float64, size(I_pert_up_init))
    diff_down = zeros(Float64, size(I_pert_down_init))
    pert_up   = copy(Float64.(I_pert_up_init))
    pert_down = copy(Float64.(I_pert_down_init))
    n1_act_down, n1_act_up = input_domain_shaped(size(I_pert_up_init))
    n1_pre_up_cur   = Float64[]
    n1_pre_down_cur = Float64[]

    # Zonotope state for diff (activated at Flatten or first Linear on 1D input,
    # conv layers use interval arithmetic)
    zono_active = false
    diff_center = Float64[]
    diff_gens = Matrix{Float64}(undef, 0, 0)

    last_relu_diff_up   = zeros(Float64, size(diff_up))
    last_relu_diff_down = zeros(Float64, size(diff_down))

    for (layer_idx, l) in enumerate(nn1.layers)
        l2 = nn2.layers[layer_idx]

        if occursin("Flatten", string(typeof(l)))
            if ndims(diff_up) > 1
                diff_up   = vec(diff_up |> l)
                diff_down = vec(diff_down |> l)
                pert_up   = vec(pert_up |> l)
                pert_down = vec(pert_down |> l)
                n1_act_up   = vec(n1_act_up |> l)
                n1_act_down = vec(n1_act_down |> l)
            end
            # Convert diff intervals to zonotope (the zonotope always starts
            # here — conv layers upstream are propagated with intervals)
            n = length(diff_up)
            diff_center = (diff_up .+ diff_down) ./ 2
            diff_radius = (diff_up .- diff_down) ./ 2
            diff_gens = diagm(diff_radius)
            zono_active = true
            println("  Zonotope: activated at Flatten, $n dims, $n initial generators")

        elseif occursin("Linear", string(typeof(l)))
            W1 = Float64.(transpose(l.matrix))
            W2 = Float64.(transpose(l2.matrix))
            b1 = Float64.(l.bias)
            b2 = Float64.(l2.bias)
            ΔW = W2 - W1
            Δb = b2 - b1

            if !zono_active && ndims(diff_up) == 1
                # FC layer before any Flatten (pure FC network): activate zonotope
                n = length(diff_up)
                diff_center = (diff_up .+ diff_down) ./ 2
                diff_radius = (diff_up .- diff_down) ./ 2
                diff_gens = diagm(diff_radius)
                zono_active = true
                println("  Zonotope: activated at Linear (FC network), $n dims")
            end

            if zono_active
                # Zonotope propagation: diff = ΔW * a_n1 + W2 * diff_prev + Δb
                n1_center = (n1_act_up .+ n1_act_down) ./ 2
                n1_radius = (n1_act_up .- n1_act_down) ./ 2

                new_center = W2 * diff_center .+ ΔW * n1_center .+ Δb

                # Correlated part: preserve existing generators
                new_gens_corr = W2 * diff_gens
                # Independent part: ΔW * n1_radius (new generators from n1 intervals)
                new_gens_indep = ΔW * diagm(n1_radius)
                diff_gens = hcat(new_gens_corr, new_gens_indep)
                diff_center = new_center
                # Update interval bounds from zonotope
                abs_sum = vec(sum(abs.(diff_gens), dims=2))
                diff_up   = diff_center .+ abs_sum
                diff_down = diff_center .- abs_sum
            else
                # Interval propagation for conv layers
                r1_min, r1_max = interval_matrix_vector_multiplication(ΔW, ΔW, n1_act_down, n1_act_up)
                r2_min, r2_max = interval_matrix_vector_multiplication(W2, W2, diff_down, diff_up)
                diff_down = r1_min .+ r2_min .+ Δb
                diff_up   = r1_max .+ r2_max .+ Δb
            end

            # Pert always interval
            rp_min, rp_max = interval_matrix_vector_multiplication(W2, W2, pert_down, pert_up)
            pert_down = rp_min
            pert_up   = rp_max

            # N1 preact bounds
            rn_min, rn_max = interval_matrix_vector_multiplication(W1, W1, n1_act_down, n1_act_up)
            n1_pre_up_cur   = rn_max .+ b1
            n1_pre_down_cur = rn_min .+ b1

        elseif occursin("Conv", string(typeof(l)))
            F1 = Float64.(l.filter);  F2 = Float64.(l2.filter)
            b1 = Float64.(l.bias);    b2 = Float64.(l2.bias)
            ΔF = F2 - F1;             Δb = b2 - b1
            zero_bias = zeros(Float64, length(b1))

            n1_4d_down  = ndims(n1_act_down) == 4 ? n1_act_down : reshape(n1_act_down, 1, :, 1, 1)
            n1_4d_up    = ndims(n1_act_up)   == 4 ? n1_act_up   : reshape(n1_act_up,   1, :, 1, 1)
            pert_4d_down = ndims(pert_down)  == 4 ? pert_down   : reshape(pert_down,   1, :, 1, 1)
            pert_4d_up   = ndims(pert_up)    == 4 ? pert_up     : reshape(pert_up,     1, :, 1, 1)

                # Interval arithmetic for conv layers (original path)
                diff_4d_down = ndims(diff_down) == 4 ? diff_down : reshape(diff_down, 1, :, 1, 1)
                diff_4d_up   = ndims(diff_up)   == 4 ? diff_up   : reshape(diff_up,   1, :, 1, 1)

                r1_min, r1_max = interval_conv2d_bounds(ΔF, ΔF, n1_4d_down, n1_4d_up, zero_bias, l.stride, l.padding)
                r2_min, r2_max = interval_conv2d_bounds(F2, F2, diff_4d_down, diff_4d_up, zero_bias, l.stride, l.padding)
                diff_down = r1_min .+ r2_min
                diff_up   = r1_max .+ r2_max
                for oc in 1:length(Δb)
                    diff_down[:, :, :, oc] .+= Δb[oc]
                    diff_up[:, :, :, oc]   .+= Δb[oc]
                end

            # Pert and N1 preact always use interval arithmetic
            rp_min, rp_max = interval_conv2d_bounds(F2, F2, pert_4d_down, pert_4d_up, zero_bias, l.stride, l.padding)
            pert_down = rp_min
            pert_up   = rp_max

            rn_min, rn_max = interval_conv2d_bounds(F1, F1, n1_4d_down, n1_4d_up, Float64.(b1), l.stride, l.padding)
            n1_pre_up_cur   = rn_max
            n1_pre_down_cur = rn_min

        elseif occursin("ReLU", string(typeof(l)))
            # Store pre-ReLU bounds (same as the retired interval pass)
            push!(relu_diff_up_bounds,   copy(diff_up))
            push!(relu_diff_down_bounds, copy(diff_down))
            push!(n1_preact_up_bounds,   copy(n1_pre_up_cur))
            push!(n1_preact_down_bounds, copy(n1_pre_down_cur))

            if zono_active
                n = length(diff_center)

                new_gen_cols = Vector{Float64}[]

                for i in 1:n
                    l_d = diff_down[i]
                    u_d = diff_up[i]

                    # Compute N1/N2 pre-activation bounds when needed by
                    # optimizing_intervals
                    l_n1 = 0.0; u_n1 = 0.0; l_n2 = 0.0; u_n2 = 0.0
                    if optimizing_intervals
                        l_n1 = n1_pre_down_cur[i]
                        u_n1 = n1_pre_up_cur[i]
                        l_n2 = l_n1 + diff_down[i]
                        u_n2 = u_n1 + diff_up[i]
                    end

                    if optimizing_intervals
                        if l_n1 >= 0 && l_n2 >= 0
                            # Both active: diff passes through
                            continue
                        elseif u_n1 <= 0 && u_n2 <= 0
                            # Both inactive: diff = 0
                            diff_center[i] = 0.0
                            diff_gens[i, :] .= 0.0
                            continue
                        end
                    end

                    # Mixed case: apply DeepZ relaxation to diff
                    if l_d >= 0
                        # Diff always non-negative: pass through
                    elseif u_d <= 0
                        # Diff always non-positive: clip to 0
                        diff_center[i] = 0.0
                        diff_gens[i, :] .= 0.0
                    else
                        # Both split: generic DeepZ — λ * z + μ ± μ * ε_new
                        λ = u_d / (u_d - l_d)
                        μ = -u_d * l_d / (2.0 * (u_d - l_d))
                        diff_center[i] = λ * diff_center[i] + μ
                        diff_gens[i, :] .*= λ
                        # New generator for this neuron
                        new_col = zeros(n)
                        new_col[i] = μ
                        push!(new_gen_cols, new_col)
                    end
                end

                # Add new generator columns
                if !isempty(new_gen_cols)
                    new_gens_matrix = hcat(new_gen_cols...)
                    diff_gens = hcat(diff_gens, new_gens_matrix)
                end

                # Update interval bounds from zonotope
                abs_sum = vec(sum(abs.(diff_gens), dims=2))
                diff_up   = diff_center .+ abs_sum
                diff_down = diff_center .- abs_sum
                println("  Zonotope ReLU: $(size(diff_gens, 2)) generators, max diff width = $(maximum(diff_up .- diff_down))")
            else
                # Interval clipping (same as the retired interval pass)
                if optimizing_intervals
                    for i in eachindex(diff_up)
                        l_n1 = n1_pre_down_cur[i]
                        u_n1 = n1_pre_up_cur[i]
                        l_n2 = l_n1 + diff_down[i]
                        u_n2 = u_n1 + diff_up[i]
                        if l_n1 >= 0 && l_n2 >= 0
                            # pass
                        elseif u_n1 <= 0 && u_n2 <= 0
                            diff_up[i] = 0.0
                            diff_down[i] = 0.0
                        else
                            diff_up[i] = max(0.0, diff_up[i])
                            diff_down[i] = -max(0.0, -diff_down[i])
                        end
                    end
                else
                    diff_up   = max.(0.0, diff_up)
                    diff_down = .- max.(0.0, .- diff_down)
                end
            end

            pert_up   = max.(0.0, pert_up)
            pert_down = .- max.(0.0, .- pert_down)
            n1_act_up   = max.(0.0, n1_pre_up_cur)
            n1_act_down = max.(0.0, n1_pre_down_cur)

            last_relu_diff_up   = copy(diff_up)
            last_relu_diff_down = copy(diff_down)
        end
    end

    # Save all bounds (same globals as the retired interval pass)
    println("compute_diff_bounds_zonotope: output-layer diff bounds width = $(maximum(vec(Float64.(diff_up)) .- vec(Float64.(diff_down))))")
    if zono_active
        println("  Final zonotope: $(size(diff_gens, 2)) generators")
    end

    println("compute_diff_bounds_zonotope: populated $(length(relu_diff_up_bounds)) ReLU layers")
end

# ── advstd: absolute N2 zonotope bound propagation with N1 tightening ──
# Source B for the --adv_std_zono_bounds feature. Propagates a zonotope for
# N2's pre-activations directly (using N2's own weights W2) starting from an
# input zonotope that covers [0-ε, 1+ε] per input pixel. At every ReLU layer
# the zonotope hull is intersected with the "Source A" bound
# [n1_preact + diff_down, n1_preact + diff_up] BEFORE the DeepZ relaxation is
# applied, so every downstream layer's zonotope starts from a tighter box.
#
# Required populated globals (caller must have run compute_diff_bounds_zonotope
# before invoking this function):
#   relu_diff_up_bounds, relu_diff_down_bounds — per-ReLU-layer diff bounds
#   n1_preact_up_bounds, n1_preact_down_bounds — per-ReLU-layer N1 preact bounds
#
# Populates:
#   n2_abs_up_bounds, n2_abs_down_bounds — per-ReLU-layer N2 preact bounds,
#     each entry a Float64 vector (layer-flat shape). Consumed in core_ops.jl
#     inside the relu() bound-tightening block.
#
# Soundness: every step is an over-approximating abstract interpretation, and
# the per-layer intersection is the intersection of two sound over-approxima-
# tions of the same quantity, so the result over-approximates N2's true value
# set. Adding these scalar bounds inside the MIP big-M encoding therefore
# preserves N2's integer optimum.
function compute_n2_bounds_zonotope_with_n1_tighten(nn2, I_pert_up_init, I_pert_down_init;
                                                    use_n1_tighten::Bool=true)
    global n2_abs_up_bounds, n2_abs_down_bounds
    global relu_diff_up_bounds, relu_diff_down_bounds
    global n1_preact_up_bounds, n1_preact_down_bounds

    n2_abs_up_bounds   = Array{Float64}[]
    n2_abs_down_bounds = Array{Float64}[]

    # Standard-mode reuse: when Source A globals are empty (no N1→N2 diff to
    # intersect against), still propagate the absolute zonotope through the
    # network and store per-ReLU-layer bounds. The intersection step at the
    # ReLU loop already falls through to (u_hull, l_hull) when n1_preact /
    # relu_diff are empty (see the relu_layer_idx <= length(n1_preact_up_bounds)
    # guard below), so the zonotope-only output is sound.
    # use_n1_tighten=false ablates ONLY the N_pre contribution to this
    # zonotope (--adv_std_zono_npre=false): the absolute N2 zonotope below is
    # still propagated, but it is not intersected with N_pre's stored
    # pre-activation bounds or the N_pre->N difference zonotope. The globals
    # are left untouched, so the perturbation-difference technique that also
    # reads relu_diff_* is unaffected -- this ablates the zonotope's use of
    # N_pre, not N_pre itself.
    source_a_available = use_n1_tighten &&
        !isempty(n1_preact_up_bounds) && !isempty(relu_diff_up_bounds)
    if !use_n1_tighten
        println("compute_n2_bounds_zonotope_with_n1_tighten: N_pre tightening DISABLED (--adv_std_zono_npre=false); absolute N2 zonotope only")
    end
    if source_a_available && length(n1_preact_up_bounds) != length(relu_diff_up_bounds)
        println("compute_n2_bounds_zonotope_with_n1_tighten: n1_preact and relu_diff layer counts differ ($(length(n1_preact_up_bounds)) vs $(length(relu_diff_up_bounds))), skipping Source A intersect (Source B = absolute zonotope only)")
        source_a_available = false
    end
    if !source_a_available
        println("compute_n2_bounds_zonotope_with_n1_tighten: Source A absent — propagating absolute zonotope only (standard-mode nn1 boost path)")
    end

    # Seed the absolute-N2 input zonotope from [0-ε, 1+ε]. We conservatively
    # cover both the clean-input (network_version="org") and perturbed-input
    # (network_version="perturbation") passes with a single set of bounds;
    # the consumer in core_ops.jl applies them to both.
    #
    # Input range: center = 0.5, radius = 0.5 + ε (per input pixel).
    I_pert_up_flat   = vec(Float64.(I_pert_up_init))
    I_pert_down_flat = vec(Float64.(I_pert_down_init))
    # Per-pixel half-width of the perturbation box (non-negative)
    pert_radius = max.(abs.(I_pert_up_flat), abs.(I_pert_down_flat))

    input_is_4d = ndims(I_pert_up_init) == 4
    # Centre/radius of the input domain. For [0,1] this is the historical
    # 0.5 / 0.5; for the ACAS box it is per-coordinate and asymmetric, so a
    # uniform fill would not cover the region.
    dom_lo_flat, dom_hi_flat = input_domain_flat(length(I_pert_up_flat))
    dom_centre_flat = (dom_lo_flat .+ dom_hi_flat) ./ 2
    dom_halfwidth_flat = (dom_hi_flat .- dom_lo_flat) ./ 2
    if input_is_4d
        in_shape_4d = size(I_pert_up_init)
        n_flat = prod(in_shape_4d)
        center_4d = reshape(dom_centre_flat, in_shape_4d)
        radius_4d = reshape(dom_halfwidth_flat .+ pert_radius, in_shape_4d)
    else
        n_flat = length(I_pert_up_flat)
        center_4d = nothing
        in_shape_4d = (0, 0, 0, 0)
    end

    # Running state
    zono_active = false               # zonotope representation is currently active
    center = Float64[]                # flat center vector when zono_active
    gens   = Matrix{Float64}(undef, 0, 0)   # (n_flat × n_gens)

    # Interval state for pre-zonotope conv/flatten propagation (matches the
    # convention used by compute_diff_bounds_zonotope)
    abs_up   = input_is_4d ? (reshape(dom_hi_flat, in_shape_4d) .+ reshape(pert_radius, in_shape_4d)) : (dom_hi_flat .+ pert_radius)
    abs_down = input_is_4d ? (reshape(dom_lo_flat, in_shape_4d) .- reshape(pert_radius, in_shape_4d)) : (dom_lo_flat .- pert_radius)

    relu_layer_idx = 0
    for (layer_idx, l2) in enumerate(nn2.layers)
        layer_type = string(typeof(l2))

        if occursin("Flatten", layer_type)
            if ndims(abs_up) > 1
                abs_up   = vec(abs_up   |> l2)
                abs_down = vec(abs_down |> l2)
            end
        elseif occursin("Linear", layer_type)
            W2 = Float64.(transpose(l2.matrix))
            b2 = Float64.(l2.bias)

            if !zono_active
                # Activate zonotope: convert current interval state to a zono.
                n = length(abs_up)
                c0 = (abs_up .+ abs_down) ./ 2
                r0 = (abs_up .- abs_down) ./ 2
                center = c0
                gens   = diagm(r0)
                zono_active = true
            end
            center = W2 * center .+ b2
            gens   = W2 * gens
            abs_sum = vec(sum(abs.(gens), dims=2))
            abs_up   = center .+ abs_sum
            abs_down = center .- abs_sum

        elseif occursin("Conv", layer_type)
            F2 = Float64.(l2.filter)
            b2 = Float64.(l2.bias)

                # Interval conv — keep interval state only, zonotope stays off
                in4_up   = ndims(abs_up)   == 4 ? abs_up   : reshape(abs_up,   1, :, 1, 1)
                in4_down = ndims(abs_down) == 4 ? abs_down : reshape(abs_down, 1, :, 1, 1)
                zero_bias = zeros(Float64, length(b2))
                rmin, rmax = interval_conv2d_bounds(F2, F2, in4_down, in4_up, zero_bias, l2.stride, l2.padding)
                abs_down = rmin
                abs_up   = rmax
                for oc in 1:length(b2)
                    abs_down[:, :, :, oc] .+= b2[oc]
                    abs_up[:, :, :, oc]   .+= b2[oc]
                end

        elseif occursin("ReLU", layer_type)
            relu_layer_idx += 1

            # 1. Get current pre-activation scalar hull (flat)
            if zono_active
                abs_sum = vec(sum(abs.(gens), dims=2))
                u_hull  = center .+ abs_sum
                l_hull  = center .- abs_sum
            else
                u_hull = vec(Float64.(abs_up))
                l_hull = vec(Float64.(abs_down))
            end

            # 2. Intersect with the Source-A-derived bound at this ReLU layer:
            #    [n1_preact + diff_down, n1_preact + diff_up]
            if relu_layer_idx <= length(n1_preact_up_bounds)
                n1_up   = vec(Float64.(n1_preact_up_bounds[relu_layer_idx]))
                n1_dn   = vec(Float64.(n1_preact_down_bounds[relu_layer_idx]))
                diff_up = vec(Float64.(relu_diff_up_bounds[relu_layer_idx]))
                diff_dn = vec(Float64.(relu_diff_down_bounds[relu_layer_idx]))
                u_fromA = n1_up .+ diff_up
                l_fromA = n1_dn .+ diff_dn
                # Shape-tolerant intersection: only when lengths match.
                if length(u_fromA) == length(u_hull)
                    u_tight = min.(u_hull, u_fromA)
                    l_tight = max.(l_hull, l_fromA)
                else
                    u_tight = u_hull
                    l_tight = l_hull
                end
            else
                u_tight = u_hull
                l_tight = l_hull
            end

            # 3. Store per-ReLU-layer N2 absolute bounds (flat vectors)
            push!(n2_abs_up_bounds,   copy(u_tight))
            push!(n2_abs_down_bounds, copy(l_tight))

            # 4. Update the running state through the ReLU, using the tightened
            #    [l_tight, u_tight] as the per-neuron box the DeepZ relaxation
            #    acts on.
            if zono_active
                n = length(center)
                new_cols = Vector{Float64}[]
                for i in 1:n
                    u_i = u_tight[i]
                    l_i = l_tight[i]
                    if u_i <= 0.0
                        # Stable inactive: clip to zero
                        center[i] = 0.0
                        gens[i, :] .= 0.0
                    elseif l_i >= 0.0
                        # Stable active: pass through (no change)
                    else
                        # Split: DeepZ relaxation  relu(x) ≈ λx + μ ± μ·ε_new
                        λ = u_i / (u_i - l_i)
                        μ = -u_i * l_i / (2.0 * (u_i - l_i))
                        center[i] = λ * center[i] + μ
                        gens[i, :] .*= λ
                        col = zeros(n)
                        col[i] = μ
                        push!(new_cols, col)
                    end
                end
                if !isempty(new_cols)
                    gens = hcat(gens, hcat(new_cols...))
                end
                # Refresh interval state for the next layer
                abs_sum = vec(sum(abs.(gens), dims=2))
                abs_up   = center .+ abs_sum
                abs_down = center .- abs_sum
            else
                # Interval mode: clip to [0, max(0, u)]
                abs_up   = max.(0.0, abs_up)
                abs_down = .- max.(0.0, .- abs_down)
            end
        end
    end

    println("compute_n2_bounds_zonotope_with_n1_tighten: populated $(length(n2_abs_up_bounds)) ReLU layers")
end

# Conditional Triangle inputs: per-neuron perturbation-difference intervals + per-copy pre-activation bounds.
function compute_n2_pert_relaxation_bounds(nn2, I_pert_up_init, I_pert_down_init)
    global relu_n2pert_up_bounds, relu_n2pert_down_bounds
    global n2_preact_up_bounds, n2_preact_down_bounds

    relu_n2pert_up_bounds   = Array{Float64}[]
    relu_n2pert_down_bounds = Array{Float64}[]
    n2_preact_up_bounds     = Array{Float64}[]
    n2_preact_down_bounds   = Array{Float64}[]

    # Running perturbation interval: z_n2_pert - z_n2_org
    pert_up   = copy(Float64.(I_pert_up_init))
    pert_down = copy(Float64.(I_pert_down_init))

    # N2 post-activation bounds, initialised to the input domain
    n2_act_down, n2_act_up = input_domain_shaped(size(I_pert_up_init))

    n2_pre_up_cur   = Float64[]
    n2_pre_down_cur = Float64[]

    for (layer_idx, l) in enumerate(nn2.layers)
        if occursin("Flatten", string(typeof(l)))
            if ndims(pert_up) > 1
                pert_up     = vec(pert_up   |> l)
                pert_down   = vec(pert_down |> l)
                n2_act_up   = vec(n2_act_up   |> l)
                n2_act_down = vec(n2_act_down |> l)
            end

        elseif occursin("Linear", string(typeof(l)))
            W2 = Float64.(transpose(l.matrix))
            b2 = Float64.(l.bias)

            # pert: W2 * pert_prev  (bias cancels in the difference)
            rp_min, rp_max = interval_matrix_vector_multiplication(W2, W2, pert_down, pert_up)
            pert_down = rp_min
            pert_up   = rp_max

            # N2 pre-activation bounds: W2 * a_n2 + b2
            rn_min, rn_max = interval_matrix_vector_multiplication(W2, W2, n2_act_down, n2_act_up)
            n2_pre_up_cur   = rn_max .+ b2
            n2_pre_down_cur = rn_min .+ b2

        elseif occursin("Conv", string(typeof(l)))
            F2 = Float64.(l.filter)
            b2 = Float64.(l.bias)
            zero_bias = zeros(Float64, length(b2))

            pert_4d_down = ndims(pert_down) == 4 ? pert_down : reshape(pert_down, 1, :, 1, 1)
            pert_4d_up   = ndims(pert_up)   == 4 ? pert_up   : reshape(pert_up,   1, :, 1, 1)
            n2_4d_down   = ndims(n2_act_down) == 4 ? n2_act_down : reshape(n2_act_down, 1, :, 1, 1)
            n2_4d_up     = ndims(n2_act_up)   == 4 ? n2_act_up   : reshape(n2_act_up,   1, :, 1, 1)

            # pert = F2 * pert_prev
            rp_min, rp_max = interval_conv2d_bounds(F2, F2, pert_4d_down, pert_4d_up, zero_bias, l.stride, l.padding)
            pert_down = rp_min
            pert_up   = rp_max

            # N2 preact bounds
            rn_min, rn_max = interval_conv2d_bounds(F2, F2, n2_4d_down, n2_4d_up, Float64.(b2), l.stride, l.padding)
            n2_pre_up_cur   = rn_max
            n2_pre_down_cur = rn_min

        elseif occursin("ReLU", string(typeof(l)))
            # Save pre-ReLU bounds for relaxation decisions in core_ops.jl
            push!(relu_n2pert_up_bounds,   copy(pert_up))
            push!(relu_n2pert_down_bounds, copy(pert_down))
            push!(n2_preact_up_bounds,     copy(n2_pre_up_cur))
            push!(n2_preact_down_bounds,   copy(n2_pre_down_cur))

            # Clip perturbation intervals through ReLU
            if optimizing_intervals
                # Tighter per-neuron clipping: if both N2(x) and N2(x') neurons
                # are in the same activation state, we can preserve or zero the pert interval.
                for i in eachindex(pert_up)
                    l_n2     = n2_pre_down_cur[i]             # N2(x) preact lower
                    u_n2     = n2_pre_up_cur[i]               # N2(x) preact upper
                    l_n2_p   = l_n2 + pert_down[i]            # N2(x') preact lower
                    u_n2_p   = u_n2 + pert_up[i]              # N2(x') preact upper

                    if l_n2 >= 0 && l_n2_p >= 0
                        # both active: post-ReLU pert = pre-ReLU pert, keep as-is
                    elseif u_n2 <= 0 && u_n2_p <= 0
                        # both inactive: post-ReLU pert = 0
                        pert_up[i]   = 0.0
                        pert_down[i] = 0.0
                    else
                        # mixed: conservative non-expansive clipping
                        pert_up[i]   = max(0.0, pert_up[i])
                        pert_down[i] = -max(0.0, -pert_down[i])
                    end
                end
            else
                # Original conservative clipping (non-expansive)
                pert_up   = max.(0.0, pert_up)
                pert_down = .- max.(0.0, .- pert_down)
            end

            # N2 post-activation bounds
            n2_act_up   = max.(0.0, n2_pre_up_cur)
            n2_act_down = max.(0.0, n2_pre_down_cur)
        end
    end

    println("compute_n2_pert_relaxation_bounds: populated $(length(relu_n2pert_up_bounds)) ReLU layers")
end

