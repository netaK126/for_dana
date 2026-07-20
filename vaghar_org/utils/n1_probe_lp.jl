# ──────────────────────────────────────────────────────────────────────
# advstd — N1-probe LP for N2 binary elimination (Source C)
# ──────────────────────────────────────────────────────────────────────
#
# compute_n2_bounds_n1_probe_lp builds a fresh LP-only JuMP model that
# contains an LP-relaxed forward pass through both N1 and N2, with
# shared input variables and perturbation-specific constraints. Each ReLU is
# replaced by the three-inequality triangle LP relaxation parameterised
# by that neuron's pre-activation interval. Then it runs per-neuron
# OBBT on every N2 ReLU pre-activation: min and max against the joint
# LP. The scalar results go into n2_probe_up_bounds_{org,pert} /
# n2_probe_down_bounds_{org,pert} and are consumed by core_ops.jl's
# relu() bound-tightening block, where they flow into the stable-flip
# short-circuit and eliminate additional N2 binaries.
#
# Soundness: the joint LP feasible set is an over-approximation of the
# true (input, N1 state, N2 state) feasible set (triangle LP relaxation
# of ReLU is the tightest 3-inequality LP over-approximation of
# max(0, x) on any [l, u] box). min/max over that set is therefore a
# sound bound on the true reachable range of any N2 pre-activation.
# Shrinking N2's ReLU [l, u] by these scalar bounds preserves
# feasibility of every concrete N2 point, so the integer optimum of
# N2's main MIP is unchanged — only more binaries drop via stable-flip.
#
# Preconditions (caller's responsibility):
#   - n1_preact_up_bounds / n1_preact_down_bounds populated for every
#     ReLU layer. Set by Technique 4's compute_diff_and_comp_bounds or
#     compute_diff_bounds_zonotope upstream in the Phase-2 pre-compute.
#   - relu_diff_up_bounds / relu_diff_down_bounds populated, same.
#     These are used to derive N2's triangle-relaxation bounds via
#     `n2_preact ≈ n1_preact + diff_interval`, which is the Source A
#     bound the existing Technique 4 consumer already uses.
#
# Supported perturbations: linf, brightness, contrast, patch, occ,
# translation. Unsupported types print a warning and return without
# populating the bounds — the consumer check `!isempty(...)` will then
# no-op and the feature degrades to no-probe behaviour.
#
using JuMP
using Gurobi
using LinearAlgebra

"""
    _setup_probe_inputs(m, perturbation, perturbation_size, input_shape, input_range, w, h, k)

Create shared v_in and v_x0 variables on the probe LP model `m`, with
perturbation-specific constraints linking them. Returns (v_in, v_x0).
"""
function _setup_probe_inputs(m, perturbation::String, perturbation_size::Vector{Float64},
                              input_shape, input_range, w::Int, h::Int, k::Int)
    # The probe LP does OBBT and its bounds are used as HARD per-neuron bounds,
    # so solving over the wrong polytope yields bounds tighter than legal.
    _use_box = internet_nets_benchmarks && input_box_lo !== nothing
    _lo(i) = _use_box ? input_box_lo[i] : 0.0
    _hi(i) = _use_box ? input_box_hi[i] : 1.0
    v_in = map(i -> @variable(m, lower_bound=_lo(i), upper_bound=_hi(i)), input_range)

    if perturbation == "linf"
        p_size = perturbation_size[1]
        v_e  = map(_ -> @variable(m, lower_bound=-p_size, upper_bound=p_size), input_range)
        v_x0 = map(i -> @variable(m, lower_bound=_lo(i), upper_bound=_hi(i)), input_range)
        @constraint(m, v_x0 .== v_in .+ v_e)

    elseif perturbation == "brightness"
        p_size = perturbation_size[1]
        v_e = @variable(m, lower_bound=0.0, upper_bound=p_size)
        v_x0 = map(_ -> @variable(m, lower_bound=0.0, upper_bound=1.0+p_size), input_range)
        @constraint(m, v_x0 .== v_in .+ v_e)

    elseif perturbation == "contrast"
        p_size = perturbation_size[1]
        # contrast: v_x0 = e * v_in, e ∈ [1, 1+p_size]
        # LP can't encode bilinear e*v_in directly, so we over-approximate:
        # v_x0[i] ∈ [0, (1+p_size)] and v_x0[i] ≥ v_in[i] (since e ≥ 1)
        # and v_x0[i] ≤ (1+p_size) * v_in[i] (since e ≤ 1+p_size, v_in ≥ 0)
        v_x0 = map(_ -> @variable(m, lower_bound=0.0, upper_bound=1.0+p_size), input_range)
        @constraint(m, v_x0 .>= v_in)
        @constraint(m, v_x0 .<= (1.0 + p_size) .* v_in)

    elseif perturbation == "patch"
        eps = perturbation_size[1]
        ind1 = Int(perturbation_size[2])
        ind2 = Int(perturbation_size[3])
        patch_w = Int(perturbation_size[4])
        v_x0 = map(_ -> @variable(m, lower_bound=0.0, upper_bound=1.0), input_range)
        res_ = w * h
        ind = ind1 + (ind2 - 1) * w
        patch_indices = Int[]
        for i_ in 0:patch_w-1
            for j_ in 0:patch_w-1
                push!(patch_indices, ind + j_ + w * i_)
                if k == 3
                    push!(patch_indices, res_ + ind + j_ + w * i_)
                    push!(patch_indices, 2 * res_ + ind + j_ + w * i_)
                end
            end
        end
        non_patch = setdiff(1:w*h*k, patch_indices)
        @constraint(m, [i=patch_indices], v_x0[i] <= v_in[i] + eps)
        @constraint(m, [i=patch_indices], v_x0[i] >= v_in[i] - eps)
        @constraint(m, [i=non_patch], v_x0[i] == v_in[i])

    elseif perturbation == "occ"
        ind1 = Int(perturbation_size[1])
        ind2 = Int(perturbation_size[2])
        occ_steps = Int(perturbation_size[3])
        v_x0 = map(_ -> @variable(m, lower_bound=0.0, upper_bound=1.0), input_range)
        res_ = w * h
        ind = ind1 + (ind2 - 1) * w
        occ_indices = Int[]
        for i_ in 0:occ_steps-1
            for j_ in 0:occ_steps-1
                push!(occ_indices, ind + j_ + w * i_)
                if k == 3
                    push!(occ_indices, res_ + ind + j_ + w * i_)
                    push!(occ_indices, 2 * res_ + ind + j_ + w * i_)
                end
            end
        end
        non_occ = setdiff(1:w*h*k, occ_indices)
        @constraint(m, [i=occ_indices], v_x0[i] == 0.0)
        @constraint(m, [i=non_occ], v_x0[i] == v_in[i])

    elseif perturbation == "translation"
        t_down = Int(perturbation_size[1])
        t_right = Int(perturbation_size[2])
        v_x0 = map(_ -> @variable(m, lower_bound=0.0, upper_bound=1.0), input_range)
        res_ = w * h
        # Interior pixels: shifted
        for i2 in 1:w-t_right
            for i1 in 1:h-t_down
                i = i1 + h * (i2 - 1)
                @constraint(m, v_x0[i + t_down + w * t_right] == v_in[i])
                if k == 3
                    @constraint(m, v_x0[res_ + i + t_down + w * t_right] == v_in[res_ + i])
                    @constraint(m, v_x0[2*res_ + i + t_down + w * t_right] == v_in[2*res_ + i])
                end
            end
        end
        # Border pixels: zeroed
        for j in 1:t_down
            @constraint(m, [i=j:w:res_], v_x0[i] == 0.0)
            if k == 3
                @constraint(m, [i=j+res_:w:2*res_], v_x0[i] == 0.0)
                @constraint(m, [i=j+2*res_:w:3*res_], v_x0[i] == 0.0)
            end
        end
        for j in 1:t_right
            @constraint(m, [i=1+w*(j-1):1:w*j], v_x0[i] == 0.0)
            if k == 3
                @constraint(m, [i=res_+1+w*(j-1):1:res_+w*j], v_x0[i] == 0.0)
                @constraint(m, [i=2*res_+1+w*(j-1):1:2*res_+w*j], v_x0[i] == 0.0)
            end
        end

    elseif perturbation == "rotation"
        # Mirrors get_perturbation_specific_keys_rotate in perturbation_models.jl.
        # Rotation is LP-representable: trig is evaluated at setup time, so each
        # rotated pixel is a linear combination of four source pixels with fixed
        # bilinear-interpolation weights. No variable × variable products.
        angle = perturbation_size[1]
        v_x0 = map(_ -> @variable(m, lower_bound=0.0, upper_bound=1.0), input_range)
        res_ = w * h
        center = [w/2, h/2]
        mapped_indices = Int[]
        for i in 1:h
            for j in 1:w
                j_c = j - center[1]
                i_c = i - center[2]
                j_r = j_c*cos(angle*pi/180) - i_c*sin(angle*pi/180) + center[1]
                i_r = j_c*sin(angle*pi/180) + i_c*cos(angle*pi/180) + center[2]
                if floor(Int,j_r) >= 1 && ceil(Int,j_r) <= w &&
                   floor(Int,i_r) >= 1 && ceil(Int,i_r) <= h
                    di = i_r - floor(i_r)
                    dj = j_r - floor(j_r)
                    fi = floor(Int, i_r); ci = ceil(Int, i_r)
                    fj = floor(Int, j_r); cj = ceil(Int, j_r)
                    dst = i + (j - 1) * h
                    @constraint(m, v_x0[dst] ==
                        (1-di)*(1-dj) * v_in[fi + (fj - 1) * h] +
                        (di)*(1-dj)   * v_in[ci + (fj - 1) * h] +
                        (1-di)*(dj)   * v_in[fi + (cj - 1) * h] +
                        (di)*(dj)     * v_in[ci + (cj - 1) * h])
                    push!(mapped_indices, dst)
                    if k == 3
                        for chan in 1:2
                            dst_c = dst + chan * res_
                            @constraint(m, v_x0[dst_c] ==
                                (1-di)*(1-dj) * v_in[fi + (fj - 1) * h + chan * res_] +
                                (di)*(1-dj)   * v_in[ci + (fj - 1) * h + chan * res_] +
                                (1-di)*(dj)   * v_in[fi + (cj - 1) * h + chan * res_] +
                                (di)*(dj)     * v_in[ci + (cj - 1) * h + chan * res_])
                            push!(mapped_indices, dst_c)
                        end
                    end
                end
            end
        end
        # Unmapped (border) pixels: zero-pad, matching the main MIP encoding.
        for tt in 1:(res_ * k)
            if !(tt in mapped_indices)
                @constraint(m, v_x0[tt] == 0.0)
            end
        end

    else
        error("_setup_probe_inputs: unsupported perturbation '$perturbation'")
    end

    return v_in, v_x0
end

function compute_n2_bounds_n1_probe_lp(
        nn1, nn2,
        perturbation::String,
        perturbation_size::Vector{Float64},
        w::Int, h::Int, k_input::Int,
        optimizer)

    global n2_probe_up_bounds_org, n2_probe_down_bounds_org
    global n2_probe_up_bounds_pert, n2_probe_down_bounds_pert
    global n1_preact_up_bounds, n1_preact_down_bounds
    global relu_diff_up_bounds, relu_diff_down_bounds

    n2_probe_up_bounds_org   = Array{Float64}[]
    n2_probe_down_bounds_org = Array{Float64}[]
    n2_probe_up_bounds_pert  = Array{Float64}[]
    n2_probe_down_bounds_pert = Array{Float64}[]

    # Precondition checks. Return false on any skip so the caller can
    # abort the Phase 2 run rather than silently proceeding without the
    # probe's tightening (which used to produce misleadingly-tagged
    # results; see Bug~1 in advstd_techniques_code_guide.tex).
    if isempty(n1_preact_up_bounds) || isempty(relu_diff_up_bounds)
        println("compute_n2_bounds_n1_probe_lp: n1_preact or relu_diff bounds not populated — Source C disabled")
        return false
    end
    if length(n1_preact_up_bounds) != length(relu_diff_up_bounds)
        println("compute_n2_bounds_n1_probe_lp: layer count mismatch (n1_preact=$(length(n1_preact_up_bounds)), relu_diff=$(length(relu_diff_up_bounds))) — Source C disabled")
        return false
    end

    supported_perturbations = ("linf", "brightness", "contrast", "patch", "occ", "translation", "rotation")
    if !(perturbation in supported_perturbations)
        println("compute_n2_bounds_n1_probe_lp: perturbation '$perturbation' not supported; skipping Source C probe (supported: $supported_perturbations)")
        return false
    end

    n_relu_layers = length(n1_preact_up_bounds)
    n2_layer_count = count(l -> occursin("ReLU", string(typeof(l))), nn2.layers)
    if n2_layer_count != n_relu_layers
        println("compute_n2_bounds_n1_probe_lp: n2 ReLU layer count ($n2_layer_count) != n1_preact layer count ($n_relu_layers) — Source C disabled")
        return false
    end

    # ── Build the probe LP model ────────────────────────────────────
    m_probe = Model(optimizer)
    set_silent(m_probe)
    try
        set_optimizer_attribute(m_probe, "OutputFlag", 0)
    catch
    end

    # Shared input variables — dispatch by perturbation type
    input_shape = (1, w, h, k_input)
    input_range = CartesianIndices(input_shape)
    v_in, v_x0 = _setup_probe_inputs(m_probe, perturbation, perturbation_size,
                                      input_shape, input_range, w, h, k_input)

    # Derive N2 triangle bounds via Source A's formulation:
    #   n2_preact ∈ [n1_preact + diff_down, n1_preact + diff_up]
    # This is sound and tighter than N2-alone interval propagation.
    n2_preact_up = Vector{Vector{Float64}}(undef, n_relu_layers)
    n2_preact_dn = Vector{Vector{Float64}}(undef, n_relu_layers)
    for r in 1:n_relu_layers
        n2_preact_up[r] = vec(Float64.(n1_preact_up_bounds[r])) .+ vec(Float64.(relu_diff_up_bounds[r]))
        n2_preact_dn[r] = vec(Float64.(n1_preact_down_bounds[r])) .+ vec(Float64.(relu_diff_down_bounds[r]))
    end

    # ── Forward pass helpers ────────────────────────────────────────
    # Walk a neural net, running each layer manually so we can replace
    # ReLU with triangle LP relaxation. Returns (final output vars,
    # per-ReLU-layer pre-activation handles).
    function forward_pass_triangle_relaxed(m, nn, input_vars::Array,
                                            preact_up::Vector{Vector{Float64}},
                                            preact_dn::Vector{Vector{Float64}})
        state = input_vars
        relu_idx = 0
        preact_handles = Vector{Vector{JuMP.AffExpr}}()
        for layer in nn.layers
            layer_type = string(typeof(layer))
            if occursin("ReLU", layer_type)
                relu_idx += 1
                if relu_idx > length(preact_up)
                    error("forward_pass_triangle_relaxed: ReLU layer $relu_idx exceeds preact bounds length $(length(preact_up))")
                end
                # state is currently the pre-activation expressions for
                # this ReLU. Snapshot handles for OBBT.
                flat_state = vec(state)
                u_r = preact_up[relu_idx]
                l_r = preact_dn[relu_idx]
                n_neurons = length(flat_state)
                if length(u_r) != n_neurons
                    error("forward_pass_triangle_relaxed: bound length mismatch at ReLU $relu_idx — bounds=$(length(u_r)), neurons=$n_neurons")
                end

                # Store pre-activation handles as AffExpr for OBBT objective.
                pre_handles = [copy(flat_state[i]) for i in 1:n_neurons]
                push!(preact_handles, pre_handles)

                # Build post-activation variables via triangle LP relaxation.
                post_flat = Vector{Union{VariableRef, AffExpr}}(undef, n_neurons)
                for i in 1:n_neurons
                    u_i = u_r[i]
                    l_i = l_r[i]
                    if u_i <= 0.0
                        post_flat[i] = zero(JuMP.AffExpr)
                    elseif l_i >= 0.0
                        post_flat[i] = flat_state[i]
                    else
                        v = @variable(m, lower_bound=0.0, upper_bound=max(0.0, u_i))
                        @constraint(m, v >= flat_state[i])
                        @constraint(m, v >= 0.0)
                        # Upper triangle: v <= (u / (u - l)) * (pre - l)
                        @constraint(m, v <= (u_i / (u_i - l_i)) * (flat_state[i] - l_i))
                        post_flat[i] = v
                    end
                end
                # Reshape post back to the input shape of the current
                # state (ReLU preserves shape).
                state = reshape(post_flat, size(state))
            else
                # Non-ReLU layer: use the existing pipe dispatch. Works
                # for Linear (1D), Conv2d (4D), Flatten (4D→1D), etc.
                state = state |> layer
            end
        end
        return state, preact_handles
    end

    # ── Run forward passes for both networks and both input seeds ───
    _, n1_org_handles = forward_pass_triangle_relaxed(m_probe, nn1, v_in,
        [vec(Float64.(n1_preact_up_bounds[r])) for r in 1:n_relu_layers],
        [vec(Float64.(n1_preact_down_bounds[r])) for r in 1:n_relu_layers])

    _, n1_pert_handles = forward_pass_triangle_relaxed(m_probe, nn1, v_x0,
        [vec(Float64.(n1_preact_up_bounds[r])) for r in 1:n_relu_layers],
        [vec(Float64.(n1_preact_down_bounds[r])) for r in 1:n_relu_layers])

    _, n2_org_handles = forward_pass_triangle_relaxed(m_probe, nn2, v_in,
        n2_preact_up, n2_preact_dn)

    _, n2_pert_handles = forward_pass_triangle_relaxed(m_probe, nn2, v_x0,
        n2_preact_up, n2_preact_dn)

    # ── Per-neuron OBBT ─────────────────────────────────────────────
    n_obbt_probes = 0
    obbt_optimize_time = 0.0
    for r in 1:length(n2_org_handles)
        n_neurons = length(n2_org_handles[r])
        push!(n2_probe_up_bounds_org, fill(Inf, n_neurons))
        push!(n2_probe_down_bounds_org, fill(-Inf, n_neurons))
        for k in 1:n_neurons
            @objective(m_probe, Min, n2_org_handles[r][k])
            obbt_optimize_time += @elapsed optimize!(m_probe)
            if termination_status(m_probe) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
                n2_probe_down_bounds_org[r][k] = objective_value(m_probe)
            end
            @objective(m_probe, Max, n2_org_handles[r][k])
            obbt_optimize_time += @elapsed optimize!(m_probe)
            if termination_status(m_probe) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
                n2_probe_up_bounds_org[r][k] = objective_value(m_probe)
            end
            n_obbt_probes += 2
        end
    end

    for r in 1:length(n2_pert_handles)
        n_neurons = length(n2_pert_handles[r])
        push!(n2_probe_up_bounds_pert, fill(Inf, n_neurons))
        push!(n2_probe_down_bounds_pert, fill(-Inf, n_neurons))
        for k in 1:n_neurons
            @objective(m_probe, Min, n2_pert_handles[r][k])
            obbt_optimize_time += @elapsed optimize!(m_probe)
            if termination_status(m_probe) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
                n2_probe_down_bounds_pert[r][k] = objective_value(m_probe)
            end
            @objective(m_probe, Max, n2_pert_handles[r][k])
            obbt_optimize_time += @elapsed optimize!(m_probe)
            if termination_status(m_probe) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
                n2_probe_up_bounds_pert[r][k] = objective_value(m_probe)
            end
            n_obbt_probes += 2
        end
    end

    global n1_probe_lp_time = obbt_optimize_time

    # Tear down the probe model
    m_probe = nothing
    GC.gc()

    # ── Count how many N2 binaries the probe eliminates compared to the
    # "everything except the probe" baseline. For each neuron, compute
    # the pre-probe [l_base, u_base] as the intersection of Source A
    # (n1_preact + relu_diff) and Source B (n2_abs, if populated). If
    # the baseline was split (u_base > 0 && l_base < 0) but the probe's
    # bound makes the intersection single-signed, the probe eliminated
    # that binary. This count goes into the result filename so we can
    # measure the flag's effect directly from the file listing. It is
    # an approximation: the actual big-M box in relu() is tightened
    # further by MIPVerify's own LP OBBT inside get_model, so the true
    # elimination count can be higher or lower. Close enough for a
    # filename-level metric.
    global n_probe_eliminated_binaries_org, n_probe_eliminated_binaries_pert, n_probe_eliminated_binaries
    n_probe_eliminated_binaries_org  = 0
    n_probe_eliminated_binaries_pert = 0
    n_probe_eliminated_binaries      = 0
    function _count_eliminated(probe_up::Vector{<:AbstractArray{Float64}},
                               probe_dn::Vector{<:AbstractArray{Float64}},
                               base_u::Vector{<:AbstractArray{Float64}},
                               base_d::Vector{<:AbstractArray{Float64}})
        cnt = 0
        for r in 1:length(probe_up)
            if r > length(base_u); break; end
            n_neurons = length(probe_up[r])
            for k in 1:n_neurons
                if k > length(base_u[r]); continue; end
                l_base = base_d[r][k]
                u_base = base_u[r][k]
                was_split = (u_base > 0.0 && l_base < 0.0)
                if !was_split; continue; end
                l_full = max(l_base, probe_dn[r][k])
                u_full = min(u_base, probe_up[r][k])
                if u_full <= 0.0 || l_full >= 0.0
                    cnt += 1
                end
            end
        end
        return cnt
    end

    # Baseline = Source A alone (Source B, if active, further tightens
    # it; we include Source B when available).
    base_up_org = [n2_preact_up[r] for r in 1:n_relu_layers]
    base_dn_org = [n2_preact_dn[r] for r in 1:n_relu_layers]
    base_up_pert = [n2_preact_up[r] for r in 1:n_relu_layers]
    base_dn_pert = [n2_preact_dn[r] for r in 1:n_relu_layers]
    if !isempty(n2_abs_up_bounds)
        for r in 1:min(n_relu_layers, length(n2_abs_up_bounds))
            for k in 1:min(length(base_up_org[r]), length(n2_abs_up_bounds[r]))
                base_up_org[r][k]  = min(base_up_org[r][k],  n2_abs_up_bounds[r][k])
                base_dn_org[r][k]  = max(base_dn_org[r][k],  n2_abs_down_bounds[r][k])
                base_up_pert[r][k] = min(base_up_pert[r][k], n2_abs_up_bounds[r][k])
                base_dn_pert[r][k] = max(base_dn_pert[r][k], n2_abs_down_bounds[r][k])
            end
        end
    end
    n_probe_eliminated_binaries_org = _count_eliminated(
        n2_probe_up_bounds_org, n2_probe_down_bounds_org, base_up_org, base_dn_org)
    n_probe_eliminated_binaries_pert = _count_eliminated(
        n2_probe_up_bounds_pert, n2_probe_down_bounds_pert, base_up_pert, base_dn_pert)
    n_probe_eliminated_binaries = n_probe_eliminated_binaries_org + n_probe_eliminated_binaries_pert

    println("compute_n2_bounds_n1_probe_lp: populated $(length(n2_probe_up_bounds_org)) ReLU layers (org) + $(length(n2_probe_up_bounds_pert)) (pert) via $n_obbt_probes OBBT probes; probe eliminated N2(x)=$n_probe_eliminated_binaries_org + N2(x')=$n_probe_eliminated_binaries_pert = $n_probe_eliminated_binaries N2 binaries")
    return true
end
