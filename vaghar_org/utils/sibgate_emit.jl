# ────────────────────────────────────────────────────────────────────────────
# advstd Technique 4 — SibGate (sibling-gated conditional triangle)
#
# apply_sibgate_constraints!(model) runs once per N2 build, AFTER both copies
# (org and pert) have been encoded by get_model. It walks n2_relax_decision
# and for each relaxed neuron emits the SibGate constraints on top of the
# simple Wong-Kolter triangle that core_ops.jl::relu() already added.
#
# Tier classification (read from n2_relax_decision; both keys store the same
# (relax_org, relax_pert) pair after the refactor in n2_relax_decision.jl):
#   • both thin  → relax_org=true, relax_pert=true
#   • one thin (org dropped, pert exact)  → relax_org=true, relax_pert=false
#   • one thin (pert dropped, org exact)  → relax_org=false, relax_pert=true
#
# Emission per tier:
#   both thin:
#     l_int ≤ ẑ^pert − ẑ^org ≤ u_int      (one linear coupling line)
#   one thin (relaxed copy c, sibling pre):
#     z^c ≤ envelope(ẑ^c; lA, uA) + M·(1 − a^pre)    (active branch)
#     z^c ≤ envelope(ẑ^c; lI, uI) + M·a^pre          (inactive branch)
#   where (lA,uA) and (lI,uI) intersect Source-A conditional intervals
#   (built from [l_int, u_int] and the sibling's per-copy preact bound)
#   with the relaxed copy's unconditional per-copy bound (l_c, u_c).
#
# Soundness: every x ∈ F_exact satisfies these constraints by construction
# (see advstd_techniques.tex §Tech 4), so F_exact ⊆ F_SibGate and
# δ_SibGate ≥ δ_exact.
# ────────────────────────────────────────────────────────────────────────────

function apply_sibgate_constraints!(model)
    if !adv_std_n2_sibling_gate || isempty(n2_relax_decision)
        return
    end
    if isempty(relu_n2pert_up_bounds)
        println("apply_sibgate_constraints!: relu_n2pert_*_bounds empty — " *
                "compute_n2_pert_relaxation_bounds wasn't called. Skipping.")
        return
    end

    n_coupling   = 0
    n_cond_org   = 0  # conditional triangle on org (org dropped, pert exact)
    n_cond_pert  = 0  # conditional triangle on pert (pert dropped, org exact)

    K = length(relu_n2pert_up_bounds)
    for m_idx in 1:K
        n_neurons = length(relu_n2pert_up_bounds[m_idx])
        for k_idx in 1:n_neurons
            # Match the dual-key layout that compute_n2_relax_decision! uses:
            # org pass keys are nn_layer = m_idx, pert pass keys are
            # nn_layer = m_idx + K. (See n2_relax_decision.jl.)
            key_org  = (m_idx,     k_idx)
            key_pert = (m_idx + K, k_idx)
            has_org  = haskey(n2_relax_decision, key_org)
            has_pert = haskey(n2_relax_decision, key_pert)

            if !has_org && !has_pert
                continue
            end

            # Pull the cached MIP state for both sides (always populated by
            # core_ops.jl::relu when adv_std_n2_sibling_gate is true).
            st_org  = get(n2_relu_state, (m_idx, k_idx, "org"),          nothing)
            st_pert = get(n2_relu_state, (m_idx, k_idx, "perturbation"), nothing)
            if st_org === nothing || st_pert === nothing
                # One side was stable (single-signed bound) so relu()
                # short-circuited before allocating x_rect. No SibGate
                # action possible — the simple-triangle / exact encoding
                # on the surviving side already handles it.
                continue
            end

            # Perturbation diff bound (org → pert preact diff in N2 alone).
            # This is the [l_int, u_int] that the formula derivation in
            # advstd_techniques.tex §Tech 4 calls for, computed by
            # compute_n2_pert_relaxation_bounds() via interval arithmetic
            # through N2 alone — never zonotope.
            l_int = relu_n2pert_down_bounds[m_idx][k_idx]
            u_int = relu_n2pert_up_bounds[m_idx][k_idx]

            if has_org && has_pert
                # ── BOTH THIN: emit pre-activation coupling line ──────────
                # No binary survives on this neuron, so no conditional
                # triangle is possible. The coupling forces the LP to
                # pick (ẑ^org, ẑ^pert) jointly inside a slab instead of
                # the rectangle the two simple triangles would allow.
                @constraint(model, st_pert.preact - st_org.preact <= u_int)
                @constraint(model, st_pert.preact - st_org.preact >= l_int)
                n_coupling += 1
                continue
            end

            # ── ONE THIN: emit conditional triangle on the relaxed copy ─
            relaxed_is_org = has_org   # only one of has_org/has_pert is true here
            (st_c, st_pre, sibling_prefix) = relaxed_is_org ?
                (st_org,  st_pert, "perturbation") :
                (st_pert, st_org,  "org")

            # Sibling binary lookup (it was created by the exact big-M
            # encoding on the surviving copy, so it must exist by now).
            # Variable-name template (core_ops.jl::relu line 617):
            #   {prefix}a_layerCount{layer_counter}_neuronCount{nueron_counter}_{neurons_names.layer}_{neurons_names.neuron}
            # `nueron_counter` is reset to 0 at every layer and never
            # bumped, so it is always 0 in practice. `neurons_names.layer`
            # is cumulative across passes: the org pass has layer indices
            # m_idx (1..K) and the pert pass has m_idx + K (K+1..2K).
            sibling_layer_idx = sibling_prefix == "org" ? m_idx : m_idx + K
            a_pre_name = string(sibling_prefix, "a_",
                                "layerCount", m_idx, "_",
                                "neuronCount0_",
                                sibling_layer_idx, "_", k_idx)
            a_pre = variable_by_name(model, a_pre_name)
            if a_pre === nothing
                # Naming-convention mismatch (e.g., layer_counter / neuron_counter
                # diverged from the m_idx/k_idx we're iterating). Fall back to
                # name-prefix scan — slower but robust.
                a_pre = _find_sibling_binary(model, sibling_prefix, sibling_layer_idx, k_idx)
                if a_pre === nothing
                    println("apply_sibgate_constraints!: missing sibling binary at " *
                            "(m_idx=$m_idx, k_idx=$k_idx, sibling=$sibling_prefix); " *
                            "falling back to simple triangle (already emitted).")
                    continue
                end
            end

            # Conditional intervals (intersection of Source-A diff bounds
            # with Tech-1 unconditional per-copy bound on the relaxed copy).
            #
            # [l_int, u_int] from compute_n2_pert_relaxation_bounds bounds
            # ẑ^pert − ẑ^org specifically. Convert to ẑ^c − ẑ^pre depending
            # on which copy is relaxed:
            #   relaxed=pert (pre=org):  ẑ^c − ẑ^pre =  ẑ^pert − ẑ^org ∈ [ l_int,  u_int]
            #   relaxed=org  (pre=pert): ẑ^c − ẑ^pre = -(ẑ^pert − ẑ^org) ∈ [-u_int, -l_int]
            # Without this sign-flip the env on the org-relaxed side would
            # cut off feasible exact-MIP points (δ_exact not in F_SibGate).
            L_diff, U_diff = relaxed_is_org ? (-u_int, -l_int) : (l_int, u_int)

            l_c, u_c = st_c.l, st_c.u
            l_pre, u_pre = st_pre.l, st_pre.u
            # Active branch  (ẑ^pre ∈ [0,    u_pre]): ẑ^c ∈ [L_diff,         u_pre + U_diff]
            # Inactive branch(ẑ^pre ∈ [l_pre, 0   ]): ẑ^c ∈ [l_pre + L_diff, U_diff       ]
            lA = max(L_diff,         l_c)
            uA = min(u_pre + U_diff, u_c)
            lI = max(l_pre + L_diff, l_c)
            uI = min(U_diff,         u_c)

            # Big-M chosen large enough that the "off" upper bound is
            # vacuous over the relaxed copy's feasible region.
            M = u_c + abs(l_c)

            # Active branch (a_pre = 1): bound holds with envelope on [lA, uA].
            if uA > lA && lA < 0.0 && uA > 0.0
                @constraint(model,
                    st_c.x_rect <= (uA / (uA - lA)) * (st_c.preact - lA) + M * (1 - a_pre))
            elseif lA >= 0.0
                # Active branch implies ẑ^c ≥ 0 ≥ lA, so envelope = ẑ^c.
                @constraint(model, st_c.x_rect <= st_c.preact + M * (1 - a_pre))
            else
                # uA ≤ 0: active branch infeasible (ReLU output 0); slack alone bounds.
                @constraint(model, st_c.x_rect <= M * (1 - a_pre))
            end

            # Inactive branch (a_pre = 0): bound holds with envelope on [lI, uI].
            if uI > lI && lI < 0.0 && uI > 0.0
                @constraint(model,
                    st_c.x_rect <= (uI / (uI - lI)) * (st_c.preact - lI) + M * a_pre)
            elseif lI >= 0.0
                @constraint(model, st_c.x_rect <= st_c.preact + M * a_pre)
            else
                @constraint(model, st_c.x_rect <= M * a_pre)
            end

            if relaxed_is_org
                n_cond_org += 1
            else
                n_cond_pert += 1
            end
        end
    end

    println("apply_sibgate_constraints!: emitted " *
            "$n_coupling coupling lines (both-thin), " *
            "$n_cond_org conditional triangles on org (org-dropped), " *
            "$n_cond_pert conditional triangles on pert (pert-dropped).")
end

# Fallback: scan all binary variables in the model and find the one whose
# name matches the trailing `_{sibling_layer_idx}_{k_idx}` produced by
# core_ops.jl::relu (line 617). Slow O(|vars|) — used only when the direct
# variable_by_name lookup misses (e.g., layer_counter / neurons_names
# diverged from the m_idx/k_idx we iterate). The `neuronCount` field in
# the variable name is always 0 (nueron_counter is reset per layer and
# never bumped), so we anchor on the trailing layer/neuron pair instead.
function _find_sibling_binary(model, sibling_prefix::AbstractString,
                              sibling_layer_idx::Int, k_idx::Int)
    needle_prefix = sibling_prefix * "a_"
    needle_suffix = string("_", sibling_layer_idx, "_", k_idx)
    for v in JuMP.all_variables(model)
        nm = JuMP.name(v)
        if startswith(nm, needle_prefix) && endswith(nm, needle_suffix)
            return v
        end
    end
    return nothing
end
