# The Conditional Triangle's constraint emission, run once per build after both copies are encoded. When BOTH copies of a
# neuron are relaxed, their triangles alone would let the copies drift apart, so one constraint ties the two pre-activations
# together by the perturbation-difference interval; when only ONE copy is relaxed, its triangle is made conditional on the
# sibling's surviving binary — a tighter envelope for each of the sibling's phases. Every point feasible under the exact
# encoding satisfies these constraints, so delta_relaxed >= delta_exact.
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

            # Fetch the pieces the encoder saved for this neuron in each copy.
            st_org  = get(n2_relu_state, (m_idx, k_idx, "org"),          nothing)
            st_pert = get(n2_relu_state, (m_idx, k_idx, "perturbation"), nothing)
            # If either copy turned out stable, there is no pair to connect — skip.
            if st_org === nothing || st_pert === nothing
                continue
            end

            # This neuron's perturbation-difference interval (perturbed copy vs clean copy), from the Conditional Triangle's input pass.
            l_int = relu_n2pert_down_bounds[m_idx][k_idx]
            u_int = relu_n2pert_up_bounds[m_idx][k_idx]

            if has_org && has_pert
                # Both copies relaxed: no binary survives to condition on, so tie the two pre-activations together by the perturbation-difference interval — without this the two triangles could drift apart independently.
                @constraint(model, st_pert.preact - st_org.preact <= u_int)
                @constraint(model, st_pert.preact - st_org.preact >= l_int)
                n_coupling += 1
                continue
            end

            # ── ONE THIN: emit conditional triangle on the relaxed copy ─
            relaxed_is_org = has_org   # only one of has_org/has_pert is true here
            (st_c, st_sibling, sibling_prefix) = relaxed_is_org ?
                (st_org,  st_pert, "perturbation") :
                (st_pert, st_org,  "org")

            # Look up the sibling's surviving binary by its variable name (clean copy = layers 1..K, perturbed copy = K+1..2K).
            sibling_layer_idx = sibling_prefix == "org" ? m_idx : m_idx + K
            a_sibling_name = string(sibling_prefix, "a_",
                                "layerCount", m_idx, "_",
                                "neuronCount0_",
                                sibling_layer_idx, "_", k_idx)
            a_sibling = variable_by_name(model, a_sibling_name)
            if a_sibling === nothing
                # Direct name lookup missed — scan all variables for the matching suffix instead (slower but robust).
                a_sibling = _find_sibling_binary(model, sibling_prefix, sibling_layer_idx, k_idx)
                if a_sibling === nothing
                    println("apply_sibgate_constraints!: missing sibling binary at " *
                            "(m_idx=$m_idx, k_idx=$k_idx, sibling=$sibling_prefix); " *
                            "falling back to simple triangle (already emitted).")
                    continue
                end
            end

            # The perturbation-difference interval bounds (perturbed - clean), but the conditional envelope needs (relaxed - sibling):
            # when the CLEAN copy is the relaxed one, the interval must be negated — skipping the flip would cut off feasible points of the exact encoding (unsound).
            L_diff, U_diff = relaxed_is_org ? (-u_int, -l_int) : (l_int, u_int)

            l_c, u_c = st_c.l, st_c.u
            l_sib, u_sib = st_sibling.l, st_sibling.u
            # Sibling active   (its pre-activation in [0, u_sib]):   relaxed copy in [L_diff,         u_sib + U_diff]
            # Sibling inactive (its pre-activation in [l_sib, 0]):   relaxed copy in [l_sib + L_diff, U_diff       ]
            lA = max(L_diff,         l_c)
            uA = min(u_sib + U_diff, u_c)
            lI = max(l_sib + L_diff, l_c)
            uI = min(U_diff,         u_c)

            # Big-M chosen large enough that the "off" upper bound is
            # vacuous over the relaxed copy's feasible region.
            M = u_c + abs(l_c)

            # Sibling active (a_sibling = 1): the envelope on [lA, uA] applies.
            if uA > lA && lA < 0.0 && uA > 0.0
                @constraint(model,
                    st_c.x_rect <= (uA / (uA - lA)) * (st_c.preact - lA) + M * (1 - a_sibling))
            elseif lA >= 0.0
                # Active branch implies ẑ^c ≥ 0 ≥ lA, so envelope = ẑ^c.
                @constraint(model, st_c.x_rect <= st_c.preact + M * (1 - a_sibling))
            else
                # uA ≤ 0: active branch infeasible (ReLU output 0); slack alone bounds.
                @constraint(model, st_c.x_rect <= M * (1 - a_sibling))
            end

            # Sibling inactive (a_sibling = 0): the envelope on [lI, uI] applies.
            if uI > lI && lI < 0.0 && uI > 0.0
                @constraint(model,
                    st_c.x_rect <= (uI / (uI - lI)) * (st_c.preact - lI) + M * a_sibling)
            elseif lI >= 0.0
                @constraint(model, st_c.x_rect <= st_c.preact + M * a_sibling)
            else
                @constraint(model, st_c.x_rect <= M * a_sibling)
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

# Fallback for the sibling-binary lookup: scan every binary variable for the right copy prefix and trailing layer/neuron pair (slow, used only when the direct name lookup misses).
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
