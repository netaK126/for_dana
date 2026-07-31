# The Conditional Triangle's decision: for every neuron, score each copy's triangle-gap area on its tightened bounds and
# record which copies to relax — both when both areas are <= tau, only the smaller-area copy when just it is, neither otherwise.
# The scoring uses exactly the bounds the encoder will use when emitting the triangle, so the triangle always contains the
# exact ReLU on the same [l, u] — which is why delta_relaxed >= delta_exact. Fills n2_relax_decision, consulted by relu().
# ────────────────────────────────────────────────────────────────────────────

function compute_n2_relax_decision!(threshold::Real)
    global n2_relax_decision
    clear_n2_relax_decision!()

    if threshold < 0.0
        return
    end

    # Layer count and per-layer neuron counts, taken from whichever bound arrays this mode has: the difference bounds (transfer) or the zonotope bounds (no transfer).
    bounds_for_shape = !isempty(relu_diff_up_bounds) ? relu_diff_up_bounds :
                       (!isempty(n2_abs_up_bounds)   ? n2_abs_up_bounds   : nothing)
    if bounds_for_shape === nothing
        println("compute_n2_relax_decision!: no shape source available " *
                "(relu_diff_up_bounds and n2_abs_up_bounds both empty) — " *
                "skipping decision precompute")
        return
    end
    K = length(bounds_for_shape)
    if K == 0
        println("compute_n2_relax_decision!: shape source has zero ReLU layers — " *
                "skipping decision precompute")
        return
    end

    # Start each neuron from a wide envelope and tighten it exactly as the encoder will; starting looser only makes the tau test more conservative (fewer relaxed neurons), never unsound.
    l_init = -Inf
    u_init = +Inf

    n_both = 0
    n_org_only = 0
    n_pert_only = 0
    n_none = 0
    n_skipped_stable = 0

    for m_idx in 1:K
        n_neurons = length(bounds_for_shape[m_idx])
        for k_idx in 1:n_neurons
            # Use the encoder's layer numbering: the clean copy is layers 1..K, the perturbed copy K+1..2K, matching N_pre's saved bounds so both copies get tightened.
            nn_layer_org  = m_idx
            nn_layer_pert = m_idx + K

            # The clean copy's tightened bounds [l, u]: the envelope intersected with N_pre's shifted bounds and the zonotope bounds, exactly as the encoder will.
            (l_org,  u_org)  = intersect_per_copy_bounds(
                l_init, u_init,
                nn_layer_org, k_idx,
                m_idx, k_idx,
                "org",
            )
            # The perturbed copy's tightened bounds, same intersections.
            (l_pert, u_pert) = intersect_per_copy_bounds(
                l_init, u_init,
                nn_layer_pert, k_idx,
                m_idx, k_idx,
                "perturbation",
            )

            # Stable neurons (single-signed interval) have no binary to relax
            # — relu() already short-circuits them. Skip here too.
            stable_org  = (l_org  >= 0.0 || u_org  <= 0.0)
            stable_pert = (l_pert >= 0.0 || u_pert <= 0.0)
            if stable_org && stable_pert
                n_skipped_stable += 1
                continue
            end

            g_org  = _tri_gap(l_org,  u_org)
            g_pert = _tri_gap(l_pert, u_pert)

            # Tiered decision. Stable copies (gap == 0) are eligible for
            # the "relax" branch trivially, but relu() short-circuits them
            # before the triangle path, so marking them true is harmless.
            if max(g_org, g_pert) <= threshold
                relax_org, relax_pert = true, true
                n_both += 1
            elseif min(g_org, g_pert) <= threshold
                if g_org <= g_pert
                    relax_org, relax_pert = true, false
                    n_org_only += 1
                else
                    relax_org, relax_pert = false, true
                    n_pert_only += 1
                end
            else
                relax_org, relax_pert = false, false
                n_none += 1
            end

            if relax_org || relax_pert
                # Store the full (relax clean?, relax perturbed?) decision under BOTH copies' keys, so whichever copy the encoder is building, it can tell "both relaxed" (couple them) from "one relaxed" (gate on the sibling).
                if relax_org
                    n2_relax_decision[(nn_layer_org, k_idx)]  = (relax_org, relax_pert)
                end
                if relax_pert
                    n2_relax_decision[(nn_layer_pert, k_idx)] = (relax_org, relax_pert)
                end
            end
        end
    end

    # Count how many neurons landed in each Conditional Triangle case (reported in the filename); stable neurons are excluded — they have no binary to drop.
    global n_sibgate_both_thin             = n_both
    global n_sibgate_one_thin_org_dropped  = n_org_only
    global n_sibgate_one_thin_pert_dropped = n_pert_only

    println("compute_n2_relax_decision!: τ=$threshold, " *
            "both=$n_both org-only=$n_org_only pert-only=$n_pert_only " *
            "none=$n_none (plus $n_skipped_stable stable neurons)")
end
