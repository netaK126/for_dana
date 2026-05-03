# ────────────────────────────────────────────────────────────────────────────
# advstd Technique 6 — BoundTightPertRelax (tiered per-copy N2 relaxation)
#
# compute_n2_relax_decision!(threshold) precomputes, for each N2 ReLU neuron,
# whether the original-input copy and/or the perturbed-input copy should be
# LP-relaxed (triangle) instead of big-M-encoded. Populates the global dict
# `n2_relax_decision` that core_ops.jl::relu() consults.
#
# Decision rule (tiered):
#   g_org  = tri_gap(l_org,  u_org)
#   g_pert = tri_gap(l_pert, u_pert)
#   if max(g_org, g_pert) ≤ τ  → relax BOTH copies
#   elif min(g_org, g_pert) ≤ τ → relax ONLY the smaller-gap copy
#   else                        → keep both exact
#
# The (l, u) bounds come from intersect_per_copy_bounds (defined in
# core_ops.jl), which mirrors exactly what relu() will intersect at emission
# time. This is the load-bearing invariant for the soundness of Technique 6:
# every (ẑ, z⁺) feasible under the exact big-M ReLU on [l, u] is inside the
# triangle on [l, u], so F_exact ⊆ F_BTPR and δ_BTPR ≥ δ_exact.
# ────────────────────────────────────────────────────────────────────────────

function compute_n2_relax_decision!(threshold::Real)
    global n2_relax_decision
    clear_n2_relax_decision!()

    if threshold < 0.0
        return
    end

    # Number of ReLU layers in one N2 pass (org or pert).
    # relu_diff_up_bounds is populated by compute_diff_and_comp_bounds /
    # compute_diff_bounds_zonotope / load_n1_diff_bounds! — always present
    # when --adv_std_bound_tightening is true, which is a precondition for
    # --adv_std_n2_relax_threshold to take effect (enforced by run.jl's warning).
    K = length(relu_diff_up_bounds)
    if K == 0
        println("compute_n2_relax_decision!: relu_diff_up_bounds empty " *
                "(bound_tightening off?) — skipping decision precompute")
        return
    end

    # Starting (l, u) for the decision: use a wide outer envelope. The helper
    # tightens via Sources A/B/C to the same interval relu() will use at
    # emission. Starting looser than relu()'s upstream interval-arithmetic
    # result only makes the decision more conservative (fewer relaxations) —
    # still sound (see Technique 6 soundness proof in advstd_techniques.tex).
    l_init = -Inf
    u_init = +Inf

    n_both = 0
    n_org_only = 0
    n_pert_only = 0
    n_none = 0
    n_skipped_stable = 0

    for m_idx in 1:K
        n_neurons = length(relu_diff_up_bounds[m_idx])
        for k_idx in 1:n_neurons
            # Mirror relu()'s per-copy nn_layer keys. After mip_reset before
            # the N2 build, neurons_names.layer increments 1..K during the
            # org pass (v_in |> nn) then continues K+1..2K during the pert
            # pass (v_x0 |> nn). n1_neuron_bounds (from N1's dual-copy MIP)
            # has the same key structure, so Source A fires for both copies.
            nn_layer_org  = m_idx
            nn_layer_pert = m_idx + K

            (l_org,  u_org)  = intersect_per_copy_bounds(
                l_init, u_init,
                nn_layer_org, k_idx,
                m_idx, k_idx,
                "org",
            )
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
                # Dual-key layout: relu()'s lookup uses (neurons_names.layer,
                # neurons_names.neuron), which differs between the org and
                # pert passes. We store the per-copy boolean under each
                # pass's key; the unused slot is false.
                if relax_org
                    n2_relax_decision[(nn_layer_org, k_idx)]  = (true, false)
                end
                if relax_pert
                    n2_relax_decision[(nn_layer_pert, k_idx)] = (false, true)
                end
            end
        end
    end

    println("compute_n2_relax_decision!: τ=$threshold, " *
            "both=$n_both org-only=$n_org_only pert-only=$n_pert_only " *
            "none=$n_none (plus $n_skipped_stable stable neurons)")
end
