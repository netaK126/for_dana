global I_z_prev_up = []
global I_z_prev_down = []
global I_z_prev_up_perturbation = []
global I_z_prev_down_perturbation = []
global all_bounds_of_original = []
global all_bounds_of_perturbation = []
global I_pert_prev_up = []
global I_pert_prev_down = []

# ── Conditional-triangle relaxation globals ──────────────────────────────
# Populated by compute_diff_and_comp_bounds() before encoding n2_org/n2_pert.
# Each entry is a Float64 vector (one value per neuron), indexed by ReLU layer.
global use_relaxations::Bool = false
global relaxation_threshold::Float64 = 0.5
global relaxation_condition_count = 0
global optimizing_intervals::Bool = true
global relaxation_gap_area::Bool = false

# ── Activation conditional-triangle relaxation (n2_org pass) ─────────────
# BRIDGE paper Section 5 / eq. (4):
#   Replace a_n2_org binary by conditioning on a_n1_org (Npre's binary).
#   Threshold and conditional intervals use diff bounds + Npre preact bounds.
#   diff bounds: z_n2_org - z_n1_org  (inter-network pre-activation difference)
global relu_diff_up_bounds::Vector   = []
global relu_diff_down_bounds::Vector = []
# Npre pre-activation bounds (before ReLU clipping), used for both relaxations
global n1_preact_up_bounds::Vector   = []
global n1_preact_down_bounds::Vector = []

# ── Perturbation conditional-triangle relaxation (n2_pert pass) ──────────
# BRIDGE paper Section 5 / eq. (6):
#   Replace a_n2_pert binary by conditioning on a_n1_org (same Npre binary).
#   Threshold and conditional intervals use COMPOSED bounds + Npre preact bounds.
#   composed bounds: (z_n2_pert - z_n1_org) = diff + pert
global relu_comp_up_bounds::Vector   = []
global relu_comp_down_bounds::Vector = []

# ── N2-only perturbation relaxation (--no_n1_binaries_and_relaxtions_only_on_n2) ─
# Relax N2(x_p) by conditioning on N2(x) binary instead of N1(x).
# Uses perturbation bounds through N2: z_n2_pert - z_n2_org
# and N2 pre-activation bounds for conditional intervals.
global no_n1_binaries_and_relaxtions_only_on_n2::Bool = false
global relu_n2pert_up_bounds::Vector   = []
global relu_n2pert_down_bounds::Vector = []
global n2_preact_up_bounds::Vector     = []
global n2_preact_down_bounds::Vector   = []

# ── No-N1-encoding mode (--no_n1_encoding_at_all) ───────────────────────────
# Output-layer diff bounds: N2(x)[k] - N1(x)[k] ∈ [output_diff_down[k], output_diff_up[k]]
# Used to replace the entire N1 encoding with interval-bounded constraints on N2 outputs.
global no_n1_encoding_at_all::Bool = false
global no_n2_xp_encoding::Bool = false
global encode_n1_last_layer::Bool = false
global n1_last_layer_use_box_scalar::Bool = false
global n1_last_layer_prune_tol::Float64 = 0.0  # threshold: drop h_n1 vars with interval width <= this; 0 = only exact singletons
global n1_adaptive_prune_budget::Float64 = 0.0  # sensitivity-based pruning budget; 0 = disabled
global hybrid_solve::Bool = false  # two-phase solve: start with scalar bound, lazily add argmax constraints
global use_zonotope::Bool = false
global refined_relu_zonotope::Bool = false
global zonotope_conv::Bool = false   # activate zonotope propagation through conv layers
global zonotope_max_order::Int = 0   # max zonotope order (generators / neurons); 0 = unlimited
global bound_n2_relu_using_zonotope::Bool = false  # tighten ReLU preact bounds of N2 by intersecting with N1 preact + zonotope diff
global n1_stability_relax_threshold::Float64 = -1.0  # transfer-aware: replace N2 binary with triangle LP relaxation when N1 neuron is stable and gap <= threshold; <0 = disabled
global bound_by_zonotope_n2_hidden_neurons_which_are_not_relu::Bool = false  # add constraints on N2 final-layer logits using N1 output + zonotope diff
global n2_derived_preact_up_bounds   = []
global n2_derived_preact_down_bounds = []
global bound_n2_xp_using_composed::Bool = false  # tighten N2(x') preact bounds using N1 preact + composed bounds to eliminate binaries
global constrain_n2_xp_via_n1_zonotope::Bool = false  # add conditional constraints linking N2(x') post-ReLU to N2(x)'s binary using perturbation bounds via N1
global branch_priority_n2x_first::Bool = false  # set Gurobi BranchPriority: N2(x) binaries high, N2(x') low → resolve N2(x) first
global bound_n2_xp_output_using_composed::Bool = false  # bound N2(x') output logits using N1 output + composed (diff+pert) bounds
global n2_xp_derived_preact_up_bounds   = []
global n2_xp_derived_preact_down_bounds = []

# ── advstd Technique 4 + zono bounds (--adv_std_zono_bounds) ───────────────
# Per-ReLU-layer scalar bounds on N2's pre-activation, produced by
# compute_n2_bounds_zonotope_with_n1_tighten (Source B). Independent of the
# diff-zonotope path (Source A), intersected against the ReLU [l,u] inside
# core_ops.jl::relu(). Empty when the --adv_std_zono_bounds flag is off.
global n2_abs_up_bounds::Vector   = []
global n2_abs_down_bounds::Vector = []

# ── advstd Technique 4 + N1-probe LP bounds (--adv_std_n1_probe) ───────────
# Per-ReLU-layer scalar bounds on N2's pre-activation, produced by
# compute_n2_bounds_n1_probe_lp (Source C). Computed via OBBT on a joint
# LP-relaxed (N1 + N2) model using N1's post-solve bounds for the triangle
# relaxation. Separate arrays for the "org" (clean-input) and "perturbation"
# (perturbed-input) network_version passes, since the probe runs once per
# pass with the appropriate input seed.
global n2_probe_up_bounds_org::Vector   = []
global n2_probe_down_bounds_org::Vector = []
global n2_probe_up_bounds_pert::Vector  = []
global n2_probe_down_bounds_pert::Vector = []
# Count of N2 binaries eliminated *specifically by the N1-probe LP step*:
# neurons that would still have been split after Technique 4 + Source A/B,
# but whose probe-derived bound is single-signed. Tracked separately for
# N2(x) ("org") and N2(x') ("pert") so the result file can report both
# numbers. `n_probe_eliminated_binaries` is the sum of the two and is
# retained for filename composition.
global n_probe_eliminated_binaries_org::Int = 0
global n_probe_eliminated_binaries_pert::Int = 0
global n_probe_eliminated_binaries::Int = 0
global n1_probe_lp_time::Float64 = 0.0

# ── advstd Technique 6: N1-gated N2/N2p triangle LP relaxation ─────────────
# When >= 0, core_ops.jl::relu() replaces the big-M binary encoding of an
# N2/N2p ReLU with a three-inequality triangle LP whenever the
# triangle-gap-area of N1's interval at the corresponding neuron is
# <= this threshold. Sound over-approximation: delta_relaxed >= delta_exact.
# -1 = disabled (default).
global adv_std_n2_relax_threshold = -1.0
# Counters for the filename suffix / result-line columns — how many
# N2(x) and N2(x') ReLU binaries were replaced by triangles this run.
# NOTE: no `::Int` type annotation here because core_ops.jl (included
# earlier via net_components.jl) already references these globals inside
# relu() via `global n_n2_relaxed_binaries_* += 1`, which creates an
# implicit untyped binding. Re-declaring with a type would raise
# "cannot set type for global ... it already has a value" at include time.
global n_n2_relaxed_binaries_org  = 0
global n_n2_relaxed_binaries_pert = 0
global output_diff_up_bounds::Vector{Float64}   = Float64[]
global output_diff_down_bounds::Vector{Float64} = Float64[]
# N1 output-layer logit bounds (final linear layer of N1 over admissible inputs).
# Used by --bound_by_zonotope_n2_hidden_neurons_which_are_not_relu to bound
# N2's output logits as [n1_out_down + d_lo, n1_out_up + d_hi].
global n1_output_up_bounds::Vector{Float64}   = Float64[]
global n1_output_down_bounds::Vector{Float64} = Float64[]
# Output-level perturbation bounds: N2(x')[k] - N2(x)[k] and N1(x')[k] - N1(x)[k]
# Used by --constrain_n1_xp to add conf(N1,x',c_target)<=0 constraint.
global output_n2_pert_up::Vector{Float64}   = Float64[]
global output_n2_pert_down::Vector{Float64} = Float64[]
global output_n1_pert_up::Vector{Float64}   = Float64[]
global output_n1_pert_down::Vector{Float64} = Float64[]

# ── N1 last-hidden-layer bounds (--encode_n1_last_layer) ────────────────────
# Post-ReLU bounds on N1's last hidden layer activations: a_n1_last[i] ∈ [down, up]
# Post-ReLU diff bounds at last hidden layer: a_n2_last[i] - a_n1_last[i] ∈ [down, up]
# Used to create interval-bounded MIP variables for N1's last hidden layer,
# then encode the final linear layer exactly → exact conf_n1_x and delta_diff.
global n1_last_hidden_up::Vector{Float64}       = Float64[]
global n1_last_hidden_down::Vector{Float64}     = Float64[]
global last_hidden_diff_up::Vector{Float64}     = Float64[]
global last_hidden_diff_down::Vector{Float64}   = Float64[]

# ── Diff-bounds cache (avoid recomputing zonotope/interval across class pairs) ──
# The diff bounds depend only on (nn1, nn2, perturbation) — NOT on (c_tag, c_target).
# Cache after first computation, restore on subsequent class pairs.
mutable struct DiffBoundsCache
    valid::Bool
    relu_diff_up::Vector{Array{Float64}}
    relu_diff_down::Vector{Array{Float64}}
    n1_preact_up::Vector{Array{Float64}}
    n1_preact_down::Vector{Array{Float64}}
    relu_comp_up::Vector{Array{Float64}}
    relu_comp_down::Vector{Array{Float64}}
    output_diff_up::Vector{Float64}
    output_diff_down::Vector{Float64}
    output_n2_pert_up::Vector{Float64}
    output_n2_pert_down::Vector{Float64}
    output_n1_pert_up::Vector{Float64}
    output_n1_pert_down::Vector{Float64}
    n1_last_hidden_up::Vector{Float64}
    n1_last_hidden_down::Vector{Float64}
    last_hidden_diff_up::Vector{Float64}
    last_hidden_diff_down::Vector{Float64}
end
global diff_bounds_cache = DiffBoundsCache(false,
    Array{Float64}[], Array{Float64}[], Array{Float64}[], Array{Float64}[],
    Array{Float64}[], Array{Float64}[],
    Float64[], Float64[], Float64[], Float64[], Float64[], Float64[],
    Float64[], Float64[], Float64[], Float64[])

function save_diff_bounds_to_cache!()
    global diff_bounds_cache
    diff_bounds_cache.relu_diff_up        = [copy(v) for v in relu_diff_up_bounds]
    diff_bounds_cache.relu_diff_down      = [copy(v) for v in relu_diff_down_bounds]
    diff_bounds_cache.n1_preact_up        = [copy(v) for v in n1_preact_up_bounds]
    diff_bounds_cache.n1_preact_down      = [copy(v) for v in n1_preact_down_bounds]
    diff_bounds_cache.relu_comp_up        = [copy(v) for v in relu_comp_up_bounds]
    diff_bounds_cache.relu_comp_down      = [copy(v) for v in relu_comp_down_bounds]
    diff_bounds_cache.output_diff_up      = copy(output_diff_up_bounds)
    diff_bounds_cache.output_diff_down    = copy(output_diff_down_bounds)
    diff_bounds_cache.output_n2_pert_up   = copy(output_n2_pert_up)
    diff_bounds_cache.output_n2_pert_down = copy(output_n2_pert_down)
    diff_bounds_cache.output_n1_pert_up   = copy(output_n1_pert_up)
    diff_bounds_cache.output_n1_pert_down = copy(output_n1_pert_down)
    diff_bounds_cache.n1_last_hidden_up   = copy(n1_last_hidden_up)
    diff_bounds_cache.n1_last_hidden_down = copy(n1_last_hidden_down)
    diff_bounds_cache.last_hidden_diff_up   = copy(last_hidden_diff_up)
    diff_bounds_cache.last_hidden_diff_down = copy(last_hidden_diff_down)
    diff_bounds_cache.valid = true
    println("diff_bounds_cache: saved ($(length(diff_bounds_cache.relu_diff_up)) ReLU layers)")
end

function restore_diff_bounds_from_cache!()
    global diff_bounds_cache
    global relu_diff_up_bounds, relu_diff_down_bounds
    global n1_preact_up_bounds, n1_preact_down_bounds
    global relu_comp_up_bounds, relu_comp_down_bounds
    global output_diff_up_bounds, output_diff_down_bounds
    global output_n2_pert_up, output_n2_pert_down
    global output_n1_pert_up, output_n1_pert_down
    global n1_last_hidden_up, n1_last_hidden_down
    global last_hidden_diff_up, last_hidden_diff_down
    relu_diff_up_bounds   = [copy(v) for v in diff_bounds_cache.relu_diff_up]
    relu_diff_down_bounds = [copy(v) for v in diff_bounds_cache.relu_diff_down]
    n1_preact_up_bounds   = [copy(v) for v in diff_bounds_cache.n1_preact_up]
    n1_preact_down_bounds = [copy(v) for v in diff_bounds_cache.n1_preact_down]
    relu_comp_up_bounds   = [copy(v) for v in diff_bounds_cache.relu_comp_up]
    relu_comp_down_bounds = [copy(v) for v in diff_bounds_cache.relu_comp_down]
    output_diff_up_bounds   = copy(diff_bounds_cache.output_diff_up)
    output_diff_down_bounds = copy(diff_bounds_cache.output_diff_down)
    output_n2_pert_up       = copy(diff_bounds_cache.output_n2_pert_up)
    output_n2_pert_down     = copy(diff_bounds_cache.output_n2_pert_down)
    output_n1_pert_up       = copy(diff_bounds_cache.output_n1_pert_up)
    output_n1_pert_down     = copy(diff_bounds_cache.output_n1_pert_down)
    n1_last_hidden_up       = copy(diff_bounds_cache.n1_last_hidden_up)
    n1_last_hidden_down     = copy(diff_bounds_cache.n1_last_hidden_down)
    last_hidden_diff_up     = copy(diff_bounds_cache.last_hidden_diff_up)
    last_hidden_diff_down   = copy(diff_bounds_cache.last_hidden_diff_down)
    println("diff_bounds_cache: restored from cache ($(length(relu_diff_up_bounds)) ReLU layers)")
end

mutable struct ReuseBoundAndDepsConfig
    is_reuse_bounds_and_deps::Bool
    reusable_indexes::Int
    reusable_bounds::Vector{Float64}
    reusable_deps::Vector{Any}
end
reuse_bounds_conf = ReuseBoundAndDepsConfig(false, 1, [],Any[])

mutable struct NeuronsAssignNames
    neuron::Int
    layer::Int
end
neurons_names = NeuronsAssignNames(0, 0)

mutable struct FirstMIPSolution
    solution::Float64
    time::Float64
end
first_mip_solution = FirstMIPSolution(-1.0, 0.0)

# ── Hybrid Solve state (lazy argmax for N1 confidence) ─────────────────
# When hybrid_solve=true, the MIP starts with a scalar lower bound on
# conf_n1 and no argmax binaries. The callback inspects integer-feasible
# solutions: if the scalar bound is too loose (actual min-margin >> L),
# it adds a lazy constraint tightening delta_diff.
mutable struct HybridSolveState
    active::Bool
    v_margin::Vector{Any}       # per-class margin variables (JuMP)
    delta_diff::Any             # JuMP variable
    conf_n2_x::Any              # JuMP variable
    c_tag::Int
    n_classes::Int
    L::Float64                  # scalar lower bound (delta_1 + 1e-3)
    n_cuts_added::Int
end
hybrid_solve_state = HybridSolveState(false, [], nothing, nothing, 0, 0, 0.0, 0)

layers_info_dict = Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}}()

# ── Advanced-standard mode: N1 neuron bounds for bound tightening ────
# Populated after solving N1's standard MIP from layers_info_dict.
# Consumed by relu() in core_ops.jl to tighten N2's big-M bounds.
global n1_neuron_bounds = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()

function set_n1_neuron_bounds(n1_layers_info::Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}})
    global n1_neuron_bounds = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
    for ((layer, neuron), (u, l, _)) in n1_layers_info
        n1_neuron_bounds[(layer, neuron)] = (u, l)
    end
    println("set_n1_neuron_bounds: loaded $(length(n1_neuron_bounds)) neuron bounds from N1")
end

function clear_n1_neuron_bounds()
    global n1_neuron_bounds = Dict{Tuple{Int,Int}, Tuple{Float64,Float64}}()
end

function clear_n2_abs_bounds()
    global n2_abs_up_bounds   = Array{Float64}[]
    global n2_abs_down_bounds = Array{Float64}[]
end

function clear_n2_probe_bounds()
    global n2_probe_up_bounds_org   = Array{Float64}[]
    global n2_probe_down_bounds_org = Array{Float64}[]
    global n2_probe_up_bounds_pert  = Array{Float64}[]
    global n2_probe_down_bounds_pert = Array{Float64}[]
    global n_probe_eliminated_binaries_org  = 0
    global n_probe_eliminated_binaries_pert = 0
    global n_probe_eliminated_binaries      = 0
    global n1_probe_lp_time                 = 0.0
end

function clear_n2_relaxed_counters!()
    global n_n2_relaxed_binaries_org  = 0
    global n_n2_relaxed_binaries_pert = 0
end

"""
    apply_n1_branch_priorities!(m_n2, n1_layers_info)

Set Gurobi BranchPriority for N2's binary variables based on N1's bound tightness.
Neurons with tighter bounds (smaller |u - l|) in N1 get higher priority in N2,
since they were likely the most contested during N1's solve.
"""
function apply_n1_branch_priorities!(m_n2, n1_layers_info::Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}})
    applied = 0
    for v in JuMP.all_variables(m_n2)
        if JuMP.is_binary(v)
            vname = JuMP.name(v)
            for ((layer, neuron), (u, l, _)) in n1_layers_info
                if occursin("_$(layer)_$(neuron)", vname) && endswith(vname, "_$(layer)_$(neuron)")
                    gap = abs(u - l)
                    priority = gap < 1.0 ? 10 : (gap < 5.0 ? 5 : 1)
                    MOI.set(JuMP.backend(m_n2), Gurobi.VariableAttribute("BranchPriority"), JuMP.index(v), priority)
                    applied += 1
                    break
                end
            end
        end
    end
    println("apply_n1_branch_priorities!: set priorities for $applied binaries")
end

"""
    compute_n1_var_scores(n1_pseudocosts)

Compute per-variable importance scores from Gurobi N1 pseudo-cost statistics.
The score `pd_down * n_down + pd_up * n_up` is the total LP objective
improvement that Gurobi attributed to branching on that variable during N1.
Variables Gurobi never branched on get score 0.
"""
function compute_n1_var_scores(n1_pseudocosts)
    scores = Dict{String, Float64}()
    for (name, pc) in n1_pseudocosts
        scores[name] = pc.pd_down * pc.n_down + pc.pd_up * pc.n_up
    end
    return scores
end

"""
    rank_to_priority(scores; max_pri=100)

Convert a Dict{name→score} into a Dict{name→Int} priority in [0, max_pri].
The highest-scoring nonzero variable gets `max_pri`; the lowest nonzero gets 1;
variables with score ≤ 0 get priority 0.
"""
function rank_to_priority(scores::Dict{String, Float64}; max_pri::Int=100)
    priorities = Dict{String, Int}()
    nonzero = sort([n for (n, s) in scores if s > 0]; by=n -> -scores[n])
    n_rank = length(nonzero)
    for (rank, name) in enumerate(nonzero)
        priorities[name] = max(1, round(Int, max_pri * (1 - (rank - 1) / n_rank)))
    end
    for (name, s) in scores
        if s <= 0
            priorities[name] = 0
        end
    end
    return priorities
end

"""
    apply_n1_branch_priorities_pseudocost!(m_n2, n1_pseudocosts, n1_layers_info)

Gurobi `BranchPriority` mode that uses N1's measured pseudo-cost × branch-count
ranking (instead of the bound-width heuristic). Falls back to the bounds-based
`apply_n1_branch_priorities!` if pseudo-costs are unavailable (e.g. legacy
saved N1 state with no pseudocost file).
"""
function apply_n1_branch_priorities_pseudocost!(m_n2, n1_pseudocosts, n1_layers_info::Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}})
    if n1_pseudocosts === nothing || isempty(n1_pseudocosts)
        @warn "pseudocost bp mode requested but n1_pseudocosts is empty — falling back to bounds heuristic"
        apply_n1_branch_priorities!(m_n2, n1_layers_info)
        return
    end
    priorities = rank_to_priority(compute_n1_var_scores(n1_pseudocosts))
    applied = 0
    for v in JuMP.all_variables(m_n2)
        if !JuMP.is_binary(v); continue; end
        name = JuMP.name(v)
        pri = get(priorities, name, nothing)
        if pri === nothing; continue; end
        MOI.set(JuMP.backend(m_n2), Gurobi.VariableAttribute("BranchPriority"), JuMP.index(v), pri)
        applied += 1
    end
    println("apply_n1_branch_priorities_pseudocost!: assigned BranchPriority to $applied N2 binaries " *
            "(from $(length(priorities)) N1 pseudocost entries)")
end

"""
    apply_n1_var_hints!(m_n2, n1_var_names, n1_var_values, n1_pseudocosts;
                        fix_mode=false)

Set Gurobi `VarHintVal` and `VarHintPri` on N2 binaries, using N1's optimal
binary values as the hint value and N1 pseudo-cost rank as the hint priority.
Unlike `BranchPriority`, these hints are soft and affect both the heuristics
and the branching rule. If pseudo-costs are unavailable, all priorities default
to a low uniform value (10) so Gurobi treats the hints as low-confidence.

When `fix_mode=true`, filter `n1_pseudocosts` down to names that still exist
as binaries in `m_n2` (post T2 elimination and T4 triangle relaxation) before
ranking, so the priority distribution is computed over the actual hintable set
rather than the pre-relaxation superset.
"""
function apply_n1_var_hints!(m_n2, n1_var_names::Vector{String},
                             n1_var_values::Vector{Float64},
                             n1_pseudocosts;
                             fix_mode::Bool=false)
    if isempty(n1_var_names)
        println("apply_n1_var_hints!: no N1 values to hint")
        return
    end
    # Build name → value lookup from the parallel vectors
    value_by_name = Dict{String, Float64}()
    for i in eachindex(n1_var_names)
        value_by_name[n1_var_names[i]] = n1_var_values[i]
    end
    # Priority source: pseudo-cost rank if available, else low uniform
    if n1_pseudocosts === nothing || isempty(n1_pseudocosts)
        @warn "apply_n1_var_hints!: n1_pseudocosts empty — using uniform priority 10 (low confidence)"
        priority_by_name = Dict{String, Int}(name => 10 for name in n1_var_names)
    elseif fix_mode
        n2_binary_names = Set{String}()
        for v in JuMP.all_variables(m_n2)
            if JuMP.is_binary(v)
                push!(n2_binary_names, JuMP.name(v))
            end
        end
        filtered_pc = Dict(name => pc for (name, pc) in n1_pseudocosts
                           if name in n2_binary_names)
        priority_by_name = rank_to_priority(compute_n1_var_scores(filtered_pc))
        println("apply_n1_var_hints!: fix_mode=true, filtered " *
                "$(length(n1_pseudocosts)) N1 pseudocost entries down to " *
                "$(length(filtered_pc)) N2-surviving entries before ranking")
    else
        priority_by_name = rank_to_priority(compute_n1_var_scores(n1_pseudocosts))
    end
    applied = 0
    for v in JuMP.all_variables(m_n2)
        if !JuMP.is_binary(v); continue; end
        name = JuMP.name(v)
        value = get(value_by_name, name, nothing)
        if value === nothing; continue; end
        pri = get(priority_by_name, name, 0)
        MOI.set(JuMP.backend(m_n2), Gurobi.VariableAttribute("VarHintVal"), JuMP.index(v), value)
        MOI.set(JuMP.backend(m_n2), Gurobi.VariableAttribute("VarHintPri"), JuMP.index(v), pri)
        applied += 1
    end
    println("apply_n1_var_hints!: set VarHintVal/VarHintPri on $applied N2 binaries")
end

mutable struct Results
    str::String
end
results = Results("")

@pyimport pickle
function mypickle(filename, obj)
    out = open(filename,"w")
    pickle.dump(obj, out)
    close(out)
 end

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end

function compute_acc(mnist, nn, is_conv,w_,h_,k_)
    num_correct = 0.0
    num_samples_ = 10000
    num_samples_ = min(num_samples_, num_samples(mnist.test))
    for sample_index in 1:num_samples_
        input = MIPVerify.get_image(mnist.test.images, sample_index)
        actual_label = MIPVerify.get_label(mnist.test.labels, sample_index)
        if is_conv
            predicted_label = (reshape(np.transpose(input),(1,w_,h_,k_))|> nn |> MIPVerify.get_max_index) - 1
        else
            predicted_label = ((input)|> nn |> MIPVerify.get_max_index) - 1
        end
        if actual_label == predicted_label
             num_correct += 1
        end
    end
    println("Model accuracy: " * string(num_correct / num_samples_))
end

function read_best_val_via_optimization(ss, tt, token_signature)
    file = open("/tmp/best_val_" * string(ss-1) * "_" * string(tt-1) * "_" * string(token_signature) * ".txt")
    line = readline(file)
    close(file)
    value = parse(Float64, line)
    return value
end

"""
    safe_filepath(dir, basename, ext=".txt") -> String

Build a file path ensuring the filename (basename + ext) stays within the
255-byte Linux filename limit.  When the name is too long, the filename is
shortened to a hash, and a `<dir>/_filename_legend.txt` file is created
that maps the hash back to the full original name (so flags are recoverable).
"""
function safe_filepath(dir::AbstractString, basename::AbstractString, ext::AbstractString=".txt")
    max_name = 255 - length(ext)
    if length(basename) <= max_name
        return dir * basename * ext
    end
    h = string(hash(basename), base=16)  # 16-char hex hash
    # Keep a recognisable prefix (token + model + mode) then append the hash
    trunc_len = max_name - length(h) - 1  # 1 for underscore separator
    short_name = basename[1:trunc_len] * "_" * h
    # Append to legend file so the full flag string is always recoverable
    mkpath(dir)
    legend_path = dir * "_filename_legend.txt"
    open(legend_path, "a") do f
        println(f, short_name * ext, " => ", basename * ext)
    end
    return dir * short_name * ext
end

function save_results(results_path, model_name, perturbation, perturbation_size, results_str, d, nn, ss, tt, w_, h_, k_,name_to_save,token_signature)
    mkpath(results_path)
    basename = token_signature*"_"*model_name * "_" * perturbation * "_" * create_perturbation_string(perturbation_size)*"_ctag"*string(ss)*"_"*name_to_save
    file = open(safe_filepath(results_path, basename), "w")
    write(file, results_str)
    close(file)
    try
        sample = d[:v_in]

        if k_ == 1
            sample = reshape(sample, w_, h_)
        else
            sample = reshape(sample, w_, h_,k_)
        end
        sample_reshaped = reshape(sample, 1, w_, h_,k_)
        sample_reshaped_result = argmax(sample_reshaped |> nn)
        println("Clean sample classification: ", sample_reshaped_result)
        matshow(sample, vmin=0, vmax=1)
        mypickle(results_path*string(ss)*"_"*string(tt)*"_org.p", sample)
        savefig(results_path*string(ss)*"_"*string(tt)*"_org.png")

        perturbed_sample = (d[:v_in_p])
        if k_ == 1
            perturbed_sample = reshape(perturbed_sample, w_, h_)
        else
            perturbed_sample = reshape(perturbed_sample, w_, h_,k_)
        end
        perturbed_sample_reshaped = reshape(perturbed_sample, 1, w_, h_,k_)
        perturbed_sample_reshaped_result = argmax(perturbed_sample_reshaped |> nn)
        println("Perturbed sample classification: ",perturbed_sample_reshaped_result)
        matshow(perturbed_sample, vmin=0, vmax=1)
        mypickle(results_path*string(ss)*"_"*string(tt)*"_perturbed.p", perturbed_sample)
        savefig(results_path*string(ss)*"_"*string(tt)*"_perturbed.png")
   catch e
        println("no results")
   end
end

function create_perturbation_string(perturbation_size)
    perturbation_size_string = ""
    for i in eachindex(perturbation_size)
        perturbation_size_string *= string(perturbation_size[i])
        if i <length(perturbation_size)
           perturbation_size_string *=","
        end
    end
    return perturbation_size_string
end

function get_default_tightening_options(optimizer)::Dict
    optimizer_type_name = string(typeof(optimizer()))
    if optimizer_type_name == "Gurobi.Optimizer"
        return Dict("OutputFlag" => 0, "TimeLimit" => 5)
    elseif optimizer_type_name == "Cbc.Optimizer"
        return Dict("logLevel" => 0, "seconds" => 20)
    else
        return Dict()
    end
end

function set_branch_priority_n2x_first!(m)
    n_org = 0
    n_pert = 0
    for v in JuMP.all_variables(m)
        if JuMP.is_binary(v)
            vname = JuMP.name(v)
            if startswith(vname, "n2_orga")
                MOI.set(m, Gurobi.VariableAttribute("BranchPriority"), JuMP.index(v), 10)
                n_org += 1
            elseif startswith(vname, "n2_perta")
                MOI.set(m, Gurobi.VariableAttribute("BranchPriority"), JuMP.index(v), 1)
                n_pert += 1
            end
        end
    end
    println("  branch_priority: N2(x) binaries=$n_org (priority=10), N2(x') binaries=$n_pert (priority=1)")
end

function my_callback(cb_data::Gurobi.CallbackData, where::Int32)
    if where == GRB_CB_MIPSOL
        resultP = Ref{Float64}()
        GRBcbget(cb_data, where, GRB_CB_MIPSOL_OBJ, resultP)
        run_time =Ref{Float64}()
        GRBcbget(cb_data, where, GRB_CB_RUNTIME, run_time)
        if first_mip_solution.solution == -1
            first_mip_solution.solution = resultP[]
            first_mip_solution.time = run_time[]
        end

        # (hybrid_solve tightening happens after optimize! in hybrid_solve_phase2!)
    end
end

function parse_numbers_to_Float64(input_str::String)
    str_numbers = rsplit(input_str, ",")
    numbers = Float64[]
    for str_num in str_numbers
        push!(numbers, parse(Float64, str_num))
    end
    return numbers
end

function parse_numbers_to_Int64(input_str::String)
    str_numbers = rsplit(input_str, ",")
    numbers = Int64[]
    for str_num in str_numbers
        push!(numbers, parse(Float64, str_num))
    end
    return numbers
end

function update_results_str(results, c_tag, c_target, d)
    hyper_time = haskey(d, :suboptimal_time) ? d[:suboptimal_time] : 0.0
    row = "c_source=" * string(c_tag-1) * "," *
          "c_target=" * string(c_target-1) * "," *
          "lower_bound=" * string(d[:incumbent_obj]) * "," *
          "upper_bound=" * string(d[:best_bound]) * "," *
          "optimization_time=" * string(d[:solve_time]) * "," *
          "hyper_attack_time=" * string(hyper_time) * "," *
          "solve_status=" * string(d[:SolveStatus]) * "," *
          "n2_org_probe_eliminated_binaries=" * string(n_probe_eliminated_binaries_org) * "," *
          "n2_pert_probe_eliminated_binaries=" * string(n_probe_eliminated_binaries_pert) * "," *
          "n2_org_relaxed_binaries=" * string(n_n2_relaxed_binaries_org) * "," *
          "n2_pert_relaxed_binaries=" * string(n_n2_relaxed_binaries_pert) * "," *
          "lp_optimization_time=" * string(n1_probe_lp_time)
    if haskey(d, :adv_std_flags)
        for (k, v) in pairs(d[:adv_std_flags])
            row *= "," * string(k) * "=" * string(v)
        end
    end
    return results * row * "\n"
end

# Read delta_1 (upper_bound) from a VHAGaR results file for a given c_target.
# Supports both formats:
#   New: c_source=0,c_target=3,lower_bound=...,upper_bound=...,optimization_time=...,hyper_attack_time=...
#   Old: source,target,incumbent_obj,best_bound,solve_time
function get_delta1_vaghar(results_path, c_target_index)
    open(results_path, "r") do io
        requested_line = ""
        while !eof(io)
            line_content = readline(io)
            if isempty(strip(line_content))
                continue
            end
            # Detect format by checking for key=value pairs
            if occursin("c_target=", line_content)
                # New named format
                kv = Dict(strip(k) => strip(v) for (k, v) in
                    (Base.split(pair, '=') for pair in Base.split(line_content, ',') if occursin("=", pair)))
                if haskey(kv, "c_target") && parse(Int, kv["c_target"]) == c_target_index - 1
                    requested_line = line_content
                end
            else
                # Old positional format
                tokens = Base.split(line_content, ',')
                if length(tokens) >= 4
                    target_in_file = parse(Int, tokens[2])
                    if target_in_file == c_target_index - 1
                        requested_line = line_content
                    end
                end
            end
        end
        if requested_line == ""
            println("Warning: no delta_1 found for c_target=$c_target_index in $results_path")
            return -1.0
        end
        # Parse the matched line
        if occursin("upper_bound=", requested_line)
            kv = Dict(strip(k) => strip(v) for (k, v) in
                (Base.split(pair, '=') for pair in Base.split(requested_line, ',') if occursin("=", pair)))
            return parse(Float64, kv["upper_bound"])
        else
            parsed_tokens = Base.split(requested_line, ',')
            return parse(Float64, parsed_tokens[4])
        end
    end
end