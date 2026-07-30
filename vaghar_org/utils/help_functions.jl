global I_z_prev_up = []
global I_z_prev_down = []
global I_z_prev_up_perturbation = []
global I_z_prev_down_perturbation = []
global all_bounds_of_original = []
global all_bounds_of_perturbation = []
global I_pert_prev_up = []
global I_pert_prev_down = []

# HAR benchmark support: master switch (--internet_nets_benchmarks) plus the network's input box; off = the image pipeline, unchanged.
global internet_nets_benchmarks = false
# The HAR model's input domain, [lo, hi] per input coordinate ([-1,1]^561); `nothing` for every other model, meaning the [0,1] image domain.
global input_box_lo = nothing
global input_box_hi = nothing
global perturbed_input_in_domain = true

function input_domain_flat(n::Int)
    if internet_nets_benchmarks && input_box_lo !== nothing
        length(vec(input_box_lo)) == n ||
            error("input box has $(length(vec(input_box_lo))) coordinates but the " *
                  "input layer expects $n; check the <model>_box.txt sidecar.")
        return (Float64.(vec(input_box_lo)[1:n]), Float64.(vec(input_box_hi)[1:n]))
    end
    return (zeros(Float64, n), ones(Float64, n))
end

function input_domain_shaped(shape)
    if internet_nets_benchmarks && input_box_lo !== nothing
        return (reshape(Float64.(vec(input_box_lo)), shape),
                reshape(Float64.(vec(input_box_hi)), shape))
    end
    return (zeros(Float64, shape), ones(Float64, shape))
end

# Retired relaxation's on/off switch — permanently false; the encoder (core_ops.jl) still checks the name, so it must stay defined.
global use_relaxations::Bool = false
# Retired relaxation's tau threshold — unused while use_relaxations is false.
global relaxation_threshold::Float64 = 0.5
# Retired relaxation's relaxed-neuron counter — stays 0.
global relaxation_condition_count = 0
# LIVE: in the difference-bound propagation, handle a ReLU exactly when the neuron is stable in both networks (both active: the difference passes through; both inactive: it is zero).
global optimizing_intervals::Bool = true
# Retired relaxation's scoring choice (triangle area vs interval width) — unused.
global relaxation_gap_area::Bool = false

# The difference bounds [d_lo, d_hi]: per neuron, how far N's pre-activation can stray from N_pre's (computed by the difference zonotope).
global relu_diff_up_bounds::Vector   = []
global relu_diff_down_bounds::Vector = []
# N_pre's pre-activation bounds [l, u] per neuron; shifted by the difference bounds wherever N reuses them.
global n1_preact_up_bounds::Vector   = []
global n1_preact_down_bounds::Vector = []

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
# --geometric_intervals: relocation-aware interval bounds for translation/rotation (default OFF = no change).
global geometric_intervals::Bool = false
global geometric_diff_map = nothing          # the (T-I) matrix for the current build, or nothing
global geometric_input_shape = nothing       # size(input) for the current build
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

# ── advstd retired N1-probe LP bounds (Source C; kept because core_ops.jl
#    consults these globals — nothing populates them anymore) ────────────────
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

# ── advstd Technique 4 (SibGate): sibling-gated conditional triangle ────────
# When true, Technique-3's per-copy decision rule is unchanged but the
# emission switches:
#   • "both thin" tier: simple triangle on each copy + ONE pre-activation
#     coupling line linking org and pert pre-acts.
#   • "one thin" tier: conditional triangle on the relaxed copy, gated on
#     the kept sibling's binary, with intervals intersected against
#     Technique-1's unconditional per-copy bound.
# Activation requires --adv_std_n2_relax_threshold >= 0 (Technique 4 inherits
# Technique-3's tiered decision rule). Untyped for the same reason as the
# n_n2_relaxed_binaries_* counters above.
global adv_std_n2_sibling_gate = false
# Per-tier neuron counters, tracked at decision time and consumed by the
# filename composer.
global n_sibgate_both_thin             = 0
global n_sibgate_one_thin_org_dropped  = 0
global n_sibgate_one_thin_pert_dropped = 0
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

function clear_n2_relaxed_counters!()
    global n_n2_relaxed_binaries_org  = 0
    global n_n2_relaxed_binaries_pert = 0
end

function clear_sibgate_tier_counters!()
    global n_sibgate_both_thin             = 0
    global n_sibgate_one_thin_org_dropped  = 0
    global n_sibgate_one_thin_pert_dropped = 0
end

# ── advstd Technique 4 (SibGate): per-neuron MIP-state cache ───────────────
# core_ops.jl::relu() records the pre-activation AffExpr `x`, the bounds
# (l, u) the encoding uses, and the post-ReLU variable `x_rect` for every
# split N2 neuron in the org/perturbation passes. After both copies are
# encoded, apply_sibgate_constraints!(model, K) walks this cache + the
# n2_relax_decision dict and adds the coupling line / conditional-triangle
# upper bounds. Cleared per c_target by mip_reset (alongside layers_info_dict).
global n2_relu_state = Dict{Tuple{Int,Int,String},
                            NamedTuple{(:preact, :l, :u, :x_rect),
                                       Tuple{Any, Float64, Float64, Any}}}()

function clear_n2_relu_state!()
    global n2_relu_state = Dict{Tuple{Int,Int,String},
                                NamedTuple{(:preact, :l, :u, :x_rect),
                                           Tuple{Any, Float64, Float64, Any}}}()
end

# ── advstd Technique 6 (BoundTightPertRelax): per-copy relax decision ──────
# Precomputed once per N2 build (after all bound-tightening sources have
# populated their globals). Maps (layer, neuron) → (relax_org, relax_pert).
# core_ops.jl::relu() consults this dict to decide whether to emit the
# exact big-M encoding or a triangle LP relaxation for each copy.
global n2_relax_decision = Dict{Tuple{Int,Int}, Tuple{Bool,Bool}}()

function clear_n2_relax_decision!()
    global n2_relax_decision = Dict{Tuple{Int,Int}, Tuple{Bool,Bool}}()
end

# ── Technique 5 (Variable Hints) mode ────────────────────────────────────
# One active method (the retired prev/direct/direct_pgd variants are gone):
#   VH_PREV_PGD — shift N1's achieved pre-activation ẑ^N1 by the diff bound,
#                 clip to [l_n2, u_n2], take p from that interval's lengths,
#                 then route hint_val into set_start_value() (Gurobi Start)
#                 filtered by PGD consensus: fill where PGD is silent, leave
#                 where PGD agrees, withdraw where PGD disagrees.
#                 No VarHintVal / VarHintPri written.
# See §4.3 of advstd_techniques.tex and compute_varhint below.
@enum VarHintMode VH_OFF VH_PREV_PGD

"""
    parse_var_hint_mode(s::AbstractString) -> VarHintMode

Accepts `{"off", "prev_pgd"}` (case-insensitive). The retired modes
(`prev`, `direct`, `direct_pgd`, legacy `true`/`false`) raise.
"""
function parse_var_hint_mode(s::AbstractString)
    t = lowercase(strip(String(s)))
    if t == "off"
        return VH_OFF
    elseif t == "prev_pgd"
        return VH_PREV_PGD
    else
        error("Invalid --adv_std_var_hint value: '$(s)'. Expected one of {off, prev_pgd} (prev/direct/direct_pgd are retired).")
    end
end

"""
    var_hint_mode_label(m::VarHintMode) -> String

Lower-case label matching the CLI/sweep vocabulary (`"off"`, `"prev_pgd"`).
Used when serialising the mode back into result CSVs so the downstream
pipeline sees the same tokens it accepted on the command line.
"""
function var_hint_mode_label(m::VarHintMode)
    m == VH_OFF        && return "off"
    m == VH_PREV_PGD   && return "prev_pgd"
    error("unknown VarHintMode: $m")
end

"""
    hint_from_p(v1_bit::Int, p::Float64) -> (hint_val::Int, hint_pri::Int)

Tail of the VarHint rule. Agrees with N1 when `p ≥ 0.5`, flips when
`p < 0.5`; priority is a V-shape in `p`, maxing at 100 when `p ∈ {0, 1}`
and floored at 1 (priority 0 is reserved by Gurobi for "ignore"). Under
VH_PREV_PGD only `hint_val` is consumed (the Start channel has no priority).
"""
function hint_from_p(v1_bit::Int, p::Float64)
    hint_val = (p >= 0.5) ? v1_bit : 1 - v1_bit
    hint_pri = max(1, round(Int, 100 * abs(2 * p - 1)))
    return (hint_val, hint_pri)
end

"""
    compute_varhint(z1_preact, v1_bit, d_lo, d_hi, l_n2, u_n2)

The VH_PREV_PGD p-rule (§4.3). Shifts N1's achieved pre-activation
`z1_preact` by the weight-drift diff bound `[d_lo, d_hi]` to form the
predicted N2 pre-activation interval, clips to `[l_n2, u_n2]`, and takes
`p` as the fraction of the clipped interval on N1's side of zero
(`v1_bit=0` → `(-inf, 0]`; `v1_bit=1` → `[0, +inf)`).

Caller must only pass surviving binaries (`l_n2 < 0 < u_n2`). Returns
`(hint_val, hint_pri)` via `hint_from_p`.
"""
function compute_varhint(z1_preact::Float64, v1_bit::Int, d_lo::Float64, d_hi::Float64,
                         l_n2::Float64, u_n2::Float64)
    # Shifted interval I_i = [z1 + d_lo, z1 + d_hi], clipped to N2's sound bounds.
    I_lo = max(l_n2, z1_preact + d_lo)
    I_hi = min(u_n2, z1_preact + d_hi)
    if I_lo >= I_hi                       # degenerate (empty or point)
        return (v1_bit, 1)
    end
    len = I_hi - I_lo
    # Length of the portion of the clipped interval supporting N1's choice:
    #   v1_bit == 0 → supporting side is (-inf, 0], length = min(I_hi, 0) - I_lo
    #   v1_bit == 1 → supporting side is [0, +inf), length = I_hi - max(I_lo, 0)
    L_support = (v1_bit == 0) ? max(0.0, min(I_hi, 0.0) - I_lo) :
                                max(0.0, I_hi - max(I_lo, 0.0))
    p = L_support / len                   # in [0, 1]
    return hint_from_p(v1_bit, p)
end

"""
    apply_n1_var_hints!(m_n2, mode, n1_var_names, n1_var_values, n1_layers_info)

Set Gurobi hints on N2 binaries. `mode` is a `VarHintMode` enum:

  * `VH_OFF` — return without setting any hint.
  * `VH_PREV_PGD` — compute `p` via `compute_varhint` (shift ẑ^N1 by the
    diff bound, clip to [l_n2, u_n2], take p), then route `hint_val` through
    `set_start_value` filtered by PGD consensus. For each surviving binary:
    if PGD is silent (no Start set yet), fill with `hint_val`; if PGD agrees
    with `hint_val`, leave PGD's value in place; if PGD disagrees, call
    `set_start_value(a_i, nothing)` to withdraw both. **No `VarHintVal`/
    `VarHintPri` is written** — MIPStart is the sole channel.

For each surviving N2 binary `a_i` (with `l^{N2}_i < 0 < u^{N2}_i`):

  1. Look up N1's binary value `v^{N1}_i` by matching variable name in
     `n1_var_values`.
  2. Look up `z1_preact` (x_rect for active neurons, `l^{N1}_i / 2` for
     inactive) and the diff bound `[d_lo, d_hi]`; compute `hint_val`.
  3. Run the Start-consensus update above.

Uses globals `relu_diff_up_bounds`/`relu_diff_down_bounds` (populated by
`load_n1_diff_bounds!` in Phase 2) and reads `JuMP.start_value(v)` on each
binary, expecting PGD's hints (set by `hyper_attack_hints()`) to have
already run for this N2 MIP.

**Soundness.** Under all modes the set attributes (`VarHintVal`, `VarHintPri`,
`Start`) are advisory. No constraint, coefficient, or variable bound is
modified. MIP feasible set is unchanged, so δ_exact is preserved whenever the
solver terminates at OPTIMAL.
"""
function apply_n1_var_hints!(m_n2, mode::VarHintMode,
                             n1_var_names::Vector{String},
                             n1_var_values::Vector{Float64},
                             n1_layers_info::Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}})
    if mode == VH_OFF
        println("apply_n1_var_hints!: mode=off, no hints set")
        return
    end
    if isempty(n1_var_names)
        println("apply_n1_var_hints!: no N1 values to hint (mode=$(mode))")
        return
    end
    @assert length(n1_var_names) == length(n1_var_values) "n1_var_names and n1_var_values length mismatch ($(length(n1_var_names)) vs $(length(n1_var_values)))"
    # prev_pgd needs diff bounds populated by load_n1_diff_bounds!.
    if isempty(relu_diff_up_bounds) || isempty(relu_diff_down_bounds)
        error("apply_n1_var_hints!: mode=$(var_hint_mode_label(mode)) requires relu_diff_*_bounds globals. " *
              "Ensure load_n1_diff_bounds! or compute_diff_bounds_* has run before Phase 2 MIP build.")
    end
    # Name → value lookup for N1's primal.
    value_by_name = Dict{String, Float64}()
    for i in eachindex(n1_var_names)
        value_by_name[n1_var_names[i]] = n1_var_values[i]
    end
    # Number of ReLU layers per network copy; layers 1..K are the org copy,
    # K+1..2K are the pert copy. Same diff bound applies to both copies.
    K = length(relu_diff_up_bounds)
    applied = 0
    flipped = 0
    n2_binary_count = 0
    n_tier1_skipped = 0
    n_no_match = 0
    # PGD Start-consensus bucket counters.
    n_pgd_silent_filled     = 0
    n_pgd_agreed_nop        = 0
    n_pgd_disagreed_withdrew = 0
    for v in JuMP.all_variables(m_n2)
        if !JuMP.is_binary(v); continue; end
        n2_binary_count += 1
        bin_name = JuMP.name(v)
        v1_raw = get(value_by_name, bin_name, nothing)
        if v1_raw === nothing
            n_no_match += 1
            continue
        end
        v1_bit = Int(round(v1_raw))  # N1's binary incumbent, rounded to {0,1}
        # Parse (layer, neuron) from the binary's name. ReLU binaries are named
        # "{nv}a_layerCount{lc}_neuronCount{nc}_{layer}_{neuron}" (see
        # MIPVerify core_ops.jl). Other binaries live in the same MIP —
        # conf_n1_bin_k, conf_n1_est_bin_k, max-anonymous (from maximum_ge) —
        # and we must skip them, not parse. The regex matches only the ReLU shape.
        rgx_match = match(r"a_layerCount\d+_neuronCount\d+_(\d+)_(\d+)$", bin_name)
        rgx_match === nothing && continue
        layer = parse(Int, rgx_match.captures[1])
        neuron = parse(Int, rgx_match.captures[2])
        # N2's tightened bounds (global layers_info_dict was rebuilt during N2's
        # MIP construction with Tech 2's final integration).
        haskey(layers_info_dict, (layer, neuron)) || continue
        (u_n2, l_n2, _var_idx) = layers_info_dict[(layer, neuron)]
        # Tier 1: Tech 2 already eliminated this binary. Should be unreachable
        # because relu() would not have created the binary — defensive skip.
        if l_n2 >= 0 || u_n2 <= 0
            n_tier1_skipped += 1
            continue
        end
        # Compute (hint_val, hint_pri) via the prev_pgd p-rule.
        local hint_val::Int
        local hint_pri::Int
        # N1's bounds for this neuron — keys match because both networks share
        # architecture; n1_layers_info was saved/loaded in Phase 1/2.
        haskey(n1_layers_info, (layer, neuron)) || continue
        (u_n1, l_n1, _) = n1_layers_info[(layer, neuron)]
        # N1's pre-activation scalar. The big-M encoding only fixes hat_z
        # exactly when the ReLU was active (then x_rect == hat_z). When
        # inactive, hat_z is free in [l_n1, 0]; take the midpoint as a proxy.
        if v1_bit == 1
            # x_rect name differs from a name by one substring: "a_layerCount" → "x_rect_layerCount"
            x_rect_name = replace(bin_name, "a_layerCount" => "x_rect_layerCount"; count=1)
            z1_preact = get(value_by_name, x_rect_name, 0.0)
        else
            z1_preact = l_n1 / 2  # midpoint of the inactive range [l_n1, 0]
        end
        # Diff bound for this neuron. Layer 1..K → org copy (r = layer);
        # layer K+1..2K → pert copy (r = layer - K). Same diff for both copies.
        r = (layer <= K) ? layer : (layer - K)
        if r < 1 || r > K
            continue  # layer index out of diff-bound range; skip defensively
        end
        diff_up_vec   = vec(relu_diff_up_bounds[r])
        diff_down_vec = vec(relu_diff_down_bounds[r])
        if neuron < 1 || neuron > length(diff_up_vec)
            continue
        end
        d_hi = diff_up_vec[neuron]
        d_lo = diff_down_vec[neuron]
        (hint_val, hint_pri) = compute_varhint(z1_preact, v1_bit, d_lo, d_hi, l_n2, u_n2)
        if hint_val != v1_bit; flipped += 1; end
        # Start-consensus: route hint_val through set_start_value, filtered
        # by PGD. Expect hyper_attack_hints() to have run earlier in Phase 2;
        # if PGD failed, JuMP.start_value is nothing on every binary and
        # this mode degrades to "varHint fills every gap".
        v_pgd_raw = JuMP.start_value(v)
        if v_pgd_raw === nothing
            JuMP.set_start_value(v, Float64(hint_val))
            n_pgd_silent_filled += 1
        else
            v_pgd_bit = Int(round(v_pgd_raw))
            if v_pgd_bit == hint_val
                n_pgd_agreed_nop += 1   # leave PGD's Start in place
            else
                JuMP.set_start_value(v, nothing)
                n_pgd_disagreed_withdrew += 1
            end
        end
        applied += 1
    end
    println("apply_n1_var_hints!: mode=$(mode), Start-consensus on $applied of $n2_binary_count N2 binaries " *
            "(pgd-silent-filled=$n_pgd_silent_filled, pgd-agreed-nop=$n_pgd_agreed_nop, " *
            "pgd-disagreed-withdrew=$n_pgd_disagreed_withdrew, flipped=$flipped, " *
            "no-N1-match=$n_no_match, Tech2-eliminated-but-survived=$n_tier1_skipped)")
    if n2_binary_count > 0 && applied == 0
        error("apply_n1_var_hints!: 0 of $n2_binary_count N2 binaries received a hint (mode=$(mode)). " *
              "Aborting before optimize! so no mislabeled results are produced. " *
              "Likely cause: variable-name mismatch (regex vs set_name) or missing inputs.")
    end
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
    # Keep a recognisable PREFIX (token + model + mode) *and* the TAIL, with
    # the hash between them. The tail carries the tags every consumer parses --
    # _depGuardFix (soundness/staleness), _PerturbedIntervals, _cTagN -- so
    # cutting it, as a prefix-only truncation does, makes the file read as a
    # different (and typically stale) run and it is silently dropped from the
    # tables. The legend below still recovers the full name either way.
    tail_len = min(length(basename), 64)
    tail = basename[end-tail_len+1:end]
    head_len = max_name - length(h) - 2 - length(tail)  # 2 underscores
    short_name = basename[1:head_len] * "_" * h * "_" * tail
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
    pgd_lower_bound = haskey(d, :suboptimal_solution) ? d[:suboptimal_solution] : 0.0
    lower_bound = max(d[:incumbent_obj], pgd_lower_bound)
    row = "c_source=" * string(c_tag-1) * "," *
          "c_target=" * string(c_target-1) * "," *
          "lower_bound=" * string(lower_bound) * "," *
          "upper_bound=" * string(d[:best_bound]) * "," *
          "optimization_time=" * string(d[:solve_time]) * "," *
          "hyper_attack_time=" * string(hyper_time) * "," *
          "solve_status=" * string(d[:SolveStatus]) * "," *
          "n2_org_probe_eliminated_binaries=" * string(n_probe_eliminated_binaries_org) * "," *
          "n2_pert_probe_eliminated_binaries=" * string(n_probe_eliminated_binaries_pert) * "," *
          "n2_org_relaxed_binaries=" * string(n_n2_relaxed_binaries_org) * "," *
          "n2_pert_relaxed_binaries=" * string(n_n2_relaxed_binaries_pert) * "," *
          "sibgate_both_thin=" * string(n_sibgate_both_thin) * "," *
          "sibgate_one_thin_org_dropped=" * string(n_sibgate_one_thin_org_dropped) * "," *
          "sibgate_one_thin_pert_dropped=" * string(n_sibgate_one_thin_pert_dropped) * "," *
          "lp_optimization_time=" * string(n1_probe_lp_time)
    if haskey(d, :adv_std_flags)
        for (k, v) in pairs(d[:adv_std_flags])
            row *= "," * string(k) * "=" * string(v)
        end
    end
    return results * row * "\n"
end

