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
# Retired relaxation's scoring choice (triangle area vs interval width) — unused.
global relaxation_gap_area::Bool = false

# The difference bounds [d_lo, d_hi]: per neuron, how far N's pre-activation can stray from N_pre's (computed by the difference zonotope).
global relu_diff_up_bounds::Vector   = []
global relu_diff_down_bounds::Vector = []
# N_pre's pre-activation bounds [l, u] per neuron; shifted by the difference bounds wherever N reuses them.
global n1_preact_up_bounds::Vector   = []
global n1_preact_down_bounds::Vector = []

# Retired transfer switch — permanently false; kept because the encoder (core_ops.jl) checks it.
global no_n1_binaries_and_relaxtions_only_on_n2::Bool = false
# Per-neuron perturbation-difference intervals (perturbed copy vs clean copy) — the Conditional Triangle's coupling input.
global relu_n2pert_up_bounds::Vector   = []
global relu_n2pert_down_bounds::Vector = []
# Each copy's pre-activation bounds [l, u] — the Conditional Triangle's gating input.
global n2_preact_up_bounds::Vector     = []
global n2_preact_down_bounds::Vector   = []

# Translation/rotation: use the exact (T-I) relocation map, composed through the first layer, for the perturbation-difference intervals (run.jl sets it; default true).
global geometric_intervals::Bool = false
# The current build's (T-I) matrix; `nothing` for every non-relocation perturbation.
global geometric_diff_map = nothing
# The current build's input shape, for reshaping the (T-I) map.
global geometric_input_shape = nothing
# Retired transfer switch — permanently false; kept because the encoder (core_ops.jl) checks it.
global bound_n2_relu_using_zonotope::Bool = false
# Retired transfer threshold — permanently disabled (-1); kept because the encoder (core_ops.jl) checks it.
global n1_stability_relax_threshold::Float64 = -1.0
# Retired transfer switch — permanently false; kept because the encoder (core_ops.jl) checks it.
global bound_n2_xp_using_composed::Bool = false
# Retired transfer switch — permanently false; kept because the encoder (core_ops.jl) checks it.
global constrain_n2_xp_via_n1_zonotope::Bool = false

# Zonotope Bound Tightening's output: per-neuron pre-activation bounds from the zonotope propagated through the network over the input box (exact for both copies — every encoder bounds the perturbed input to the same box), intersected into each ReLU's [l, u] by the encoder; empty when --adv_std_zono_bounds / --nn1_zono_bounds is off.
global n2_abs_up_bounds::Vector   = []
global n2_abs_down_bounds::Vector = []

# Retired technique's bound arrays — permanently empty; kept because the encoder (core_ops.jl) checks them for every neuron.
global n2_probe_up_bounds_org::Vector   = []
global n2_probe_down_bounds_org::Vector = []
global n2_probe_up_bounds_pert::Vector  = []
global n2_probe_down_bounds_pert::Vector = []
# Retired technique's counters — always 0; kept only so old and new result lines share the same columns.
global n_probe_eliminated_binaries_org::Int = 0
global n_probe_eliminated_binaries_pert::Int = 0
global n1_probe_lp_time::Float64 = 0.0

# The Conditional Triangle's tau: relax a copy's ReLU (drop its binary for a triangle) when its triangle-gap area is <= tau; -1 disables.
global adv_std_n2_relax_threshold = -1.0
# How many clean-copy / perturbed-copy ReLUs the Conditional Triangle relaxed this run (reported in the filename and result line); must stay untyped — the encoder already binds these names before this file loads.
global n_n2_relaxed_binaries_org  = 0
global n_n2_relaxed_binaries_pert = 0

# Conditional Triangle emission switch: gate a lone relaxed copy's triangle on its sibling's binary, and couple two relaxed copies by their pre-activation difference interval; needs tau >= 0, untyped for the same load-order reason as above.
global adv_std_n2_sibling_gate = false
# How many neurons landed in each Conditional Triangle case this run (both copies relaxed / only clean / only perturbed) — reported in the filename.
global n_sibgate_both_thin             = 0
global n_sibgate_one_thin_org_dropped  = 0
global n_sibgate_one_thin_pert_dropped = 0
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

# N_pre's verified per-neuron bounds for this class pair; the encoder intersects each of N's bounds with them, shifted by the difference bounds.
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

# The Conditional Triangle must connect a neuron to its sibling in the OTHER copy — but the encoder builds one copy at a time,
# so while encoding it saves each neuron's pieces here (input expression, bounds, output variable) and a second pass adds the
# cross-copy constraints once both copies exist. Emptied for every class pair.
global n2_relu_state = Dict{Tuple{Int,Int,String},
                            NamedTuple{(:preact, :l, :u, :x_rect),
                                       Tuple{Any, Float64, Float64, Any}}}()

function clear_n2_relu_state!()
    global n2_relu_state = Dict{Tuple{Int,Int,String},
                                NamedTuple{(:preact, :l, :u, :x_rect),
                                           Tuple{Any, Float64, Float64, Any}}}()
end

# The Conditional Triangle's decision map, filled once per build: (layer, neuron) -> (relax clean copy?, relax perturbed copy?); relu() reads it to choose the exact encoding or the triangle for each copy.
global n2_relax_decision = Dict{Tuple{Int,Int}, Tuple{Bool,Bool}}()

function clear_n2_relax_decision!()
    global n2_relax_decision = Dict{Tuple{Int,Int}, Tuple{Bool,Bool}}()
end

# Warm Start mode: off, or prev_pgd — hint each unstable neuron's phase from N_pre's solution shifted by the difference bounds, kept only where it agrees with the attack's hints.
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

# Turn the probability p that N keeps N_pre's phase into the hint: N_pre's phase when p >= 0.5, the opposite otherwise (the returned priority is unused under prev_pgd).
function hint_from_p(v1_bit::Int, p::Float64)
    hint_val = (p >= 0.5) ? v1_bit : 1 - v1_bit
    hint_pri = max(1, round(Int, 100 * abs(2 * p - 1)))
    return (hint_val, hint_pri)
end

# The Warm Start p-rule: shift N_pre's achieved pre-activation by the difference bounds (= where N's pre-activation can land), clip to N's bounds, and take p as the fraction of that interval on N_pre's side of zero.
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

# Warm Start: for each unstable neuron, compute the hint from N_pre's solution (the p-rule above) and reconcile it with the attack's
# hint — fill where the attack is silent, keep where they agree, withdraw both where they disagree; the attack's hints must be set first.
# Hints are advisory only: the feasible set is untouched, so the optimum is unchanged.
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
        # The phase N_pre's solution chose for this neuron (0 = inactive, 1 = active).
        v1_bit = Int(round(v1_raw))
        # Read (layer, neuron) out of the variable's name; the pattern matches only ReLU on/off binaries, so the MIP's other binaries (e.g. the argmax encoding's) are skipped.
        rgx_match = match(r"a_layerCount\d+_neuronCount\d+_(\d+)_(\d+)$", bin_name)
        rgx_match === nothing && continue
        layer = parse(Int, rgx_match.captures[1])
        neuron = parse(Int, rgx_match.captures[2])
        # N's tightened bounds [l, u] for this neuron, from the encoding just built.
        haskey(layers_info_dict, (layer, neuron)) || continue
        (u_n2, l_n2, _var_idx) = layers_info_dict[(layer, neuron)]
        # A stable neuron (always active or always inactive) has no binary to hint — skip (defensive; the encoder should not have created one).
        if l_n2 >= 0 || u_n2 <= 0
            n_tier1_skipped += 1
            continue
        end
        # Compute the hint via the p-rule.
        local hint_val::Int
        local hint_pri::Int
        # N_pre's verified bounds for the same neuron (same architecture, same keys).
        haskey(n1_layers_info, (layer, neuron)) || continue
        (u_n1, l_n1, _) = n1_layers_info[(layer, neuron)]
        # The pre-activation N_pre achieved at its optimum: read exactly when the neuron was active; when inactive the solution only says it is in [l, 0], so take the midpoint.
        if v1_bit == 1
            x_rect_name = replace(bin_name, "a_layerCount" => "x_rect_layerCount"; count=1)
            z1_preact = get(value_by_name, x_rect_name, 0.0)
        else
            z1_preact = l_n1 / 2
        end
        # The difference bounds are per network layer; layers 1..K are the clean copy and K+1..2K the perturbed copy, with the same bounds for both.
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
        # Reconcile with the attack's hint: fill where the attack is silent, keep where they agree, withdraw both where they disagree (if the attack found nothing, N_pre's hints simply fill everything).
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

