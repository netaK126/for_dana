function hyper_attack_hints(m, token, c_tag, c_target)
    av = JuMP.all_variables(m)
    if isfile("/tmp/fail_"*string(c_tag-1)*"_"*string(c_target-1)*"_"*token*".txt")
        rm("/tmp/fail_"*string(c_tag-1)*"_"*string(c_target-1)*"_"*token*".txt")
    else
        file = open("/tmp/booleans_"*string(c_tag-1)*"_"*string(c_target-1)*"_"*token*".txt", "r")
        data = read(file, String)
        data_array = (rsplit(data,","))
        arr_booleans = []
        for n in eachindex(data_array)
            append!(arr_booleans,parse(Float64, data_array[n]))
        end
        file = open("/tmp/strings_"*string(c_tag-1)*"_"*string(c_target-1)*"_"*token*".txt", "r")
        data = read(file, String)
        arr_strings = (rsplit(data,","))
        # find indexes
        indexes_ = []
        for n in eachindex(arr_strings)
            ind_to_save = -1
            for k in eachindex(av)
                if arr_strings[n] == JuMP.name(av[k])
                    ind_to_save = deepcopy(k)
                    break
                end
            end
            append!(indexes_,ind_to_save)
        end
        for n in eachindex(arr_strings)
            if indexes_[n]==-1 || arr_booleans[n] == -1
                continue
            end
            set_start_value(av[indexes_[n]],arr_booleans[n])
        end
    end
end

function hyper_attack_transfer(dataset, c_tag, c_target, token_signature,
                               model_name, model_path, model_path2,
                               perturbation, perturbation_size,
                               delta_1, c_tag_mode, n1_p_mode;
                               force_cpu::Bool=false)
    best_feasible_via_optimization = 0
    pre_time = 0
    pre_time = @elapsed begin
        cmd_args = ["python3", "./utils/hyper_attack_transfer.py",
            "--dataset", string(dataset),
            "--source", string(c_tag-1),
            "--target", string(c_target-1),
            "--token", token_signature,
            "--model", model_name,
            "--model_path", model_path*"th",
            "--model_path2", model_path2*"th",
            "--perturbation", perturbation,
            "--perturbation_size", create_perturbation_string(perturbation_size),
            "--delta1", string(delta_1),
            "--c_tag_mode", string(c_tag_mode),
            "--n1_p_mode", string(n1_p_mode)]
        if force_cpu
            push!(cmd_args, "--cpu")
        end
        run(Cmd(cmd_args))
        best_feasible_via_optimization = read_best_val_via_optimization(c_tag, c_target, token_signature)
    end
    return best_feasible_via_optimization, pre_time
end

function hyper_attack_transfer_distilation(dataset, c_tag, c_target, token_signature,
                               model_name, model_name2, model_path, model_path2,
                               perturbation, perturbation_size,
                               delta_1, c_tag_mode, n1_p_mode;
                               force_cpu::Bool=false)
    best_feasible_via_optimization = 0
    pre_time = 0
    pre_time = @elapsed begin
        cmd_args = ["python3", "./utils/hyper_attack_transfer.py",
            "--dataset", string(dataset),
            "--source", string(c_tag-1),
            "--target", string(c_target-1),
            "--token", token_signature,
            "--model", model_name,
            "--model2", model_name2,
            "--model_path", model_path*"th",
            "--model_path2", model_path2*"th",
            "--perturbation", perturbation,
            "--perturbation_size", create_perturbation_string(perturbation_size),
            "--delta1", string(delta_1),
            "--c_tag_mode", string(c_tag_mode),
            "--n1_p_mode", string(n1_p_mode)]
        if force_cpu
            push!(cmd_args, "--cpu")
        end
        run(Cmd(cmd_args))
        best_feasible_via_optimization = read_best_val_via_optimization(c_tag, c_target, token_signature)
    end
    return best_feasible_via_optimization, pre_time
end

function hyper_attack(dataset, c_tag, c_target, token_signature, model_name, model_path, perturbation, perturbation_size;
                      force_cpu::Bool=false)
    best_feasible_via_optimization = 0
    pre_time = 0
    pre_time = @elapsed begin
        cmd_args = ["python3", "./utils/hyper_attack.py","--dataset", string(dataset), "--source", string(c_tag-1),
        "--target", string(c_target-1), "--token", token_signature, "--model", model_name, "--model_path", model_path*"th", "--perturbation", perturbation,
         "--perturbation_size", create_perturbation_string(perturbation_size)]
        if force_cpu
            push!(cmd_args, "--cpu")
        end
        run(Cmd(cmd_args))
        best_feasible_via_optimization = read_best_val_via_optimization(c_tag, c_target, token_signature)
    end
    return best_feasible_via_optimization, pre_time
end

# ── Standard-to-transfer warm-start helpers ──────────────────────────────

"""
    extract_binary_values(m) -> (names::Vector{String}, values::Vector{Float64})

Extract solved binary variable names and values from a JuMP model after optimize!().
Returns empty vectors if the model has no solution.
"""
function extract_binary_values(m)
    binary_names = String[]
    binary_values = Float64[]
    if !JuMP.has_values(m)
        println("extract_binary_values: no solution available, returning empty")
        return binary_names, binary_values
    end
    for v in JuMP.all_variables(m)
        if JuMP.is_binary(v)
            push!(binary_names, JuMP.name(v))
            push!(binary_values, JuMP.value(v))
        end
    end
    println("extract_binary_values: extracted $(length(binary_names)) binary values")
    return binary_names, binary_values
end

"""
    apply_standard_warmstart!(m, std_names, std_values,
                              no_n1_encoding_at_all, n1_p_mode, no_n2_xp_encoding;
                              n1_only=false)

Apply binary values from a standard-mode solve as warm-start hints to a transfer-mode MIP.

Mapping (standard prefix → transfer prefix):
- "org"          → "n1_org"  (if N1 is encoded) + "n2_org"
- "perturbation" → "n1_pert" (if n1_p_mode)     + "n2_pert" (if N2(xp) encoded)

When `n1_only=true`, only "n1_org" is hinted — n1_pert, n2_org, and n2_pert are skipped.
"""
function apply_standard_warmstart!(m, std_names::Vector{String}, std_values::Vector{Float64},
                                   no_n1_encoding_at_all::Bool, n1_p_mode::Bool, no_n2_xp_encoding::Bool;
                                   n1_only::Bool=false)
    if isempty(std_names)
        println("apply_standard_warmstart!: no hints to apply")
        return
    end

    # Build name→index lookup for the transfer model
    av = JuMP.all_variables(m)
    name_to_idx = Dict{String, Int}()
    for k in eachindex(av)
        n = JuMP.name(av[k])
        if !isempty(n)
            name_to_idx[n] = k
        end
    end

    n_applied = 0
    n_missing = 0
    for i in eachindex(std_names)
        sname = std_names[i]
        sval = std_values[i]

        # Determine which transfer prefixes this standard binary maps to
        transfer_names = String[]
        if startswith(sname, "orga_") || startswith(sname, "orgx_rect_")
            prefix = startswith(sname, "orga_") ? "orga_" : "orgx_rect_"
            suffix = sname[length(prefix)+1:end]
            a_or_xrect = startswith(sname, "orga_") ? "a_" : "x_rect_"
            if !no_n1_encoding_at_all
                push!(transfer_names, "n1_org" * a_or_xrect * suffix)
            end
            if !n1_only
                push!(transfer_names, "n2_org" * a_or_xrect * suffix)
            end
        elseif startswith(sname, "perturbationa_") || startswith(sname, "perturbationx_rect_")
            if n1_only
                continue
            end
            prefix = startswith(sname, "perturbationa_") ? "perturbationa_" : "perturbationx_rect_"
            suffix = sname[length(prefix)+1:end]
            a_or_xrect = startswith(sname, "perturbationa_") ? "a_" : "x_rect_"
            if n1_p_mode && !no_n1_encoding_at_all
                push!(transfer_names, "n1_pert" * a_or_xrect * suffix)
            end
            if !no_n2_xp_encoding
                push!(transfer_names, "n2_pert" * a_or_xrect * suffix)
            end
        end

        for tname in transfer_names
            idx = get(name_to_idx, tname, -1)
            if idx > 0
                set_start_value(av[idx], sval)
                n_applied += 1
            else
                n_missing += 1
            end
        end
    end
    println("apply_standard_warmstart!: applied $n_applied hints, $n_missing names not found in transfer model")
end

# ── Advanced-standard mode: N1 → N2 solver state transfer ─────────────

"""
    extract_all_variable_values(m)

Extract ALL variable names and values (binary + continuous) from a solved JuMP model.
Returns empty vectors if no solution is available.
"""
function extract_all_variable_values(m)
    names = String[]
    values = Float64[]
    if !JuMP.has_values(m)
        println("extract_all_variable_values: no solution available, returning empty")
        return names, values
    end
    for v in JuMP.all_variables(m)
        push!(names, JuMP.name(v))
        push!(values, JuMP.value(v))
    end
    println("extract_all_variable_values: extracted $(length(names)) values")
    return names, values
end

"""
    apply_n1_hints!(m_n2, n1_names, n1_values)

Apply N1's solved variable values as MIP start hints to N2's model.
Uses direct name matching (both are standard-mode MIPs with identical architecture).
"""
function apply_n1_hints!(m_n2, n1_names::Vector{String}, n1_values::Vector{Float64})
    if isempty(n1_names)
        println("apply_n1_hints!: no hints to apply")
        return
    end
    av = JuMP.all_variables(m_n2)
    name_to_idx = Dict{String, Int}()
    for k in eachindex(av)
        n = JuMP.name(av[k])
        if !isempty(n)
            name_to_idx[n] = k
        end
    end
    applied = 0
    for i in eachindex(n1_names)
        idx = get(name_to_idx, n1_names[i], -1)
        if idx > 0
            set_start_value(av[idx], n1_values[i])
            applied += 1
        end
    end
    println("apply_n1_hints!: applied $applied / $(length(n1_names)) hints")
end

"""
    extract_vbasis(m)

Extract LP basis status (VBasis) for all variables from a solved Gurobi model.
Returns a Dict mapping variable name → basis status integer.
"""
function extract_vbasis(m)
    vbasis = Dict{String, Int}()
    first_err = nothing
    for v in JuMP.all_variables(m)
        try
            status = MOI.get(m, Gurobi.VariableAttribute("VBasis"), v)
            vbasis[JuMP.name(v)] = status
        catch e
            if first_err === nothing; first_err = e; end
        end
    end
    if isempty(vbasis) && first_err !== nothing
        open("/tmp/julia_extract_debug.log", "a") do io
            println(io, "[", now(), "] extract_vbasis: all queries failed. First error:")
            println(io, sprint(showerror, first_err))
        end
    end
    println("extract_vbasis: extracted $(length(vbasis)) basis statuses")
    return vbasis
end

"""
    extract_pseudocosts(m)

Extract Gurobi per-variable branching statistics from a solved model:
PsDDown/PsDUp (avg LP objective change per unit fractional value) and
PsDDownCnt/PsDUpCnt (number of times each variable was branched on).
Only binary variables are queried. Returns a Dict name → (pd_down, pd_up, n_down, n_up).
"""
function extract_pseudocosts(m)
    pseudocosts = Dict{String, NamedTuple{(:pd_down, :pd_up, :n_down, :n_up),
                                          Tuple{Float64, Float64, Float64, Float64}}}()
    first_err = nothing
    n_binaries = 0
    for v in JuMP.all_variables(m)
        if !JuMP.is_binary(v); continue; end
        n_binaries += 1
        try
            pd_down = MOI.get(m, Gurobi.VariableAttribute("PsDDown"), v)
            pd_up   = MOI.get(m, Gurobi.VariableAttribute("PsDUp"),   v)
            n_down  = MOI.get(m, Gurobi.VariableAttribute("PsDDownCnt"), v)
            n_up    = MOI.get(m, Gurobi.VariableAttribute("PsDUpCnt"),   v)
            pseudocosts[JuMP.name(v)] = (pd_down=pd_down, pd_up=pd_up,
                                          n_down=n_down, n_up=n_up)
        catch e
            if first_err === nothing; first_err = e; end
            continue
        end
    end
    if isempty(pseudocosts) && n_binaries > 0 && first_err !== nothing
        open("/tmp/julia_extract_debug.log", "a") do io
            println(io, "[", now(), "] extract_pseudocosts: all queries failed across $n_binaries binaries. First error:")
            println(io, sprint(showerror, first_err))
        end
    end
    n_nonzero = count(p -> (p.pd_down * p.n_down + p.pd_up * p.n_up) > 0,
                      values(pseudocosts))
    println("extract_pseudocosts: extracted $(length(pseudocosts)) binaries, " *
            "$n_nonzero with nonzero branching score")
    return pseudocosts
end

"""
    apply_vbasis!(m_n2, vbasis)

Apply LP basis status from N1 to N2's model as warm start for the root LP.
"""
function apply_vbasis!(m_n2, vbasis::Dict{String, Int})
    if isempty(vbasis)
        println("apply_vbasis!: no basis to apply")
        return
    end
    applied = 0
    for v in JuMP.all_variables(m_n2)
        name = JuMP.name(v)
        if haskey(vbasis, name)
            MOI.set(JuMP.backend(m_n2), Gurobi.VariableAttribute("VBasis"), JuMP.index(v), vbasis[name])
            applied += 1
        end
    end
    println("apply_vbasis!: applied $applied / $(length(vbasis)) basis statuses")
end

# ── N1 state persistence (advanced_standard_n1 → advanced_standard_n2) ──

"""
    save_n1_state(state_dir, c_tag, c_target, n1_var_names, n1_var_values,
                  n1_layers_info, n1_vbasis; n1_pseudocosts=nothing)

Save N1 solver state for one (c_tag, c_target) pair to disk.
Files: n1_vars_{c_tag}_{c_target}.bin, n1_layers_{c_tag}_{c_target}.bin,
       n1_vbasis_{c_tag}_{c_target}.bin, and optionally
       n1_pseudocosts_{c_tag}_{c_target}.bin when `n1_pseudocosts` is supplied.
"""
function save_n1_state(state_dir::String, c_tag::Int, c_target::Int,
                       n1_var_names::Vector{String}, n1_var_values::Vector{Float64},
                       n1_layers_info::Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}},
                       n1_vbasis::Dict{String, Int};
                       n1_pseudocosts=nothing)
    mkpath(state_dir)
    tag = "$(c_tag)_$(c_target)"
    serialize(joinpath(state_dir, "n1_vars_$tag.bin"), (n1_var_names, n1_var_values))
    serialize(joinpath(state_dir, "n1_layers_$tag.bin"), n1_layers_info)
    serialize(joinpath(state_dir, "n1_vbasis_$tag.bin"), n1_vbasis)
    if n1_pseudocosts !== nothing
        serialize(joinpath(state_dir, "n1_pseudocosts_$tag.bin"), n1_pseudocosts)
    end
    println("save_n1_state: saved to $state_dir (c_tag=$c_tag, c_target=$c_target)")
end

"""
    save_n1_diff_bounds(state_dir)

Save diff bounds (computed once, shared across all c_target pairs) to disk.
Also writes an auxiliary `n1_preact_bounds.bin` when the N1 per-layer
pre-activation bounds are available — these are required by the
--adv_std_zono_bounds Source B pass in Phase 2 but are optional (legacy
state dirs without this file simply disable Source B).
"""
function save_n1_diff_bounds(state_dir::String)
    mkpath(state_dir)
    serialize(joinpath(state_dir, "diff_bounds.bin"),
             (relu_diff_up_bounds, relu_diff_down_bounds))
    if !isempty(n1_preact_up_bounds)
        serialize(joinpath(state_dir, "n1_preact_bounds.bin"),
                  (n1_preact_up_bounds, n1_preact_down_bounds))
        println("save_n1_diff_bounds: saved $(length(relu_diff_up_bounds)) ReLU layers " *
                "(+ n1_preact_bounds.bin) to $state_dir")
    else
        println("save_n1_diff_bounds: saved $(length(relu_diff_up_bounds)) ReLU layers to $state_dir")
    end
end

"""
    load_n1_state(state_dir, c_tag, c_target)

Load N1 solver state for one (c_tag, c_target) pair from disk.
Returns (n1_var_names, n1_var_values, n1_layers_info, n1_vbasis, n1_pseudocosts).
`n1_pseudocosts` is `nothing` for legacy saved states that pre-date the
pseudo-cost feature.
"""
function load_n1_state(state_dir::String, c_tag::Int, c_target::Int)
    tag = "$(c_tag)_$(c_target)"
    n1_var_names, n1_var_values = deserialize(joinpath(state_dir, "n1_vars_$tag.bin"))
    n1_layers_info = deserialize(joinpath(state_dir, "n1_layers_$tag.bin"))
    vbasis_path = joinpath(state_dir, "n1_vbasis_$tag.bin")
    n1_vbasis = isfile(vbasis_path) ? deserialize(vbasis_path) : Dict{String, Int}()
    pseudocost_path = joinpath(state_dir, "n1_pseudocosts_$tag.bin")
    n1_pseudocosts = isfile(pseudocost_path) ? deserialize(pseudocost_path) : nothing
    pc_msg = n1_pseudocosts === nothing ? "no pseudocost file" : "$(length(n1_pseudocosts)) pseudocosts"
    vb_msg = isfile(vbasis_path) ? "$(length(n1_vbasis)) basis" : "no vbasis file"
    println("load_n1_state: loaded from $state_dir (c_tag=$c_tag, c_target=$c_target, $(length(n1_var_names)) vars, $(length(n1_layers_info)) neurons, $vb_msg, $pc_msg)")
    return n1_var_names, n1_var_values, n1_layers_info, n1_vbasis, n1_pseudocosts
end

"""
    load_n1_diff_bounds!(state_dir)

Load diff bounds from disk and set the globals. Also loads the optional
`n1_preact_bounds.bin` used by --adv_std_zono_bounds Source B; if missing
(legacy state dir) the n1_preact globals are cleared and Source B will
detect the absence and skip with a warning.
"""
function load_n1_diff_bounds!(state_dir::String)
    global relu_diff_up_bounds, relu_diff_down_bounds
    global n1_preact_up_bounds, n1_preact_down_bounds
    relu_diff_up_bounds, relu_diff_down_bounds = deserialize(joinpath(state_dir, "diff_bounds.bin"))
    preact_path = joinpath(state_dir, "n1_preact_bounds.bin")
    if isfile(preact_path)
        n1_preact_up_bounds, n1_preact_down_bounds = deserialize(preact_path)
        println("load_n1_diff_bounds!: loaded $(length(relu_diff_up_bounds)) ReLU layers " *
                "(+ n1_preact_bounds.bin) from $state_dir")
    else
        n1_preact_up_bounds   = Array{Float64}[]
        n1_preact_down_bounds = Array{Float64}[]
        println("load_n1_diff_bounds!: loaded $(length(relu_diff_up_bounds)) ReLU layers from $state_dir " *
                "(no n1_preact_bounds.bin — Source B will be disabled for --adv_std_zono_bounds)")
    end
end
