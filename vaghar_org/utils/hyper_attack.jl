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
                  n1_layers_info, n1_vbasis)

Save N1 solver state for one (c_tag, c_target) pair to disk.
Files: n1_vars_{c_tag}_{c_target}.bin, n1_layers_{c_tag}_{c_target}.bin,
       n1_vbasis_{c_tag}_{c_target}.bin.
"""
function save_n1_state(state_dir::String, c_tag::Int, c_target::Int,
                       n1_var_names::Vector{String}, n1_var_values::Vector{Float64},
                       n1_layers_info::Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}},
                       n1_vbasis::Dict{String, Int})
    mkpath(state_dir)
    tag = "$(c_tag)_$(c_target)"
    serialize(joinpath(state_dir, "n1_vars_$tag.bin"), (n1_var_names, n1_var_values))
    serialize(joinpath(state_dir, "n1_layers_$tag.bin"), n1_layers_info)
    serialize(joinpath(state_dir, "n1_vbasis_$tag.bin"), n1_vbasis)
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
Returns (n1_var_names, n1_var_values, n1_layers_info, n1_vbasis).
"""
function load_n1_state(state_dir::String, c_tag::Int, c_target::Int;
                       require_vbasis::Bool=false)
    tag = "$(c_tag)_$(c_target)"
    vars_path   = joinpath(state_dir, "n1_vars_$tag.bin")
    layers_path = joinpath(state_dir, "n1_layers_$tag.bin")
    vbasis_path = joinpath(state_dir, "n1_vbasis_$tag.bin")
    # n1_vars and n1_layers are consumed by every advstd technique (Tech 2
    # bound tightening, Tech 3 varHint, Tech 4 relax) — always mandatory.
    # n1_vbasis is consumed only by LP Basis transfer (--adv_std_lp_basis=true);
    # pass require_vbasis=use_lp_basis at the call site. When LP Basis is off,
    # tolerate absence with an empty-dict fallback rather than killing a run
    # that would never have read the file anyway.
    isfile(vars_path)   || error("load_n1_state: mandatory file missing: $vars_path — re-run Phase 1 for c_tag=$c_tag, c_target=$c_target")
    isfile(layers_path) || error("load_n1_state: mandatory file missing: $layers_path — re-run Phase 1 for c_tag=$c_tag, c_target=$c_target")
    n1_var_names, n1_var_values = deserialize(vars_path)
    n1_layers_info = deserialize(layers_path)
    if isfile(vbasis_path)
        n1_vbasis = deserialize(vbasis_path)
        vb_msg = "$(length(n1_vbasis)) basis"
    elseif require_vbasis
        error("load_n1_state: mandatory file missing: $vbasis_path — required when --adv_std_lp_basis=true. Re-run Phase 1 for c_tag=$c_tag, c_target=$c_target, or disable --adv_std_lp_basis.")
    else
        n1_vbasis = Dict{String, Int}()
        vb_msg = "no vbasis file (LP Basis disabled, not required)"
    end
    println("load_n1_state: loaded from $state_dir (c_tag=$c_tag, c_target=$c_target, $(length(n1_var_names)) vars, $(length(n1_layers_info)) neurons, $vb_msg)")
    return n1_var_names, n1_var_values, n1_layers_info, n1_vbasis
end

"""
    load_n1_diff_bounds!(state_dir; require_preact::Bool=false)

Load diff bounds from disk and set the globals. `diff_bounds.bin` is always
required (crash on miss). `n1_preact_bounds.bin` is required iff `require_preact`
is true (pass `true` when `--adv_std_zono_bounds` is active, since Source B
depends on it); otherwise its absence is tolerated and the n1_preact globals
are cleared.
"""
function load_n1_diff_bounds!(state_dir::String; require_preact::Bool=false)
    global relu_diff_up_bounds, relu_diff_down_bounds
    global n1_preact_up_bounds, n1_preact_down_bounds
    diff_path = joinpath(state_dir, "diff_bounds.bin")
    isfile(diff_path) || error("load_n1_diff_bounds!: mandatory file missing: $diff_path — re-run Phase 1 on this state dir")
    relu_diff_up_bounds, relu_diff_down_bounds = deserialize(diff_path)
    preact_path = joinpath(state_dir, "n1_preact_bounds.bin")
    if isfile(preact_path)
        n1_preact_up_bounds, n1_preact_down_bounds = deserialize(preact_path)
        println("load_n1_diff_bounds!: loaded $(length(relu_diff_up_bounds)) ReLU layers " *
                "(+ n1_preact_bounds.bin) from $state_dir")
    elseif require_preact
        error("load_n1_diff_bounds!: mandatory file missing: $preact_path — required when --adv_std_zono_bounds is active (Source B). Re-run Phase 1 with zono-bound extraction, or disable --adv_std_zono_bounds.")
    else
        n1_preact_up_bounds   = Array{Float64}[]
        n1_preact_down_bounds = Array{Float64}[]
        println("load_n1_diff_bounds!: loaded $(length(relu_diff_up_bounds)) ReLU layers from $state_dir " *
                "(no n1_preact_bounds.bin; --adv_std_zono_bounds disabled, so not required)")
    end
end
