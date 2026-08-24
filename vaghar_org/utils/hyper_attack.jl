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

# ── N1 state persistence (advanced_standard_n1 → advanced_standard_n2) ──

"""
    save_n1_state(state_dir, c_tag, c_target, n1_var_names, n1_var_values,
                  n1_layers_info)

Save N1 solver state for one (c_tag, c_target) pair to disk.
Files: n1_vars_{c_tag}_{c_target}.bin, n1_layers_{c_tag}_{c_target}.bin.
"""
function save_n1_state(state_dir::String, c_tag::Int, c_target::Int,
                       n1_var_names::Vector{String}, n1_var_values::Vector{Float64},
                       n1_layers_info::Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}})
    mkpath(state_dir)
    tag = "$(c_tag)_$(c_target)"
    serialize(joinpath(state_dir, "n1_vars_$tag.bin"), (n1_var_names, n1_var_values))
    serialize(joinpath(state_dir, "n1_layers_$tag.bin"), n1_layers_info)
    println("save_n1_state: saved to $state_dir (c_tag=$c_tag, c_target=$c_target)")
end

# Filename suffix for the diff-bounds cache: "" for the default zonotope mode,
# "_arithTransfer" for --arithmetic_transfer_bounds true, so the two modes never
# overwrite or silently reuse each other's files in a shared state dir.
diff_bounds_suffix(arithmetic::Bool) = arithmetic ? "_arithTransfer" : ""

"""
    save_n1_diff_bounds(state_dir; arithmetic=false)

Save diff bounds (computed once, shared across all c_target pairs) to disk.
Also writes an auxiliary `n1_preact_bounds.bin` when the N1 per-layer
pre-activation bounds are available — these are required by the
--adv_std_zono_bounds Source B pass in Phase 2 but are optional (legacy
state dirs without this file simply disable Source B).
With `arithmetic=true` (--arithmetic_transfer_bounds) both files carry the
`_arithTransfer` suffix so the two modes coexist in one state dir.
"""
function save_n1_diff_bounds(state_dir::String; arithmetic::Bool=false)
    mkpath(state_dir)
    sfx = diff_bounds_suffix(arithmetic)
    serialize(joinpath(state_dir, "diff_bounds$(sfx).bin"),
             (relu_diff_up_bounds, relu_diff_down_bounds))
    if !isempty(n1_preact_up_bounds)
        serialize(joinpath(state_dir, "n1_preact_bounds$(sfx).bin"),
                  (n1_preact_up_bounds, n1_preact_down_bounds))
        println("save_n1_diff_bounds: saved $(length(relu_diff_up_bounds)) ReLU layers " *
                "(+ n1_preact_bounds$(sfx).bin) to $state_dir")
    else
        println("save_n1_diff_bounds: saved $(length(relu_diff_up_bounds)) ReLU layers to $state_dir")
    end
end

"""
    load_n1_state(state_dir, c_tag, c_target)

Load N1 solver state for one (c_tag, c_target) pair from disk.
Returns (n1_var_names, n1_var_values, n1_layers_info).
"""
function load_n1_state(state_dir::String, c_tag::Int, c_target::Int)
    tag = "$(c_tag)_$(c_target)"
    vars_path   = joinpath(state_dir, "n1_vars_$tag.bin")
    layers_path = joinpath(state_dir, "n1_layers_$tag.bin")
    # n1_vars and n1_layers are consumed by every advstd technique (bound
    # tightening, varHint, relax) — always mandatory.
    isfile(vars_path)   || error("load_n1_state: mandatory file missing: $vars_path — re-run Phase 1 for c_tag=$c_tag, c_target=$c_target")
    isfile(layers_path) || error("load_n1_state: mandatory file missing: $layers_path — re-run Phase 1 for c_tag=$c_tag, c_target=$c_target")
    n1_var_names, n1_var_values = deserialize(vars_path)
    n1_layers_info = deserialize(layers_path)
    println("load_n1_state: loaded from $state_dir (c_tag=$c_tag, c_target=$c_target, $(length(n1_var_names)) vars, $(length(n1_layers_info)) neurons)")
    return n1_var_names, n1_var_values, n1_layers_info
end

"""
    load_n1_diff_bounds!(state_dir; require_preact::Bool=false, arithmetic::Bool=false)

Load diff bounds from disk and set the globals. `diff_bounds.bin` is always
required (crash on miss). `n1_preact_bounds.bin` is required iff `require_preact`
is true (pass `true` when `--adv_std_zono_bounds` is active, since Source B
depends on it); otherwise its absence is tolerated and the n1_preact globals
are cleared. With `arithmetic=true` (--arithmetic_transfer_bounds) both files
carry the `_arithTransfer` suffix, i.e. the bounds Phase 1 computed with plain
interval arithmetic are loaded instead of the zonotope ones.
"""
function load_n1_diff_bounds!(state_dir::String; require_preact::Bool=false, arithmetic::Bool=false)
    global relu_diff_up_bounds, relu_diff_down_bounds
    global n1_preact_up_bounds, n1_preact_down_bounds
    sfx = diff_bounds_suffix(arithmetic)
    diff_path = joinpath(state_dir, "diff_bounds$(sfx).bin")
    if !isfile(diff_path)
        other_path = joinpath(state_dir, "diff_bounds$(diff_bounds_suffix(!arithmetic)).bin")
        hint = isfile(other_path) ?
            " (found $(basename(other_path)): this state dir was built with --arithmetic_transfer_bounds $(!arithmetic) — pass the same value here, or re-run Phase 1 with --arithmetic_transfer_bounds $(arithmetic))" : ""
        error("load_n1_diff_bounds!: mandatory file missing: $diff_path — re-run Phase 1 on this state dir$hint")
    end
    relu_diff_up_bounds, relu_diff_down_bounds = deserialize(diff_path)
    preact_path = joinpath(state_dir, "n1_preact_bounds$(sfx).bin")
    if isfile(preact_path)
        n1_preact_up_bounds, n1_preact_down_bounds = deserialize(preact_path)
        println("load_n1_diff_bounds!: loaded $(length(relu_diff_up_bounds)) ReLU layers " *
                "(+ $(basename(preact_path))) from $state_dir")
    elseif require_preact
        error("load_n1_diff_bounds!: mandatory file missing: $preact_path — required when --adv_std_zono_bounds is active (Source B). Re-run Phase 1 with zono-bound extraction, or disable --adv_std_zono_bounds.")
    else
        n1_preact_up_bounds   = Array{Float64}[]
        n1_preact_down_bounds = Array{Float64}[]
        println("load_n1_diff_bounds!: loaded $(length(relu_diff_up_bounds)) ReLU layers from $state_dir " *
                "(no $(basename(preact_path)); --adv_std_zono_bounds disabled, so not required)")
    end
end
