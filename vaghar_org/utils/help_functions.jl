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

function save_results(results_path, model_name, perturbation, perturbation_size, results_str, d, nn, ss, tt, w_, h_, k_,name_to_save,token_signature)

    file = open(results_path * token_signature*"_"*model_name * "_" * perturbation * "_" * create_perturbation_string(perturbation_size)*"_ctag"*string(ss)*"_"*name_to_save*".txt", "w")
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
    return results *
        "c_source=" * string(c_tag-1) * "," *
        "c_target=" * string(c_target-1) * "," *
        "lower_bound=" * string(d[:incumbent_obj]) * "," *
        "upper_bound=" * string(d[:best_bound]) * "," *
        "optimization_time=" * string(d[:solve_time]) * "," *
        "hyper_attack_time=" * string(hyper_time) * "," *
        "solve_status=" * string(d[:SolveStatus]) * "\n"
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