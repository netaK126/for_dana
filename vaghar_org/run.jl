ENV["PYTHON"]="/usr/bin/python3.8"

using Gurobi
using PyCall
using PyPlot
using Gurobi
using Images
using Printf
using Dates
using Base.Cartesian
using JuMP
using MathOptInterface
using Memento
using DocStringExtensions
using ProgressMeter
using ArgParse

np = pyimport("numpy")

include("utils/MIPVerify.jl/src/MIPVerify.jl")
const dependencies_path = joinpath(@__DIR__, "utils/MIPVerify.jl/", "deps")
@enum TighteningAlgorithm interval_arithmetic = 1 lp = 2 mip = 3
const DEFAULT_TIGHTENING_ALGORITHM = mip

include("utils/MIPVerify.jl/src/vendor/ConditionalJuMP.jl")
include("utils/MIPVerify.jl/src/net_components.jl")
include("utils/perturbation_dependencies.jl")
include("utils/MIPVerify.jl/src/logging.jl")
include("utils/MIPVerify.jl/src/models.jl")
include("utils/MIPVerify.jl/src/utils.jl")
include("utils/perturbation_models.jl")
include("utils/help_functions.jl")
include("utils/hyper_attack.jl")
include("utils/datasets.jl")
include("utils/models.jl")
include("utils/mip.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dataset", "-d"
        help = "mnist, fmnist, or cifar10"
        arg_type = String
        required = false
        default = "mnist"
        "--model_name", "-n"
        help = "3x10, 3x50, cnn0, cnn1, or cnn2"
        arg_type = String
        required = false
        default = "4x10"
        "--model_path", "-m"
        help = "model name"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr17.p"
        "--perturbation", "-p"
        help = "perturbation type: occ, patch, brightness, linf, contrast, translation, rotation, or max"
        arg_type = String
        required = false
        default = "linf"
        "--perturbation_size", "-s"
        help = "occ: i,j,width , patch: eps,i,j,width, brightness: eps, linf: eps, contrast: eps, translation: tx,ty, rotation: angle"
        arg_type = String
        required = false
        default = "0.05"
        "--ctag", "-c"
        help = "ctag, source class"
        arg_type = Int
        required = false
        default = 1
        "--timout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 4000
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "2,3,4,5,6,7,8,9"
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "./results/"
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true
        "--name_to_save"
        help = "string for results name file"
        arg_type = String
        required = false
        default = "itr17"
        "--mode"
        help = "standard or transfer"
        arg_type = String
        required = false
        default = "standard"
        "--model_path2"
        help = "path to second network N2 (transfer mode)"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr18.p"
        "--vaghar_results"
        help = "path to VHAGaR results file for delta_1 values (transfer mode)"
        arg_type = String
        required = false
        default = "/root/Downloads/vaghar_org/results/63902078677641_4x10_linf_0.05_ctag0_itr17.txt"
        "--c_tag_mode"
        help = "true: c_pert=c_tag (untargeted), false: c_pert=c_target (targeted)"
        arg_type = Bool
        required = false
        default = true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    dataset = args["dataset"]
    model_name = args["model_name"]
    model_path = args["model_path"]
    perturbation = args["perturbation"]
    name_to_save = args["name_to_save"]
    perturbation_size = parse_numbers_to_Float64(args["perturbation_size"])
    mode = args["mode"]

    if mode == "transfer"
        main_transfer(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save)
    else
        main_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save)
    end
end

function main_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save)
    c_tag_list = [1,2,3,4,5,6,7,8,9]
    for c_tag in c_tag_list
        results.str = ""
        c_targets = parse_numbers_to_Int64(args["ct"])
        results_path = args["output_dir"]
        timout = args["timout"]
        w, h, k, c = get_dataset_params( dataset )
        nn = get_nn(model_path, model_name, w, h, k, c, dataset)
        token_signature = string(now().instant.periods.value)
        for c_target in c_targets
            suboptimal_solution, suboptimal_time =  hyper_attack(dataset, c_tag, c_target, token_signature, model_name, model_path, perturbation, perturbation_size)
            optimizer = Gurobi.Optimizer
            d = Dict()
            d[:TargetIndex] = get_target_indexes(c_target, c)
            d[:SourceIndex] = get_target_indexes(c_tag, c)
            d[:suboptimal_solution] = suboptimal_solution
            d[:suboptimal_time] = suboptimal_time
            mip_reset()
            bounds_time = @elapsed begin
                merge!(d, get_model(w, h, k, perturbation, perturbation_size, nn, zeros(Float64, 1, w, h, k), optimizer,
                get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
            end
            d[:bounds_time] = bounds_time
            m = d[:Model]
            hyper_attack_hints(m, token_signature, c_tag, c_target)
            perturbation_dependencies(m, nn, perturbation, perturbation_size, w, h, k)
            mip_set_delta_property(m, perturbation, d)
            set_optimizer(m, optimizer)
            mip_set_attr(m, perturbation, d, timout)
            MOI.set(m, Gurobi.CallbackFunction(), my_callback)
            optimize!(m)
            mip_log(m, d)
            mip_reuse_bounds()
            results.str = update_results_str(results.str, c_tag, c_target, d)
            println(results_path)
            save_results(results_path, model_name, perturbation, perturbation_size, results.str, d, nn, c_tag-1, c_target-1, w, h, k,name_to_save*"_cTag"*string(c_tag),token_signature)
        end
    end
end

function main_transfer(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save)
    model_path2 = args["model_path2"]
    vaghar_results = args["vaghar_results"]
    c_tag_mode = args["c_tag_mode"]
    results_path = args["output_dir"]
    timout = args["timout"]
    c_tag_list = [1,2,3,4,5,6,7,8,9]
    c_targets = parse_numbers_to_Int64(args["ct"])
    w, h, k, c = get_dataset_params(dataset)

    println("Loading N1 from: $model_path")
    nn1 = get_nn(model_path, model_name, w, h, k, c, dataset)
    println("Loading N2 from: $model_path2")
    nn2 = get_nn(model_path2, model_name, w, h, k, c, dataset)

    K = layers_number(nn1)
    println("ReLU layers per network: $K, dependency offset: $(2*K)")

    for c_tag in c_tag_list
        results.str = ""
        for c_target in c_targets
            # In c_tag_mode: skip when c_target == c_tag (same class)
            # In c_target_mode: skip when c_target != c_tag
            if c_tag_mode
                if c_target == c_tag
                    continue
                end
            else
                if c_target != c_tag
                    continue
                end
            end

            println("=== c_tag=$c_tag, c_target=$c_target ===")
            delta_1 = get_delta1_vaghar(vaghar_results, c_target)
            println("delta_1 = $delta_1")
            if delta_1 <= 0
                println("Skipping: delta_1 <= 0")
                continue
            end

            # PGD attack for warm-start lower bound
            token_signature = string(now().instant.periods.value)
            suboptimal_solution, suboptimal_time = hyper_attack_transfer(
                dataset, c_tag, c_target, token_signature,
                model_name, model_path, model_path2,
                perturbation, perturbation_size, delta_1, c_tag_mode)
            println("Hyper attack: best_val=$suboptimal_solution, time=$suboptimal_time")

            optimizer = Gurobi.Optimizer
            mip_reset()

            println("Encoding four-network MIP...")
            d = Dict()
            bounds_time = @elapsed begin
                merge!(d, get_model_transfer(w, h, k, perturbation, perturbation_size,
                    nn1, nn2, zeros(Float64, 1, w, h, k), optimizer,
                    get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
            end
            d[:bounds_time] = bounds_time
            m = d[:Model]

            # Apply warm-start hints from PGD attack
            hyper_attack_hints(m, token_signature, c_tag, c_target)

            # Dependencies for N1: original layers 1..K ↔ perturbed layers 2K+1..3K
            perturbation_dependencies(m, nn1, perturbation, perturbation_size, w, h, k;
                                      activation_start=1, layers_offset=2*K)
            # Dependencies for N2: original layers K+1..2K ↔ perturbed layers 3K+1..4K
            perturbation_dependencies(m, nn2, perturbation, perturbation_size, w, h, k;
                                      activation_start=K+1, layers_offset=2*K)

            # Set transfer proof constraints and objective
            mip_set_transfer_property(m, d, delta_1, c_tag, c_target, c_tag_mode)
            set_optimizer(m, optimizer)
            mip_set_attr_transfer(m, timout, suboptimal_solution)
            MOI.set(m, Gurobi.CallbackFunction(), my_callback)

            println("Optimizing...")
            optimize!(m)
            mip_log(m, d)

            results.str = update_results_str(results.str, c_tag, c_target, d)
            println(results.str)
        end

        # Save results for this c_tag
        ct_str = c_tag_mode ? "cTagMode" : "cTargetMode"
        token_signature = string(now().instant.periods.value)
        file = open(results_path * token_signature * "_" * model_name * "_transfer_" *
                    perturbation * "_" * create_perturbation_string(perturbation_size) *
                    "_ctag" * string(c_tag) * "_" * ct_str * "_" * name_to_save * ".txt", "w")
        write(file, results.str)
        close(file)
    end
    println("Transfer proof computation complete.")
end

main()