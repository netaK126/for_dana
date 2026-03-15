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
        default = 10800
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "2,3,4,5,6,7,8,9,10"
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "./results/"
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true
        "--vaghar_results"
        help = "path to VHAGaR results file for delta_1 values (transfer mode)"
        arg_type = String
        required = false
        default = "/root/Downloads/vaghar_org/results/63902078677641_4x10_linf_0.05_ctag0_itr17.txt"
        "--model_path", "-m"
        help = "model name"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr17.p"
        "--model_path2"
        help = "path to second network N2 (transfer mode)"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr18.p"
    end
    return parse_args(s)
end


function get_delta1_vaghar(results_path, c_target_index)
    open(results_path, "r") do io
        requested_line = ""
        while !eof(io)
            line_content = readline(io)
            if isempty(strip(line_content))
                continue
            end
            tokens = Base.split(line_content, ',')
            if length(tokens) >= 4
                target_in_file = parse(Int, tokens[2])
                if target_in_file == c_target_index - 1
                    requested_line = line_content
                end
            end
        end
        if requested_line == ""
            println("Warning: no delta_1 found for c_target=$c_target_index in $results_path")
            return -1.0
        end
        parsed_tokens = Base.split(requested_line, ',')
        return parse(Float64, parsed_tokens[4])
    end
end

function main()
    args = parse_commandline()
    dataset = args["dataset"]
    vaghar_results = args["vaghar_results"]
    model_path2 = args["model_path2"]
    model_name = args["model_name"]
    model_path = args["model_path"]
    perturbation = args["perturbation"]
    perturbation_size = parse_numbers_to_Float64(args["perturbation_size"])
    c_tag = args["ctag"]
    c_targets = parse_numbers_to_Int64(args["ct"])
    results_path = args["output_dir"]
    timout = args["timout"]
    w, h, k, c = get_dataset_params( dataset )
    
    token_signature = string(now().instant.periods.value)
    
    for c_target in c_targets
        nn1 = get_nn(model_path, model_name, w, h, k, c, dataset)
        nn2 = get_nn(model_path2, model_name, w, h, k, c, dataset)
        optimizer = Gurobi.Optimizer
        d = Dict()
        suboptimal_solution, suboptimal_time = 0, 0
        d[:TargetIndex] = get_target_indexes(c_target, c)
        d[:SourceIndex] = get_target_indexes(c_tag, c)
        d[:suboptimal_solution] = suboptimal_solution
        d[:suboptimal_time] = suboptimal_time
        mip_reset()
        bounds_time = @elapsed begin
            merge!(d, get_model(w, h, k, perturbation, perturbation_size, nn1, nn2, zeros(Float64, 1, w, h, k), optimizer,
             get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
        end
        d[:bounds_time] = bounds_time
        m = d[:Model]
        delta_1 = get_delta1_vaghar(vaghar_results, c_target)
        mip_set_transfer_property(m, d, delta_1, c_tag, c_target, false, false, true)
        set_optimizer(m, optimizer)
        mip_set_attr(m, perturbation, d, timout)
        MOI.set(m, Gurobi.CallbackFunction(), my_callback)
        optimize!(m)
        mip_log(m, d)
        mip_reuse_bounds()
        results.str = update_results_str(results.str, c_tag, c_target, d)
        save_results(results_path, model_name, perturbation, perturbation_size, results.str, d, c_tag-1, c_target-1, w, h, k)
    end
end

main()