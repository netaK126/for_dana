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
        default = "3x10"
        "--model_path", "-m"
        help = "model name"
        arg_type = String
        required = false
        default = "./models/3x10/model.p"
        "--perturbation", "-p"
        help = "perturbation type: occ, patch, brightness, linf, contrast, translation, rotation, or max"
        arg_type = String
        required = false
        default = "brightness"
        "--perturbation_size", "-s"
        help = "occ: i,j,width , patch: eps,i,j,width, brightness: eps, linf: eps, contrast: eps, translation: tx,ty, rotation: angle"
        arg_type = String
        required = false
        default = "0.1"
        "--ctag", "-c"
        help = "ctag, source class"
        arg_type = Int
        required = false
        default = 1
        "--timout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 10800#500
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "2"
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "/root/Downloads/code/results/"
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true
    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    dataset = args["dataset"]
    model_name = "10x10"#args["model_name"]
    perturbation = "diff"
    perturbation_size = 0
    c_tag = args["ctag"]
    c_targets = parse_numbers_to_Int64(args["ct"])
    results_path = args["output_dir"]
    timout = args["timout"]
    w, h, k, c = get_dataset_params( dataset )
    opt_version = false
    weight_addition = 0.000002#0.01f0#0.0001f0

    token_signature = string(now().instant.periods.value)
    model_path_nn1_list = ["/root/Downloads/code/10x10_models/19/model.p"]#["/root/Downloads/vhagar_21_10_24/model/3x10/20/model.p", "/root/Downloads/vhagar_21_10_24/model/3x10/40/model.p"]
    model_path_nn2 = "/root/Downloads/code/10x10_models/19/model_modified3.p"
    problem_type_str_list = ["Max", "Min"]
    encode_deps_vhagar = [false, true]
    encode_deps_neta = [false]
    for model_path_nn1 in model_path_nn1_list
        # itr = Base.split(model_path_nn1,"/")[end-1]
        itr="19Modified3"
        weight_addition_str = string(weight_addition)
        total_noise = nothing
        if opt_version
            print("OPT")
            nn, nn1, nn2 = get_nn_variables_opt(model_path_nn1, model_name, w, h, k, c, dataset, weight_addition)
        else
            print("NOT OPT")
            nn, _ = get_nn(model_path_nn1, model_name, w, h, k, c, dataset, 0)
            nn1, _ = get_nn(model_path_nn2, model_name, w, h, k, c, dataset, 0)
            # nn2, total_noise = get_nn(model_path_nn1, model_name, w, h, k, c, dataset, weight_addition)
            nn2,_=get_nn(model_path_nn2, model_name, w, h, k, c, dataset, 0)
        end
        for is_encode_deps_neta in encode_deps_neta
            for is_deps_vaghar in encode_deps_vhagar
                for problem_type_str in problem_type_str_list
                    results_str = ""
                    for c_target in c_targets

                        optimizer = Gurobi.Optimizer
                        d = Dict()
                        d[:TargetIndex] = get_target_indexes(c_target, c)
                        d[:SourceIndex] = get_target_indexes(c_tag, c)
                        d[:suboptimal_solution] = 0
                        d[:suboptimal_time] = 0
                        mip_reset()
                        bounds_time = @elapsed begin
                            merge!(d, get_model(w, h, k, perturbation, perturbation_size, nn, nn1, nn2, zeros(Float64, 1, w, h, k), optimizer,
                            get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM, opt_version))
                        end
                        d[:bounds_time] = bounds_time
                        m = d[:Model]
                        if is_deps_vaghar || is_encode_deps_neta
                            perturbation_dependencies(m, nn1, perturbation, perturbation_size, w, h, k, is_encode_deps_neta)
                        end
                        mip_set_delta_property(m, perturbation, d, problem_type_str)
                        set_optimizer(m, optimizer)
                        mip_set_attr(m, perturbation, d, timout)
                        MOI.set(m, Gurobi.CallbackFunction(), my_callback)
                        optimize!(m)
                        mip_log(m, d)
        #                 results_str = results_str*"c_tag = "*string(c_tag-1)*", lower_bound = "*string(round(d[:incumbent_obj], digits=3))*", upper_bound = "*
        #                                         string(round(d[:best_bound], digits=3))*", solve_time = "*string(d[:solve_time])*"\n"
                        results_str = results_str*"c_tag = "*string(c_tag-1)*", lower_bound = "*string(d[:incumbent_obj])*", upper_bound = "*
                                                string(d[:best_bound])*", solve_time = "*string(d[:solve_time])*"\n"
                        
                        save_results(results_path, model_name, perturbation, results_str, d, nn1, c_tag-1, w, h, k,problem_type_str,itr, opt_version, weight_addition_str, is_deps_vaghar, is_encode_deps_neta)
        #                 println(d[:v_output_nn2]-d[:v_output_nn1])
        #                 println(d[:v_output_nn1])
        #                 println(typeof(d[:v_output_nn1][1]))
        #                 exit()
                    end
                end
            end
        end
    end
end

main()