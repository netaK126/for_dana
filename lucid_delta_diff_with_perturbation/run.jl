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
include("utils/MIPVerify.jl/src/logging.jl")
include("utils/MIPVerify.jl/src/models.jl")
include("utils/MIPVerify.jl/src/utils.jl")
include("utils/perturbation_models.jl")
include("utils/help_functions.jl")
include("utils/datasets.jl")
include("utils/models.jl")
include("utils/mip.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dataset", "-d"
        help = "twitter, crypto, adult, or credit"
        arg_type = String
        required = false
        default = "mnist"
        "--model_name", "-n"
        help = ""
        arg_type = String
        required = false
        default = "4x10"
        "--model_path", "-m"
        help = "model path"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr18.p"
        "--hypers_dir_path"
        help = "hypers model path"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/"
        "--ctag", "-c" 
        help = "ctag, source class"
        arg_type = String
        required = false
        default = "1"
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "1,2,3,4,5,6,7,8,9,10"
        "--timout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 2000
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/results/"
        "--deps"
        help = "is deps"
        arg_type = Int
        required = false
        default = 1
        "--me"
        help = "me_th"
        arg_type = Float64
        required = false
        default = 0.01
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true
        "--image_mode"
        help = "image_mode"
        arg_type = Bool
        required = false
        default = true
        "--perturbation", "-p" #not needed
        help = "perturbation type: occ, patch, brightness, linf, contrast, translation, rotation, or max"
        arg_type = String
        required = false
        default = "linf"
        "--perturbation_size", "-s" #not needed
        help = "occ: i,j,width , patch: eps,i,j,width, brightness: eps, linf: eps, contrast: eps, translation: tx,ty, rotation: angle"
        arg_type = String
        required = false
        default = "0.05"
        "--model_path_vaghar_results"
        help = "model_path_vaghar_results"
        arg_type = String
        required = false
        default = "/root/Downloads/vaghar_org/results/63902082439234_4x10_linf_0.05_ctag0_itr18.txt"
        
    end
    return parse_args(s)
end

function save_results_neta(results_path, model_name, results_str, type_of_problem,c_tag)
    global separation_index
    file = open(results_path*model_name *"_"*type_of_problem*"DeltaDiff_itr18and18_cTargetVersion"*".txt", "w")
    write(file, results_str)
    close(file)
end

function get_delta1_vaghar(model_path_vaghar_results, line_index)
    open(model_path_vaghar_results, "r") do io
        current_line_number = 0
        requested_line = ""
        while !eof(io)
            current_line_number += 1
            line_content = readline(io)
            c_target = Base.split(line_content, ',')[2]
            if c_target == string(line_index)
                requested_line = line_content
            end
        end
        if requested_line==""
            println("Error with requested_line")
            exit()
        end
        parsed_tokens = Base.split(requested_line, ',')
        return parse(Float64, parsed_tokens[end-1])
    end
end

function main()
    # 18 is hyper nn
    # 17 is regular nn
    args = parse_commandline()
    dataset = args["dataset"]
    model_name = args["model_name"]
    model_path_nn = args["model_path"]
    perturbation = args["perturbation"]
    perturbation_size = parse_numbers_to_Float64(args["perturbation_size"])
    model_path_vaghar_results = args["model_path_vaghar_results"]
    hypers_dir_path = args["hypers_dir_path"]
    c_targets = parse_numbers_to_Int64(args["ct"])
    results_path = args["output_dir"]
    timout = args["timout"]
    is_deps = args["deps"]
    image_mode = args["image_mode"]
    print("image_mode = ")
    println(image_mode)
    global me_th
    me_th = args["me"]
    running_type_list = ["noLucid"]
    dim, c = get_dataset_params( dataset )
    c_tag_list = parse_numbers_to_Int64(args["ctag"])
    results.str = ""
    for c_tag in c_tag_list
        for c_target in c_targets
            if c_target==c_tag
                continue
            end
            delta1_vaghar = get_delta1_vaghar(model_path_vaghar_results, c_target)
            println("delta1_vaghar")
            println(string(delta1_vaghar))
            nn,is_conv = get_nn(model_path_nn, model_name, dim, c, dataset)
            nn_hyper = get_nn_hyper(model_path_nn, model_name, dim, c, dataset, hypers_dir_path, is_deps)
            for problem_type_str in running_type_list
                if !occursin("with",problem_type_str)
                    me_th = 0
                end
                global all_bounds_of_original
                global all_bounds_of_perturbation
                all_bounds_of_original = []
                all_bounds_of_perturbation = []
                optimizer = Gurobi.Optimizer
                d= Dict()
                d[:SourceIndex] = get_target_indexes(c_tag, c)
                mip_reset()
                println("Run: computing bounds.")
                dummy_input = zeros(Float64, 1,1,1,dim)
                bounds_time = @elapsed begin
                    merge!(d, get_model(perturbation, perturbation_size,nn, nn_hyper, dummy_input, optimizer,
                    get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
                end
                d[:bounds_time] = bounds_time
                m = d[:Model]

                # mip_set_delta_diff_propery(m, d, c_tag)
                mip_set_delta_diff_property_neta(m, d,delta1_vaghar, c_tag, c_target)
                set_optimizer(m, optimizer)
                mip_set_attr(m, d, timout)
                MOI.set(m, Gurobi.CallbackFunction(), my_callback)
                println("Run: optimize.")
                optimize!(m)
                mip_log(m, d)
                results.str = update_results_str(results.str, c_tag, d, c_target)

                global network_version
                global diff_
                diff_  = []
                save_results_neta(results_path, model_name, results.str, problem_type_str*"_",c_tag)
            end
        end
    end
    println("---------------------------")
    
end

main()
