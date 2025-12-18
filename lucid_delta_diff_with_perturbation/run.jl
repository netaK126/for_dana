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
        default = "/root/Downloads/code_deprecated_active_just_for_models/models/4x10/19/model.p"
        "--hypers_dir_path"
        help = "hypers model path"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid/models_mnist/"
        "--ctag", "-c" 
        help = "ctag, source class"
        arg_type = String
        required = false
        default = "1,2,3,4,5,6,7,8,9,10"
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "2"
        "--timout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 2000
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid/results/"
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

    end
    return parse_args(s)
end

function save_results(results_path, model_name, results_str, type_of_problem,c_tag)
    global separation_index
    file = open(results_path*model_name *"_"*type_of_problem*"DeltaDiff_differentNetworks_EddedSomeContations"*"_mnist"*".txt", "w")
    write(file, results_str)
    close(file)
end

function main()
    args = parse_commandline()
    dataset = args["dataset"]
    model_name = args["model_name"]
    model_path_nn = args["model_path"]
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
        nn,is_conv = get_nn(model_path_nn, model_name, dim, c, dataset)
        nn_hyper = get_nn_hyper(model_path_nn, model_name, dim, c, dataset, hypers_dir_path, is_deps)
        # nn_hyper, _ = get_nn("/root/Downloads/code_deprecated_active_just_for_models/models/4x10/19/model.p", model_name, dim, c, dataset)
        # nn_hyper = get_nn_hyper("/root/Downloads/code_deprecated_active_just_for_models/models/4x10/19/model.p", model_name, dim, c, dataset, hypers_dir_path, is_deps)

        for problem_type_str in running_type_list
            global activate_lucid
            if occursin("with",problem_type_str)
                activate_lucid=true
            else
                activate_lucid=false
                me_th = 0
            end
            global all_bounds_of_original
            all_bounds_of_original = []
            optimizer = Gurobi.Optimizer
            d= Dict()
            d[:SourceIndex] = get_target_indexes(c_tag, c)
            mip_reset()
            println("Run: computing bounds.")
            dummy_input = zeros(Float64, 1,1,1,dim)
            bounds_time = @elapsed begin
                merge!(d, get_model(nn, nn_hyper, dummy_input, optimizer,
                get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
            end
            d[:bounds_time] = bounds_time
            m = d[:Model]
            # println(I_z_prev_up)
            # println(I_z_prev_down)
            # diff_max_upper_bound = I_z_prev_up[c_target] - maximum(I_z_prev_down[[i for i in eachindex(I_z_prev_up) if i != c_target]])
            # diff_max_lower_bound = I_z_prev_down[c_target] - maximum(I_z_prev_up[[i for i in eachindex(I_z_prev_up) if i != c_target]])
            # println(diff_max_upper_bound)
            # println(diff_max_lower_bound)

            mip_set_delta_diff_propery(m, d, c_tag)
            set_optimizer(m, optimizer)
            mip_set_attr(m, d, timout)
            MOI.set(m, Gurobi.CallbackFunction(), my_callback)
            println("Run: optimize.")
            optimize!(m)
            mip_log(m, d)
            results.str = update_results_str(results.str, c_tag, d)

            global network_version
            global upper_bound_prev
            global lower_bound_prev
            global u_for_spread
            global l_for_spread
            global diff_
            global I_u
            global I_l
            upper_bound_prev = []
            lower_bound_prev = []
            u_for_spread = []
            l_for_spread = []
            diff_  = []
            I_u = []
            I_l = []
            save_results(results_path, model_name, results.str, problem_type_str*"_",c_tag)
        end
    end
    println("---------------------------")
    
end

main()
