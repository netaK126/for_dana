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
        default = "twitter"
        "--model_name", "-n"
        help = ""
        arg_type = String
        required = false
        default = "2x10"
        "--model_path", "-m"
        help = "model name"
        arg_type = String
        required = false
        default = "./models/twitter.p"
        "--hypers_dir_path"
        help = "hypers model path"
        arg_type = String
        required = false
        default = "./models/"
        "--ctag", "-c"
        help = "ctag, source class"
        arg_type = Int
        required = false
        default = 1
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
        default = "/root/Downloads/lucid/results_mnist_ensamble_bagging/"
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
        default = false

    end
    return parse_args(s)
end

function save_results(results_path, model_name, results_str, type_of_problem,c_tag)
    global separation_index
    file = open(results_path*model_name *"_"*type_of_problem*"ObjectiveDiff_differentNetworks"*"_mnist_Bagging_itr19_SplitingIndex"*string(separation_index)*"_ctagHalfMilpOptimizing"*string(c_tag)*".txt", "w")
    write(file, results_str)
    close(file)
end

function main()
    args = parse_commandline()
    dataset = args["dataset"]
    model_name = "3x10"#args["model_name"]
    # model_path = args["model_path"]
    # hyper_model_path = "/root/Downloads/lucid/models_mnist1/"
    model_path_list = [ #"/root/Downloads/lucid/models_mnist4/mnist4.p",
                        #"/root/Downloads/lucid/models_mnist5/mnist5.p",
                        #"/root/Downloads/lucid/models_mnist6/mnist6.p",
                        #"/root/Downloads/lucid/models_mnist7/mnist7.p",
                        "/root/Downloads/lucid/models_mnist_ensamble/mnist2.p"]
    hypers_dir_path = "/root/Downloads/lucid/models_mnist_ensamble/" #args["hypers_dir_path"]
    # c_tag = args["ctag"]
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
    c_tag_list = [1,2]
    # create_modified_nn(model_path, model_name, dim, c, dataset)
    # exit()
    separation_index_list = [2]
    for model_path_nn in model_path_list
        # mnist_str = replace(Base.split(model_path_nn, "/")[end-1], "models_" => "")
        for i in separation_index_list
            global separation_index = i
            for c_tag in c_tag_list
                nn,is_conv = get_nn(model_path_nn, model_name, dim, c, dataset)
                # nn_preffix,is_conv = get_nn_preffix(model_path_nn, model_name, dim, c, dataset)
                # nn_suffix,is_conv = get_nn_suffix(model_path_nn, model_name, dim, c, dataset)
                nn_hyper = get_nn_hyper(model_path_nn, model_name, dim, c, dataset, hypers_dir_path, is_deps)
                # nn_hyper_preffix = get_nn_hyper_preffix(model_path_nn, model_name, dim, c, dataset, hypers_dir_path, is_deps)
                # nn_hyper_suffix = get_nn_hyper_suffix(model_path_nn, model_name, dim, c, dataset, hypers_dir_path, is_deps)

                for problem_type_str in running_type_list
                    global activate_lucid
                    if occursin("with",problem_type_str)
                        activate_lucid=true
                    else
                        activate_lucid=false
                    end
                    for c_target in c_targets
                        start_time = time()
                        global all_bounds_of_original
                        all_bounds_of_original = []
                        
                        ############################ preffix ############################
                        global I_z_prev_up = []
                        global I_z_prev_down = []
                        global print_m = 2
                        optimizer = Gurobi.Optimizer
                        d = Dict()
                        d[:TargetIndex] = get_target_indexes(c_target, c)
                        d[:SourceIndex] = get_target_indexes(c_tag, c)
                        mip_reset()
                        println("Run: computing bounds.")
                        dummy_input = zeros(Float64, 1,1,1,dim)
                        if is_conv
                            dummy_input = zeros(Float64, 1,dim,1,1)
                        end
                        bounds_time = @elapsed begin
                            merge!(d, get_model(nn, nn_hyper, dummy_input, optimizer,
                            get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
                        end
                        d[:bounds_time] = bounds_time
                        m = d[:Model]
                        global I_z_prev_up
                        global I_z_prev_down
                        diff_max_upper_bound = I_z_prev_up[c_target] - maximum(I_z_prev_down[[i for i in eachindex(I_z_prev_up) if i != c_target]])
                        diff_max_lower_bound = I_z_prev_down[c_target] - maximum(I_z_prev_up[[i for i in eachindex(I_z_prev_up) if i != c_target]])
                        println(time() - start_time)
                        println(diff_max_upper_bound)
                        println(diff_max_lower_bound)
                        println("v_out for regular nn is")
                        println(d[:v_out])
                        println("v_out for hyper nn is")
                        # println([JuMP.name(v) for v in d[:v_out_p]])
                        # println(d[:v_out_p])
                        # exit()
                        # println("--------- Results ---------")
                        mip_set_delta_diff_propery(m, d, c_tag)
                        set_optimizer(m, optimizer)
                        mip_set_attr(m, d, timout)
                        MOI.set(m, Gurobi.CallbackFunction(), my_callback)
                        println("Run: optimize.")
                        optimize!(m)
                        mip_log(m, d)
                        stop_time = time()
                        elapsed_time = stop_time - start_time
                        d[:solve_time] = elapsed_time
                        results.str = update_results_str(results.str, c_tag, c_target, d)
                        results_str="c:"*string(c_tag-1)*" ,t:"*string(c_target-1)*" ,bound:"*string(d[:best_bound])*" ,solve time:"*string(d[:solve_time])
                        println("c:"*string(c_tag-1)*" ,t:"*string(c_target-1)*" ,bound:"*string(d[:best_bound])*" ,solve time:"*string(d[:solve_time]))
                        # results_str = "c_tag = "*string(c_tag-1)*", lower_bound = "*string(d[:incumbent_obj])*", upper_bound = "*string(d[:best_bound])*", solve_time = "*string(d[:solve_time])*"\n"
                        # println(results_str)
                        save_results(results_path, model_name, results_str, problem_type_str*"_"*Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"),c_tag)
                        exit()
                        ############################ suffix ############################
                        println("Starting suffix encoding")
                        global all_bounds_of_original
                        all_bounds_of_original = []
                        optimizer = Gurobi.Optimizer
                        d_suffix = Dict()
                        d_suffix[:Model] = d[:Model]
                        d_suffix[:TargetIndex] = get_target_indexes(c_target, c)
                        d_suffix[:SourceIndex] = get_target_indexes(c_tag, c)
                        mip_reset()
                        println("Run: computing bounds.")
                        dummy_input = zeros(Float64,1,1,1,c)
                        bounds_time = @elapsed begin
                            merge!(d_suffix, get_model_suffix(nn_suffix, nn_hyper_suffix, dummy_input, optimizer,
                            get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM,d_suffix[:Model]))
                        end
                        d_suffix[:bounds_time] = bounds_time
                        m = d_suffix[:Model]
                        println(I_z_prev_up)
                        println(I_z_prev_down)

                        stop_time = time()
                        elapsed_time = stop_time - start_time
                        println("")
                        println(elapsed_time)
                        diff_max_upper_bound = I_z_prev_up[c_target] - maximum(I_z_prev_down[[i for i in eachindex(I_z_prev_up) if i != c_target]])
                        diff_max_lower_bound = I_z_prev_down[c_target] - maximum(I_z_prev_up[[i for i in eachindex(I_z_prev_up) if i != c_target]])
                        println(diff_max_upper_bound)
                        println(diff_max_lower_bound)
                        # exit()

                        # d[:incumbent_obj] = diff_max_lower_bound
                        # d[:best_bound] = diff_max_upper_bound
                        # results.str = ""
                        # results.str = update_results_str(results.str, c_tag, c_target, d)
                        # save_results(results_path, model_name, results.str, problem_type_str*"_"*Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"),c_target)
                        # continue
                        # mip_set_delta_property(m, d)
                        mip_set_delta_diff_propery(m, d_suffix, c_tag)
                        set_optimizer(m, optimizer)
                        mip_set_attr(m, d_suffix, timout)
                        MOI.set(m, Gurobi.CallbackFunction(), my_callback)
                        # println("v_out for regular nn_suffix is")
                        # println(d_suffix[:v_out])
                        # println("v_out for hyper nn_suffix is")
                        # println([JuMP.name(v) for v in d_suffix[:v_out_p]])
                        # exit()
                        println("Run: optimize.")
                        optimize!(m)
                        mip_log(m, d_suffix)
                        stop_time = time()
                        elapsed_time = stop_time - start_time
                        d_suffix[:solve_time] = elapsed_time
                        results.str = update_results_str(results.str, c_tag, c_target, d_suffix)

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
                        println("--------- Results ---------")
                        println(I_z_prev_up)
                        println(I_z_prev_down)
                        println("c:"*string(c_tag-1)*" ,t:"*string(c_target-1)*" ,bound:"*string(d_suffix[:best_bound])*" ,solve time:"*string(d_suffix[:solve_time]))
                        results_str = "c_tag = "*string(c_tag-1)*", lower_bound = "*string(d_suffix[:incumbent_obj])*", upper_bound = "*string(d_suffix[:best_bound])*", solve_time = "*string(d_suffix[:solve_time])*"\n"
                        save_results(results_path, model_name, results_str, problem_type_str*"_"*Dates.format(now(), "yyyy-mm-dd_HH-MM-SS"),c_tag)
                    end
                end
            end
            println("---------------------------")
        end
    end
    
end

main()
