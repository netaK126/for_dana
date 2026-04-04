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
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr17.p"
        "--hypers_dir_path"
        help = "hypers model path"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist _hyper_is_17/"#"/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/"
        "--ctag", "-c" 
        help = "ctag, source class"
        arg_type = String
        required = false
        default = "2"
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "2"
        "--timout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 500
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
        default = "0.008"
        "--model_path_vaghar_results"
        help = "model_path_vaghar_results"
        arg_type = String
        required = false
        default = "/root/Downloads/vaghar_org/results/63902078677641_4x10_linf_0.05_ctag0_itr17.txt"#"/root/Downloads/vaghar_as_should_be_originally_no_c_target/results_max/4x10_model_itr17.p_linf_0.05_NoCtarget_RegularVaghar_Itr17.txt"#"/root/Downloads/vaghar_org/results/63902082439234_4x10_linf_0.05_ctag0_itr18.txt"#"/root/Downloads/vaghar_org/results/63904068084000_4x10_linf_0.02_ctag0_itr18_cTag1.txt"
        "--c_tag_mode"
        help = "c_tag_mode"
        arg_type = Bool
        required = false
        default = true
        
    end
    return parse_args(s)
end

function save_results_neta(results_path, model_name, results_str, type_of_problem,c_tag,c_tag_mode)
    global separation_index
    ct_str="NocTragetVersion"
    if c_tag_mode
        ct_str="cTargetVersion"
    end
    file = open(results_path*model_name *"_"*type_of_problem*"DeltaDiff_itr18and17_ctag"*string(c_tag)*"_"*ct_str*"_RemovingHyperPerturbationEncoding_HyperIs17"*".txt", "w")
    write(file, results_str)
    close(file)
end

function get_delta1_vaghar(model_path_vaghar_results, line_index)
    open(model_path_vaghar_results, "r") do io
        current_line_number = 0
        requested_line = ""
        while !eof(io)
            line_content = readline(io)
            println(line_content)
            c_target = Base.split(line_content, ',')[2]
            if parse(Int, c_target) == parse(Int, string(line_index))-1
                requested_line = line_content
                println("FOUND")
            end
            current_line_number += 1
        end
        if requested_line==""
            println("Error with requested_line")
            exit()
        end
        parsed_tokens = Base.split(requested_line, ',')
        return parse(Float64, parsed_tokens[end-1])
    end
end

function find_var(var_name, m)
    v_ref = variable_by_name(m, var_name)

    if v_ref !== nothing && has_values(m)
        # Using the explicit module prefix to avoid the "String" error
        val = JuMP.value(v_ref) 
        var_name = JuMP.name(v_ref)
        println("Variable $var_name found. Optimization result: $val")
        exit()
    else
        println("Variable not found or no solution exists.")
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
    c_tag_mode = args["c_tag_mode"]
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
            if c_tag_mode
                if c_target==c_tag
                    continue
                end
            else
                if c_target!=c_tag
                    continue
                end
            end
            println("c_tag = "* string(c_tag))
            println("c_target = "* string(c_target))
            delta1_vaghar = get_delta1_vaghar(model_path_vaghar_results, c_target)
            println("delta1_vaghar")
            println(string(delta1_vaghar))
            if delta1_vaghar<=0
                println("delta1_vaghgar is negative")
                continue
            end
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
                mip_set_delta_diff_property_neta(m, d,delta1_vaghar, c_tag, c_target, c_tag_mode)
                set_optimizer(m, optimizer)
                mip_set_attr(m, d, timout)
                MOI.set(m, Gurobi.CallbackFunction(), my_callback)
                println("Run: optimize.")
                optimize!(m)
                try
                    mip_log(m, d)
                    results.str = update_results_str(results.str, c_tag, d, c_target)
                catch e
                    results.str = results.str * "Couldnt find Delta_diff for c_tag="*str(c_tag)*", c_target="*str(c_target)*"\n"
                end     

                find_var("conf2_p",m)
                # exit()
                global network_version
                global diff_
                diff_  = []
                save_results_neta(results_path, model_name, results.str, problem_type_str*"_"*perturbation*"_"*args["perturbation_size"],c_tag,c_tag_mode)
            end
        end
    end

    
end

main()
