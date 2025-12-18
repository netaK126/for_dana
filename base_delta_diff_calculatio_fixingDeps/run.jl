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
using MathOptInterface
using Pkg

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
        "--model_name", "-n" #not needed
        help = "3x10, 3x50, cnn0, cnn1, or cnn2"
        arg_type = String
        required = false
        default = "3x10"
        "--model_path", "-m" #not needed
        help = "model name"
        arg_type = String
        required = false
        default = "./models/3x10/model.p"
        "--perturbation", "-p" #not needed
        help = "perturbation type: occ, patch, brightness, linf, contrast, translation, rotation, or max"
        arg_type = String
        required = false
        default = "brightness"
        "--perturbation_size", "-s" #not needed
        help = "occ: i,j,width , patch: eps,i,j,width, brightness: eps, linf: eps, contrast: eps, translation: tx,ty, rotation: angle"
        arg_type = String
        required = false
        default = "0.05"
        "--ctag", "-c" #not needed
        help = "ctag, source class"
        arg_type = String
        required = false
        default = "1,2,3,4,5,6,7,8,9,10"
        "--timout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 2000
        "--ct", "-t"
        help = "target classes" #not needed
        arg_type = String
        required = false
        default = "2,3,4,5,6,7,8,9,10"
        "--folder_path"
        help = "a folder with all the models i want to verify"
        arg_type = String
        required = false
        default = "/root/Downloads/code_deprecated_active_just_for_models/models/4x10_GeminiDistilation/"
        "--model_path_org"
        help = "the path to the original model i want to check the delta_diff with"
        arg_type = String
        required = false
        default = "/root/Downloads/code_deprecated_active_just_for_models/models/4x10/19/model.p"
        "--model_path_second"
        help = "the path to the original model i want to check the delta_diff with"
        arg_type = String
        required = false
        default = "/root/Downloads/code_deprecated_active_just_for_models/models/4x10_distilation/EC_and_conf_alphaVal_alphaVal0.3_model.p"
        "--model_type_1"
        help = "3x10, 5x10..."
        arg_type = String
        required = false
        default = "4x10"
        "--model_type_2"
        help = "3x10, 5x10..."
        arg_type = String
        required = false
        default = "4x10"
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "./results/"
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true
        "--model_path_vaghar_results"
        help = "the path to the original model i want to check the delta_diff with"
        arg_type = String
        required = false
        default = "/root/Downloads/vaghar_as_should_be_originally_no_c_target/results_trying_something/4x10_model.p_linf_0.05_NoCtarget_RegularVaghar_Itr18.txt"
        "--string_for_name"
        help = "the path to the original model i want to check the delta_diff with"
        arg_type = String
        required = false
        default = ""
    end
    return parse_args(s)
end


function get_delta1_vaghar(model_path_vaghar_results, line_index)
    
    open(model_path_vaghar_results, "r") do io
        current_line_number = 0
        requested_line = ""
        while !eof(io)
            current_line_number += 1
            line_content = readline(io)
            if current_line_number == line_index
                requested_line = line_content
            end
        end
        if requested_line==""
            println("Error with requested_line")
            exit()
        end
        parsed_tokens = Base.split(requested_line, ',')
        println("vaghar_solve_time")
        println(string(parse(Float64, parsed_tokens[end])))
        exit()
        return parse(Float64, parsed_tokens[end-1])
    end
end

function print_variables_by_base_name!(model, base_name)
    println("--- Printing variables starting with '$(base_name)' ---")

    # 1. Get all variables from the model
    all_vars = all_variables(model)

    # 2. Iterate and check the name
    for v in all_vars
        var_name = JuMP.name(v)
        
        # Check if the variable name starts with the desired base name
        if startswith(var_name, base_name)
            # The 'value()' function retrieves the final solution value in JuMP
            var_value = JuMP.value(v) 
            println("$(var_name): $(var_value)")
            return var_value
        end
    end

    println("-----------------------------------------------------")
end

function main()
    args = parse_commandline()
    dataset = args["dataset"]
    string_for_name = args["string_for_name"]
    perturbation = "delta_diff"
    perturbation_size = parse_numbers_to_Float64(args["perturbation_size"])
    c_tags = parse_numbers_to_Int64(args["ctag"])
    results_path = args["output_dir"] # where to save the results
    timout = args["timout"]
    w, h, k, c = get_dataset_params( dataset )
    model_path_vaghar_results = args["model_path_vaghar_results"]
    model_path1 = args["model_path_org"]
    model_type1 = args["model_type_1"]
    model_type2 = args["model_type_2"]
    model_path2 = args["model_path_second"] # a path to a directory/folder with multiple models of the same type (as "model_name")
    results.str = "" 
    deps = [true, false]
    for dep in deps
        depstring = "Deps"
        if !dep
            depstring = "No" * depstring
        end
        for c_tag in c_tags
            nn1 = get_nn(model_path1, model_type1, w, h, k, c, dataset)
            nn2 = get_nn(model_path2, model_type2, w, h, k, c, dataset)
            delta1_vaghar = get_delta1_vaghar(model_path_vaghar_results, c_tag)
            println("delta1_vaghar")
            println(string(delta1_vaghar))
            exit()
            suboptimal_solution, suboptimal_time =  0,0
            optimizer = Gurobi.Optimizer
            d = Dict()
            d[:suboptimal_solution] = suboptimal_solution
            d[:suboptimal_time] = suboptimal_time
            mip_reset()
            bounds_time = @elapsed begin
                merge!(d, get_model(w, h, k, perturbation, perturbation_size, nn1, nn2, zeros(Float64, 1, w, h, k), optimizer,
                get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
            end
            d[:bounds_time] = bounds_time
            m = d[:Model]
            if dep
                perturbation_dependencies(m, nn1, perturbation, perturbation_size, w, h, k, 0)
                perturbation_dependencies(m, nn2, perturbation, perturbation_size, w, h, k, 2)
            end
            mip_set_delta_diff_property(m, d,delta1_vaghar, c_tag)
            set_optimizer(m, optimizer)
            mip_set_attr(m, perturbation, d, timout)
            MOI.set(m, Gurobi.CallbackFunction(), my_callback)
            optimize!(m)
            mip_log(m, d)
            println("Termination Status: ", termination_status(m))
            println("Primal Status: ", primal_status(m))
            # println(JuMP.value.(d[:v_out_1])==JuMP.value.(d[:v_out_2]))
            # println()
            print_variables_by_base_name!(m,"conf2")
            print_variables_by_base_name!(m,"conf1")
            conf2_p = print_variables_by_base_name!(m,"conf2_p")
            conf1_p = print_variables_by_base_name!(m,"conf1_p")
            print_variables_by_base_name!(m,"diff_obj")
            diff_p = -conf2_p+conf1_p
            println("diff_p: $(diff_p)")
            results.str = update_results_str(results.str, c_tag, "None", d)
            model_name_in_path=basename(model_path2)
            save_results(results_path, "deltaDiff_"*depstring*"_"*string_for_name*"_"*model_type1*"_"*model_type2, results.str, d, nn1, c_tag-1, "None", w, h, k,model_name_in_path)
        end
    end
end

main()
