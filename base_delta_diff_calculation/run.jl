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
        default = "0.1"
        "--ctag", "-c" #not needed
        help = "ctag, source class"
        arg_type = String
        required = false
        default = "1,2,3,4,5,6,7,8,9,10"
        "--timout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 10800
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
        default = "./results_distilation_geminiModels/"
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true
    end
    return parse_args(s)
end



function main()
    args = parse_commandline()
    dataset = args["dataset"]
    perturbation = "delta_diff"
    c_tags = parse_numbers_to_Int64(args["ctag"])
    results_path = args["output_dir"] # where to save the results
    timout = args["timout"]
    w, h, k, c = get_dataset_params( dataset )
    model_path1 = args["model_path_org"]
    model_type1 = args["model_type_1"]
    model_type2 = args["model_type_2"]
    folder_path = args["folder_path"] # a path to a directory/folder with multiple models of the same type (as "model_name")
    files_and_dirs = readdir(folder_path) # getting all the models' paths from folder_path

    # the following code only considers ".p" files (and skips ".pth" files)
    models_path_nn2 = []
    for item in files_and_dirs
        full_path = joinpath(folder_path, item)
        # println(item)
        if isfile(full_path)
            # Perform operations on the file here
            if endswith(full_path, "_model.p")
                if occursin("Tval10",full_path)
                    continue
                elseif occursin("Tval11",full_path)
                    continue
                else
                    # println("Processing file: $full_path")
                    push!(models_path_nn2,full_path)
                end
            end
        end
    end
    forcingDelta1ToBePositive_list = [false, true]
    for forcingDelta1ToBePositive_item in forcingDelta1ToBePositive_list
        forcing_string = "ForcingDelta1ToBePositive_"
        if forcingDelta1ToBePositive_item==false
            forcing_string="Not"*forcing_string
        end
        for model_path2 in models_path_nn2
            results.str = "" 
            
            nn1 = get_nn(model_path1, model_type1, w, h, k, c, dataset)
            nn2 = get_nn(model_path2, model_type2, w, h, k, c, dataset)
            for c_tag in c_tags
                suboptimal_solution, suboptimal_time =  0,0
                optimizer = Gurobi.Optimizer
                d = Dict()
                d[:suboptimal_solution] = suboptimal_solution
                d[:suboptimal_time] = suboptimal_time
                mip_reset()
                bounds_time = @elapsed begin
                    merge!(d, get_model(w, h, k, perturbation, nn1, nn2, zeros(Float64, 1, w, h, k), optimizer,
                    get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
                end
                d[:bounds_time] = bounds_time
                m = d[:Model]
                # perturbation_dependencies(m, nn1, perturbation, 0, w, h, k)
                vars = mip_set_delta_diff_property!(m, d, c_tag,forcingDelta1ToBePositive_item, M=1e6)
                set_optimizer(m, optimizer)
                mip_set_attr(m, perturbation, d, timout)
                MOI.set(m, Gurobi.CallbackFunction(), my_callback)
                optimize!(m)
                mip_log(m, d)
                results.str = update_results_str(results.str, c_tag, "None", d)
                model_name_in_path=basename(model_path2)
                
                save_results(results_path, forcing_string*"deltaDiff_"*model_type1*"_"*model_type2, results.str, d, nn1, c_tag-1, "None", w, h, k,model_name_in_path)
            end
        end
    end
end

main()
