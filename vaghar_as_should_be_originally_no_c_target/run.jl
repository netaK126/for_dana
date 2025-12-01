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
        default = "./models/3x10/model.p"
        "--perturbation", "-p"
        help = "perturbation type: occ, patch, brightness, linf, contrast, translation, rotation, or max"
        arg_type = String
        required = false
        default = "linf"
        "--perturbation_size", "-s"
        help = "occ: i,j,width , patch: eps,i,j,width, brightness: eps, linf: eps, contrast: eps, translation: tx,ty, rotation: angle"
        arg_type = String
        required = false
        default = "0.1"
        "--ctag", "-c"
        help = "ctag, source class"
        arg_type = Int
        required = false
        default = 3
        "--timout"
        help = "MIP timeout"
        arg_type = Int
        required = false
        default = 108000000
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "4"
        "--folder_path"
        help = "a folder with all the models i want to verify"
        arg_type = String
        required = false
        default = "/root/Downloads/code_deprecated_active_just_for_models/models/4x10_distilation/"
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "/root/Downloads/vaghar_as_should_be_originally_no_c_target/results_trying_something/"
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true
        
    end
    return parse_args(s)
end

# max_number_in_dir.jl
function max_number_in_dir(dir_path, splitting_index, c_tag)
    # Get all files in the directory
    files = filter(f -> isfile(joinpath(dir_path, f)) && occursin(splitting_index, f) && occursin("zing"*string(c_tag)*".txt", f) , readdir(dir_path))
    
    # Read each file, parse the number, and collect them
    numbers = Float64[]
    for file in files
        filepath = joinpath(dir_path, file)
        line = strip(read(filepath, String))
        m = match(r"bound\s*:\s*([0-9\.\-eE]+)", line)
        print("m = ")
        println(m)
        if m !== nothing
            push!(numbers, parse(Float64, m.captures[1]))
        else
            println("No upper_bound found in $file")
            exit(1)
        end
    end

    
    return maximum(numbers)
    
end

# Example usage:
function main()
    
    args = parse_commandline()
    dataset = args["dataset"]
    model_name = args["model_name"] # the model architecture. it an be 4x10, 3x10 and so on...
    folder_path = args["folder_path"] # a path to a directory/folder with multiple models of the same type (as "model_name")
    files_and_dirs = readdir(folder_path) # getting all the models' paths from folder_path
    models_path_list= []

    # the following code only considers ".p" files (and skips ".pth" files)
    for item in files_and_dirs
        full_path = joinpath(folder_path, item)
        if isfile(full_path)
            # Perform operations on the file here
            if endswith(full_path, ".p")
                if startswith(item,"CONF_alphaVal") # these are models with low accuracy, so there was no point in verifying them here.
                    continue
                else
                    push!(models_path_list,full_path)
                end
            end
        end
    end


    for model_path in models_path_list
        perturbation = args["perturbation"]
        perturbation_size_list = [0.05,0.1]
        c_tag_list = [1,2,3,4,5] #args["ctag"]
        c_targets = parse_numbers_to_Int64(args["ct"])
        results_path = args["output_dir"]
        timout = args["timout"]
        w, h, k, c = get_dataset_params( dataset )
        splitingIndex_list = [""] # this is not in use (it would be used after finding delta_diff would be complete)
        for perturbation_size in perturbation_size_list
            for spliting_index in splitingIndex_list
                results.str = ""
                for c_tag in c_tag_list
                    for c_target in c_targets # this is here historicly. not in use in our settings.
                        nn = get_nn(model_path, model_name, w, h, k, c, dataset)
                        token_signature = string(now().instant.periods.value)
                        max_abs_value = 0 # this is delta_diff
                        if spliting_index!=""
                            println("spliting index")
                            max_abs_value = max_number_in_dir("/root/Downloads/lucid/results_mnist_ensamble_bagging/", spliting_index, c_tag)
                        end
                        
                        print("delta_diff = ")
                        println(max_abs_value)
                        suboptimal_solution, suboptimal_time = 0,0
                        if perturbation != "max"
                            println("Applying hyper attack")
                            suboptimal_solution, suboptimal_time =  hyper_attack(dataset, c_tag, c_target, token_signature, model_name, model_path, perturbation, perturbation_size, max_abs_value)
                        end

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
                        println("encoded model")
                        d[:max_abs_value] = Float64(max_abs_value)
                        if perturbation != "max"
                            hyper_attack_hints(m, token_signature, c_tag, c_target)
                            println("encoded hints")
                        end
                        perturbation_dependencies(m, nn, perturbation, perturbation_size, w, h, k)
                        println("encoded deps")
                        mip_set_delta_property(m, perturbation, d,c_tag)
                        println("encoded delta and delta_p")
                        set_optimizer(m, optimizer)
                        println("encoded optimizer")
                        mip_set_attr(m, perturbation, d, timout)
                        println("encoded attr")
                        MOI.set(m, Gurobi.CallbackFunction(), my_callback)
                        optimize!(m)
                        mip_log(m, d)
                        try
                            results.str = update_results_str(results.str, c_tag, c_target, d)
                        catch  e
                            continue
                        end
                        model_name_in_path=basename(model_path)
                        save_results(results_path, model_name, perturbation, perturbation_size, results.str, d, nn, c_tag-1, c_target-1, w, h, k, c_tag,model_name_in_path)
                    end
                end
            end
        end
    end
end

main()