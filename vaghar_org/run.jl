ENV["PYTHON"]="/usr/bin/python3.8"

using Gurobi
using PyCall
using PyPlot
using LinearAlgebra
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
include("utils/perturbation_intervals.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dataset", "-d"
        help = "mnist, fmnist, or cifar10"
        arg_type = String
        required = false
        default = "mnist"
        "--model_name", "-n"
        help = "3x10, 3x50, 6x10, 6x100, 9x200, cnn0, cnn1, cnn2, or cnn3"
        arg_type = String
        required = false
        default = "4x10"
        "--model_path", "-m"
        help = "model name"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr18.p"
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
        default = 1000#500#4000
        "--ct", "-t"
        help = "target classes"
        arg_type = String
        required = false
        default = "1,2,3,4,5,6,7,8,9,10"
        "--output_dir", "-o"
        help = "output dir"
        arg_type = String
        required = false
        default = "/root/Downloads/vaghar_org/results_PerturbationInterval/"
        "--verbose", "-v"
        help = "Increase verbosity"
        action = :store_true
        "--name_to_save"
        help = "string for results name file"
        arg_type = String
        required = false
        default = ""#"itr18"
        "--mode"
        help = "standard or transfer"
        arg_type = String
        required = false
        default = "transfer"
        "--model_name2"
        help = "architecture name for N2 (transfer_distilation mode, e.g. 4x10 when N1 is 2x10)"
        arg_type = String
        required = false
        default = ""
        "--model_path2"
        help = "path to second network N2 (transfer mode)"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr18.p"
        "--vaghar_results"
        help = "path to VHAGaR results file for delta_1 values (transfer mode)"
        arg_type = String
        required = false
        default = "/root/Downloads/vaghar_org/results/63902078677641_4x10_linf_0.05_ctag0_itr17.txt" #"/root/Downloads/vaghar_org/results/63902082439234_4x10_linf_0.05_ctag0_itr18.txt"#
        "--c_tag_mode"
        help = "true: c_pert=c_tag (untargeted), false: c_pert=c_target (targeted)"
        arg_type = Bool
        required = false
        default = false
        "--use_intervals"
        help = "activate interval bound constraints between N1 and N2 (transfer mode)"
        arg_type = Bool
        required = false
        default = false
        "--use_hyper_attack"
        help = "activate hyper attack"
        arg_type = Bool
        required = false
        default = false
        "--use_perturbed_intervals"
        help = "activate perturbation interval constraints between clean and perturbed copies"
        arg_type = Bool
        required = false
        default = false
        "--activate_vaghgar_deps"
        help = "activate  vaghgar depandencies"
        arg_type = Bool
        required = false
        default = false
        "--n1_p_mode"
        help = "activate n1_p mode and encode it (relevant for transfer)"
        arg_type = Bool
        required = false
        default = false
        "--n2_fewer_binars_encoding"
        help = "activate n2_fewer_binars_encoding(relevant for transfer=true,n1_p_mode=false,c_tag_mode=false)"
        arg_type = Bool
        required = false
        default = false
        "--composed_interval"
        help = "activate composed interval constraints I^C linking N1(x) directly to N2(x_p) (transfer mode)"
        arg_type = Bool
        required = false
        default = false
        "--use_relaxations"
        help = "activate conditional-triangle relaxation; in standard mode relaxes perturbation copy based on perturbed intervals, in transfer mode relaxes n2_org and n2_pert; eliminates binary variables for qualifying neurons"
        arg_type = Bool
        required = false
        default = false
        "--relaxation_threshold"
        help = "interval-width threshold Trelax: relax neuron when width < Trelax (0.1=conservative, 0.5=default, 1.0=aggressive, Inf=all)"
        arg_type = Float64
        required = false
        default = 0.5
        "--optimizing_intervals"
        help = "tighter per-neuron ReLU clipping in interval propagation (uses N1/N2 preact stability)"
        arg_type = Bool
        required = false
        default = true
        "--relaxation_gap_area"
        help = "use triangle relaxation-gap area scoring instead of interval width for relaxation threshold decision (Method 2)"
        arg_type = Bool
        required = false
        default = false
        "--no_n1_binaries_and_relaxtions_only_on_n2"
        help = "LP-relax all N1 binaries (a ∈ [0,1]) and relax N2(x_p) by conditioning on N2(x) instead of N1(x); keeps N2(x) exact as anchor"
        arg_type = Bool
        required = false
        default = false
        "--no_n1_encoding_at_all"
        help = "Skip N1 encoding entirely; replace conf(N1,x,c)>=delta_1 with interval-bounded constraints on N2 outputs using weight diff bounds"
        arg_type = Bool
        required = false
        default = false
        "--no_n2_xp_encoding"
        help = "Skip N2(x') encoding entirely; replace conf(N2,x',c) with interval-bounded output variables using perturbation bounds through N2. Assumes no_n1_encoding_at_all=false (N1 is fully encoded). Supports --use_zonotope, --refined_relu_zonotope, --zonotope_conv for tighter bounds."
        arg_type = Bool
        required = false
        default = false
        "--encode_n1_last_layer"
        help = "When no_n1_encoding_at_all is active, encode N1's last linear layer exactly using interval-bounded hidden variables; gives exact delta_diff instead of upper bound"
        arg_type = Bool
        required = false
        default = false
        "--n1_last_layer_no_binaries"
        help = "When encode_n1_last_layer is active, use pre-computed scalar lower bound on conf_n1 instead of binary max encoding; zero extra binaries, sound upper bound on delta_diff"
        arg_type = Bool
        required = false
        default = false
        "--constrain_n1_xp"
        help = "Add interval-based constraint that conf(N1,x',c_target)<=0 (N1 does not classify perturbed input as c_target); no extra variables, uses pre-computed pert bounds through N1"
        arg_type = Bool
        required = false
        default = false
        "--use_zonotope"
        help = "Use zonotope (affine arithmetic) instead of interval arithmetic for diff bound propagation; tighter bounds by tracking correlations between neurons"
        arg_type = Bool
        required = false
        default = false
        "--refined_relu_zonotope"
        help = "Refined ReLU case analysis in zonotope propagation: when one network is stable-active and the other is split, use tighter sub-case bounds instead of generic DeepZ; reduces generator count and bound width at zero extra cost (requires --use_zonotope true)"
        arg_type = Bool
        required = false
        default = false
        "--sparse_zonotope"
        help = "Sparse generator representation: split generators into dense correlated matrix and diagonal independent vector; same bounds, less computation (requires --use_zonotope true)"
        arg_type = Bool
        required = false
        default = false
        "--zonotope_gen_budget"
        help = "Generator reduction budget K: after each layer, keep top K generators by L1 norm and merge the rest into one bounding-box generator; 0 = disabled (requires --use_zonotope true)"
        arg_type = Int
        required = false
        default = 0
        "--zonotope_conv"
        help = "Propagate zonotope through convolutional layers (exact for linear conv, DeepZ for ReLU); activates zonotope at first Conv instead of Flatten (requires --use_zonotope true)"
        arg_type = Bool
        required = false
        default = false
        "--tighten_n2_bounds"
        help = "Derive tighter N2 pre-activation bounds from N1 + diff bounds; can flip split neurons to stable, eliminating binary variables (requires diff bounds to be computed)"
        arg_type = Bool
        required = false
        default = false
        "--delta_diff_positive"
        help = "force delta_diff to be positive."
        arg_type = Bool
        required = false
        default = true
        "--force_cpu"
        help = "force CPU-only mode for hyper attack (no GPU)"
        arg_type = Bool
        required = false
        default = false
        "--Threads_num"
        help = "Number of threads to use"
        arg_type = Int
        required = false
        default = 32

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
    use_hyper_attack = args["use_hyper_attack"]
    global Threads_num = args["Threads_num"]
    perturbation_size = parse_numbers_to_Float64(args["perturbation_size"])
    mode = args["mode"]

    if mode == "transfer"
        main_transfer(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    elseif mode == "transfer_distilation"
        main_transfer_distilation(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    else
        main_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    end
end

function main_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_attack)
    c_tag_list = [args["ctag"]]
    activate_vaghgar_deps = args["activate_vaghgar_deps"]
    global use_relaxations = args["use_relaxations"]
    global relaxation_threshold = args["relaxation_threshold"]
    global optimizing_intervals = args["optimizing_intervals"]
    global relaxation_gap_area = args["relaxation_gap_area"]
    name_to_save_init = name_to_save
    for c_tag in c_tag_list
        results.str = ""
        c_targets = parse_numbers_to_Int64(args["ct"])
        results_path = args["output_dir"]
        timout = args["timout"]
        w, h, k, c = get_dataset_params( dataset )
        token_signature = string(now().instant.periods.value)
        nn = get_nn(model_path, model_name, w, h, k, c, dataset)
        for c_target in c_targets
            name_to_save = name_to_save_init
            global relaxation_condition_count = 0
            if c_tag==c_target
                continue
            end
            suboptimal_solution, suboptimal_time =  0,0
            if use_hyper_attack
                suboptimal_solution, suboptimal_time =  hyper_attack(dataset, c_tag, c_target, token_signature, model_name, model_path, perturbation, perturbation_size; force_cpu=args["force_cpu"])
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
            if use_hyper_attack
                hyper_attack_hints(m, token_signature, c_tag, c_target)
                name_to_save = name_to_save*"_HyperAttackHints"
            end
            if activate_vaghgar_deps
                name_to_save = name_to_save*"_VagharDeps"
                perturbation_dependencies(m, nn, perturbation, perturbation_size, w, h, k;
                                          perturbation_var=d[:Perturbation])
            end
            if args["use_perturbed_intervals"]
                name_to_save = name_to_save*"_PertruebedIntervals"
                println("Adding perturbed interval constraints...")
                perturbed_interval_constraints(m, nn, "org", "perturbation")
            end
            if args["use_relaxations"]
                name_to_save = name_to_save*"_Relaxations"*string(args["relaxation_threshold"])
                if args["relaxation_gap_area"]
                    name_to_save = name_to_save*"_GapArea"
                end
                println("Applying conditional triangle relaxations with threshold $(args["relaxation_threshold"]) (gap_area=$(args["relaxation_gap_area"]))...")
            end
            mip_set_delta_property(m, perturbation, d)
            set_optimizer(m, optimizer)
            mip_set_attr(m, perturbation, d, timout)
            MOI.set(m, Gurobi.CallbackFunction(), my_callback)
            optimize!(m)
            mip_log(m, d)
            mip_reuse_bounds()
            results.str = update_results_str(results.str, c_tag, c_target, d)
            println(results_path)
            if args["use_relaxations"]
                name_to_save = name_to_save * "_RelaxCount" * string(relaxation_condition_count)
            end
            save_results(results_path, model_name, perturbation, perturbation_size, results.str, d, nn, c_tag-1, c_target-1, w, h, k,name_to_save*"_cTag"*string(c_tag),token_signature)
        end
    end
end

function main_transfer(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_Attack_delta_diff)
    model_path2 = args["model_path2"]
    vaghar_results = args["vaghar_results"]
    c_tag_mode = args["c_tag_mode"]
    use_intervals = args["use_intervals"]
    results_path = args["output_dir"]
    timout = args["timout"]
    c_tag_list = [args["ctag"]]
    c_targets = parse_numbers_to_Int64(args["ct"])
    n1_p_mode = args["n1_p_mode"]
    global use_relaxations = args["use_relaxations"]
    global relaxation_threshold = args["relaxation_threshold"]
    global optimizing_intervals = args["optimizing_intervals"]
    global relaxation_gap_area = args["relaxation_gap_area"]
    global no_n1_binaries_and_relaxtions_only_on_n2 = args["no_n1_binaries_and_relaxtions_only_on_n2"]
    global no_n1_encoding_at_all = args["no_n1_encoding_at_all"]
    global no_n2_xp_encoding = args["no_n2_xp_encoding"]
    global encode_n1_last_layer = args["encode_n1_last_layer"]
    global use_zonotope = args["use_zonotope"]
    global refined_relu_zonotope = args["refined_relu_zonotope"]
    global sparse_zonotope = args["sparse_zonotope"]
    global zonotope_gen_budget = args["zonotope_gen_budget"]
    global zonotope_conv = args["zonotope_conv"]
    global tighten_n2_bounds = args["tighten_n2_bounds"]
    # no_n1_encoding_at_all implies no_n1_binaries_and_relaxtions_only_on_n2
    # (N1 isn't encoded, so N2(x') must be relaxed onto N2(x) instead of N1)
    if no_n1_encoding_at_all
        global no_n1_binaries_and_relaxtions_only_on_n2 = true
        n1_p_mode = false  # can't encode N1(x') without N1
    end
    # no_n2_xp_encoding assumes no_n1_encoding_at_all is OFF (N1 fully encoded)
    if no_n2_xp_encoding && no_n1_encoding_at_all
        println("WARNING: --no_n2_xp_encoding requires --no_n1_encoding_at_all=false, disabling no_n1_encoding_at_all")
        global no_n1_encoding_at_all = false
    end
    # encode_n1_last_layer only makes sense with no_n1_encoding_at_all
    if encode_n1_last_layer && !no_n1_encoding_at_all
        println("WARNING: --encode_n1_last_layer ignored (requires --no_n1_encoding_at_all)")
        global encode_n1_last_layer = false
    end
    global n1_last_layer_no_binaries = args["n1_last_layer_no_binaries"]
    if n1_last_layer_no_binaries && !encode_n1_last_layer
        println("WARNING: --n1_last_layer_no_binaries ignored (requires --encode_n1_last_layer)")
        global n1_last_layer_no_binaries = false
    end
    constrain_n1_xp = args["constrain_n1_xp"]
    if constrain_n1_xp && !no_n1_encoding_at_all
        println("WARNING: --constrain_n1_xp ignored (requires --no_n1_encoding_at_all)")
        constrain_n1_xp = false
    end
    use_vaghgarDeps = args["activate_vaghgar_deps"]
    n2_fewer_binars_encoding = args["n2_fewer_binars_encoding"]
    w, h, k, c = get_dataset_params(dataset)

    println("Loading N1 from: $model_path")
    nn1 = get_nn(model_path, model_name, w, h, k, c, dataset)
    println("Loading N2 from: $model_path2")
    nn2 = get_nn(model_path2, model_name, w, h, k, c, dataset)

    K = layers_number(nn1)
    println("ReLU layers per network: $K, dependency offset: $(2*K)")
    name_to_save_init = name_to_save

    for c_tag in c_tag_list
        token_signature = string(now().instant.periods.value)
        results.str = ""
        for c_target in c_targets
            if c_tag_mode
                if c_target != c_tag
                    continue
                end
            else
                if c_target == c_tag
                    continue
                end
            end

            println("=== c_tag=$c_tag, c_target=$c_target ===")
            delta_1 = get_delta1_vaghar(vaghar_results, c_target)
            name_to_save = name_to_save_init
            global relaxation_condition_count = 0
            println("delta_1 = $delta_1")
            if delta_1 <= 0
                println("Skipping: delta_1 <= 0")
                continue
            end

            # PGD attack for warm-start lower bound
            suboptimal_solution, suboptimal_time = 0, 0
            if use_hyper_Attack_delta_diff
                name_to_save = name_to_save * "_HyperAttack"
                suboptimal_solution, suboptimal_time =hyper_attack_transfer(
                    dataset, c_tag, c_target, token_signature,
                    model_name, model_path, model_path2,
                    perturbation, perturbation_size, delta_1,
                    c_tag_mode, n1_p_mode, args["delta_diff_positive"];
                    force_cpu=args["force_cpu"])
            end
            println("Hyper attack: best_val=$suboptimal_solution, time=$suboptimal_time")

            optimizer = Gurobi.Optimizer
            mip_reset()

            println("Encoding four-network MIP...")
            d = Dict()
            d[:suboptimal_time] = suboptimal_time
            bounds_time = @elapsed begin
                merge!(d, get_model_transfer(w, h, k, perturbation, perturbation_size,
                    nn1, nn2, zeros(Float64, 1, w, h, k), optimizer,
                    get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM,
                    n1_p_mode))
            end
            d[:bounds_time] = bounds_time
            m = d[:Model]

            # Apply warm-start hints from PGD attack
            if use_hyper_Attack_delta_diff
                name_to_save = name_to_save * "_Hints"
                hyper_attack_hints(m, token_signature, c_tag, c_target)
            end

            if use_vaghgarDeps
                name_to_save = name_to_save * "_VaghgarDeps"
                # Dependencies for N2: original vs perturbed (requires N2(xp) encoding)
                if no_n2_xp_encoding
                    println("Skipping N2 dependencies (--no_n2_xp_encoding, no N2(x') variables)")
                elseif no_n1_encoding_at_all
                    # N1 not encoded → N2(x) starts at layer 1, N2(x') at K+1
                    perturbation_dependencies(m, nn2, perturbation, perturbation_size, w, h, k;
                                            activation_start=1, layers_offset=K,
                                            perturbation_var=d[:Perturbation])
                else
                    # N1 encoded → N2(x) at layers K+1..2K, N2(x') at 3K+1..4K
                    perturbation_dependencies(m, nn2, perturbation, perturbation_size, w, h, k;
                                            activation_start=K+1, layers_offset=2*K,
                                            perturbation_var=d[:Perturbation])
                end
            end

            # Interval bounds between N1 and N2 (requires N1 encoding)
            if use_intervals && !no_n1_encoding_at_all
                name_to_save = name_to_save * "_LucidIntervals"
                println("Adding interval constraints between N1 and N2...")
                transfer_interval_constraints(m, nn1, nn2, perturbation, perturbation_size, w, h, k)
            end
            if args["use_relaxations"]
                name_to_save = name_to_save*"_Relaxations"*string(args["relaxation_threshold"])
                if args["relaxation_gap_area"]
                    name_to_save = name_to_save*"_GapArea"
                end
                println("Applying conditional triangle relaxations with threshold $(args["relaxation_threshold"]) (gap_area=$(args["relaxation_gap_area"]))...")
            end

            # Perturbation interval bounds (clean ↔ perturbed for each network)
            if args["use_perturbed_intervals"]
                name_to_save = name_to_save * "_PerturbedIntervals"
                println("Adding perturbed interval constraints...")
                if !no_n1_encoding_at_all
                    perturbed_interval_constraints(m, nn1, "n1_org", "n1_pert")
                end
                if !no_n2_xp_encoding
                    perturbed_interval_constraints(m, nn2, "n2_org", "n2_pert")
                else
                    println("Skipping N2 perturbed interval constraints (--no_n2_xp_encoding)")
                end
            end

            # Composed interval constraints: I^C linking N1(x) ↔ N2(x_p) directly (requires N1 and N2(xp))
            if args["composed_interval"] && !no_n1_encoding_at_all && !no_n2_xp_encoding
                name_to_save = name_to_save * "_ComposedIntervals"
                println("Adding composed interval constraints (I^C) between N1(x) and N2(x_p)...")
                composed_interval_constraints(m, nn1, nn2, perturbation, perturbation_size, w, h, k)
            end

            if n2_fewer_binars_encoding
                name_to_save = name_to_save * "_N2encodingWithFewerBinars"
            end
            if optimizing_intervals
                name_to_save = name_to_save * "_OptimizingIntervals"
            end
            # if no_n1_binaries_and_relaxtions_only_on_n2
            #     name_to_save = name_to_save * "_NoN1BinRelaxOnN2only"
            # end
            if no_n1_encoding_at_all
                name_to_save = name_to_save * "_NoN1Enc"
            end
            if no_n2_xp_encoding
                name_to_save = name_to_save * "_NoN2xpEnc"
            end
            if encode_n1_last_layer
                name_to_save = name_to_save * "_N1LastLayer"
            end
            if n1_last_layer_no_binaries
                name_to_save = name_to_save * "_NoBin"
            end
            if use_zonotope
                name_to_save = name_to_save * "_Zonotope"
            end
            if refined_relu_zonotope
                name_to_save = name_to_save * "_RefinedReLU"
            end
            if sparse_zonotope
                name_to_save = name_to_save * "_SparseZono"
            end
            if zonotope_gen_budget > 0
                name_to_save = name_to_save * "_GenBudget" * string(zonotope_gen_budget)
            end
            if zonotope_conv
                name_to_save = name_to_save * "_ZonoConv"
            end
            if tighten_n2_bounds
                name_to_save = name_to_save * "_TightenN2"
            end

            # Set transfer proof constraints and objective
            if no_n2_xp_encoding
                mip_set_transfer_property_no_n2_xp(m, d, delta_1, c_tag, c_target,
                    c_tag_mode, n1_p_mode, n2_fewer_binars_encoding,
                    args["delta_diff_positive"])
            elseif no_n1_encoding_at_all && encode_n1_last_layer
                mip_set_transfer_property_n1_last_layer(m, d, delta_1, c_tag, c_target,
                    c_tag_mode, n2_fewer_binars_encoding,
                    args["delta_diff_positive"], nn1, n1_last_layer_no_binaries)
            elseif no_n1_encoding_at_all
                mip_set_transfer_property_no_n1(m, d, delta_1, c_tag, c_target,
                    c_tag_mode, n2_fewer_binars_encoding,
                    args["delta_diff_positive"])
            else
                mip_set_transfer_property(m, d, delta_1, c_tag, c_target,
                    c_tag_mode, n1_p_mode, n2_fewer_binars_encoding,
                    args["delta_diff_positive"])
            end
            # Add interval-based constraint: conf(N1, x', c_target) <= 0
            if constrain_n1_xp && !c_tag_mode
                name_to_save = name_to_save * "_N1xpConf"
                add_n1_xp_confidence_constraint!(m, d, c_tag, c_target)
            end

            set_optimizer(m, optimizer)
            mip_set_attr_transfer(m, timout, suboptimal_solution, args["delta_diff_positive"])
            MOI.set(m, Gurobi.CallbackFunction(), my_callback)

            println("Optimizing...")
            optimize!(m)
            mip_log(m, d)

            results.str = update_results_str(results.str, c_tag, c_target, d)
            println(results.str)
            # Save results for this c_tag
            ct_str = c_tag_mode ? "cTagMode" : "cTargetMode"
            
            if args["use_relaxations"] || no_n1_binaries_and_relaxtions_only_on_n2
                name_to_save = name_to_save * "_RelaxCount" * string(relaxation_condition_count)
            end

            name_to_save = name_to_save * "_Therads" * string(Threads_num)
            global Threads_num
            basename = token_signature * "_" * model_name * "_transfer_" *
                        perturbation * "_" * create_perturbation_string(perturbation_size) *
                        "_ctag" * string(c_tag) * "_" * ct_str * "_" * name_to_save
            file = open(safe_filepath(results_path, basename), "w")
            write(file, results.str)
            close(file)
        end


    end
    println("Transfer proof computation complete.")
end

function main_transfer_distilation(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_Attack_delta_diff)
    model_name2 = args["model_name2"]
    if model_name2 == ""
        error("transfer_distilation mode requires --model_name2 (e.g. 4x10 when N1 is 2x10)")
    end
    model_path2 = args["model_path2"]
    vaghar_results = args["vaghar_results"]
    c_tag_mode = args["c_tag_mode"]
    use_intervals = args["use_intervals"]
    results_path = args["output_dir"]
    timout = args["timout"]
    c_tag_list = [args["ctag"]]
    c_targets = parse_numbers_to_Int64(args["ct"])
    n1_p_mode = args["n1_p_mode"]
    use_vaghgarDeps = args["activate_vaghgar_deps"]
    n2_fewer_binars_encoding = args["n2_fewer_binars_encoding"]
    w, h, k, c = get_dataset_params(dataset)

    println("Loading N1 ($model_name) from: $model_path")
    nn1 = get_nn(model_path, model_name, w, h, k, c, dataset)
    println("Loading N2 ($model_name2) from: $model_path2")
    nn2 = get_nn(model_path2, model_name2, w, h, k, c, dataset)

    K1 = layers_number(nn1)
    K2 = layers_number(nn2)
    println("N1 ReLU layers: $K1, N2 ReLU layers: $K2")
    name_to_save_init = name_to_save

    for c_tag in c_tag_list
        token_signature = string(now().instant.periods.value)
        results.str = ""
        for c_target in c_targets
            if c_tag_mode
                if c_target != c_tag
                    continue
                end
            else
                if c_target == c_tag
                    continue
                end
            end

            println("=== c_tag=$c_tag, c_target=$c_target ===")
            delta_1 = get_delta1_vaghar(vaghar_results, c_target)
            name_to_save = name_to_save_init
            global relaxation_condition_count = 0
            println("delta_1 = $delta_1")
            if delta_1 <= 0
                println("Skipping: delta_1 <= 0")
                continue
            end

            # PGD attack for warm-start lower bound
            suboptimal_solution, suboptimal_time = 0, 0
            if use_hyper_Attack_delta_diff
                name_to_save = name_to_save * "_HyperAttack"
                suboptimal_solution, suboptimal_time = hyper_attack_transfer_distilation(
                    dataset, c_tag, c_target, token_signature,
                    model_name, model_name2, model_path, model_path2,
                    perturbation, perturbation_size, delta_1, c_tag_mode, n1_p_mode;
                    force_cpu=args["force_cpu"])
            end
            println("Hyper attack: best_val=$suboptimal_solution, time=$suboptimal_time")

            optimizer = Gurobi.Optimizer
            mip_reset()

            println("Encoding MIP for distillation transfer...")
            d = Dict()
            d[:suboptimal_time] = suboptimal_time
            bounds_time = @elapsed begin
                merge!(d, get_model_transfer(w, h, k, perturbation, perturbation_size,
                    nn1, nn2, zeros(Float64, 1, w, h, k), optimizer,
                    get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM,
                    n1_p_mode))
            end
            d[:bounds_time] = bounds_time
            m = d[:Model]

            # Apply warm-start hints from PGD attack
            if use_hyper_Attack_delta_diff
                name_to_save = name_to_save * "_Hints"
                hyper_attack_hints(m, token_signature, c_tag, c_target)
            end

            if use_vaghgarDeps
                name_to_save = name_to_save * "_VaghgarDeps"
                # Dependencies for N2: original layers K1+1..K1+K2 ↔ perturbed layers
                deps_offset = n1_p_mode ? K1 + K2 : K2
                perturbation_dependencies(m, nn2, perturbation, perturbation_size, w, h, k;
                                        activation_start=K1+1, layers_offset=deps_offset,
                                        perturbation_var=d[:Perturbation])
            end

            # Interval bounds between N1 and N2 (distillation: every 2nd layer of N2)
            if use_intervals
                name_to_save = name_to_save * "_LucidIntervals"
                println("Adding distillation interval constraints (every 2nd layer)...")
                transfer_interval_constraints_distilation(m, nn1, nn2)
            end

            # Perturbation interval bounds (clean ↔ perturbed for each network)
            if args["use_perturbed_intervals"]
                name_to_save = name_to_save * "_PerturbedIntervals"
                println("Adding perturbed interval constraints for N1 and N2...")
                perturbed_interval_constraints(m, nn2, "n2_org", "n2_pert")
            end

            # Composed interval constraints (distillation: every 2nd layer mapping)
            if args["composed_interval"]
                name_to_save = name_to_save * "_ComposedIntervals"
                println("Adding distillation composed interval constraints...")
                composed_interval_constraints_distilation(m, nn1, nn2)
            end

            if n2_fewer_binars_encoding
                name_to_save = name_to_save * "_N2encodingWithFewerBinars"
            end
            if optimizing_intervals
                name_to_save = name_to_save * "_OptimizingIntervals"
            end

            # Set transfer proof constraints and objective
            mip_set_transfer_property(m, d, delta_1, c_tag, c_target, c_tag_mode, n1_p_mode, n2_fewer_binars_encoding)
            set_optimizer(m, optimizer)
            mip_set_attr_transfer(m, timout, suboptimal_solution)
            MOI.set(m, Gurobi.CallbackFunction(), my_callback)

            println("Optimizing...")
            optimize!(m)
            mip_log(m, d)

            results.str = update_results_str(results.str, c_tag, c_target, d)
            println(results.str)
            ct_str = c_tag_mode ? "cTagMode" : "cTargetMode"

            if args["use_relaxations"]
                name_to_save = name_to_save * "_RelaxCount" * string(relaxation_condition_count)
            end
        
            basename = token_signature * "_" * model_name * "_" * model_name2 *
                        "_transfer_distilation_" *
                        perturbation * "_" * create_perturbation_string(perturbation_size) *
                        "_ctag" * string(c_tag) * "_" * ct_str * "_" * name_to_save
            file = open(safe_filepath(results_path, basename), "w")
            write(file, results.str)
            close(file)
        end
    end
    println("Transfer distillation computation complete.")
end

main()