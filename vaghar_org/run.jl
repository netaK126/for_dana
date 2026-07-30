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
using Serialization

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
include("utils/n2_relax_decision.jl")
include("utils/sibgate_emit.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dataset", "-d"
        help = "mnist, fmnist, cifar10, or (with --internet_nets_benchmarks) har"
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
        help = "standard, advanced_standard_n1, or advanced_standard_n2"
        arg_type = String
        required = false
        default = "standard"
        "--model_path2"
        help = "path to second network N2 (transfer mode)"
        arg_type = String
        required = false
        default = "/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/model_itr18.p"
        "--use_hyper_attack"
        help = "activate hyper attack"
        arg_type = Bool
        required = false
        default = false
        "--use_perturbed_intervals"
        help = "activate perturbation interval constraints between clean and perturbed copies"
        arg_type = Bool
        required = false
        default = true
        "--geometric_intervals"
        help = "Exploit pixel-relocation structure for translation/rotation in the perturbed-interval bounds " *
               "(exact (T-I) map composed through the first layer). Interval-only, no zonotope. " *
               "Sound: delta unchanged. Default true; ignored (with a warning) for other perturbations " *
               "and when --use_perturbed_intervals=false."
        arg_type = Bool
        required = false
        default = true
        "--activate_vaghgar_deps"
        help = "activate  vaghgar depandencies"
        arg_type = Bool
        required = false
        default = false
        "--optimizing_intervals"
        help = "tighter per-neuron ReLU clipping in interval propagation (uses N1/N2 preact stability)"
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
        "--gurobi_seed"
        help = "Gurobi Seed parameter — perturbs tie-breaking for variance measurement across identical runs"
        arg_type = Int
        required = false
        default = 0
        # ── Advanced-standard mode flags ─────────────────────────────────
        "--adv_std_var_hint"
        help = "advanced_standard variable-hint mode (Technique 5): off | prev_pgd. " *
               "'prev_pgd' shifts N1's pre-activation by the diff bound, clips to " *
               "[l_n2, u_n2], takes p from interval-length ratios, and routes hint_val into " *
               "set_start_value() (Gurobi Start) with PGD consensus filtering: fill where PGD " *
               "is silent, leave where PGD agrees, withdraw where PGD disagrees. VarHintVal/" *
               "VarHintPri are never set."
        arg_type = String
        required = false
        default = "off"
        range_tester = x -> lowercase(strip(String(x))) in ("off", "prev_pgd")
        "--adv_std_bound_tightening"
        help = "advanced_standard: tighten N2 bounds using N1's bounds shifted by the zonotope diff bounds (Technique 4)"
        arg_type = Bool
        required = false
        default = true
        "--adv_std_zono_bounds"
        help = "advanced_standard: within Technique 4, compute N2 bounds via zonotope propagation " *
               "(compute_diff_bounds_zonotope) plus a second N1-tightened absolute N2 zonotope pass, " *
               "and intersect both against the ReLU [l,u]. Strictly tighter than the default interval " *
               "path; preserves the integer optimum. Requires --adv_std_bound_tightening true."
        arg_type = Bool
        required = false
        default = false
        "--adv_std_zono_npre"
        help = "advanced_standard: let the zonotope use N_pre. Default true. Set false to ablate " *
               "ONLY the N_pre contribution: the absolute N2 zonotope is still propagated, but it " *
               "is NOT intersected with N_pre's pre-activation bounds or the N_pre->N difference " *
               "zonotope (Source A). The perturbation-difference technique, which reads the same " *
               "diff bounds, is unaffected. Only meaningful with --adv_std_zono_bounds true; " *
               "filename tag _noNpreZono."
        arg_type = Bool
        required = false
        default = true
        "--adv_std_n2_relax_threshold"
        help = "advanced_standard (Technique 6): replace N2/N2p ReLU binaries with a " *
               "triangle LP relaxation (no binary) when the triangle-gap-area of N1's interval " *
               "at the same neuron is <= this value. Sound over-approximation: delta_relaxed >= " *
               "delta_exact, and every concrete feasible point remains feasible in the relaxed " *
               "MIP. Default -1.0 disables. Requires --adv_std_bound_tightening true so that " *
               "n1_neuron_bounds is populated."
        arg_type = Float64
        required = false
        default = -1.0
        "--adv_std_n2_sibling_gate"
        help = "advanced_standard (Technique 4 / SibGate): augment Technique 3's tiered " *
               "decision rule by emitting a conditional triangle gated on the surviving sibling " *
               "binary in the 'one thin' tier and a pre-activation coupling line in the 'both " *
               "thin' tier. Sound over-approximation: delta_relaxed >= delta_exact. Requires " *
               "--adv_std_n2_relax_threshold >= 0. Filename tag _SibGate with per-tier counts."
        arg_type = Bool
        required = false
        default = false
        "--n1_state_dir"
        help = "Directory with pre-saved N1 solver state (from advanced_standard_n1). When set, skip N1 solve and load state from disk."
        arg_type = String
        required = false
        default = ""

        # ── Standard-mode boosting flags (Boosting Standard Mode, single-network N1) ──
        # Mirror the advstd techniques but applied to N1's own dual-copy MIP
        # (org + perturbation). See advstd_techniques.tex §3 (sec:std).
        "--nn1_zono_bounds"
        help = "standard mode (Boosting Standard Mode): tighten N1's per-neuron ReLU pre-activation " *
               "bounds via an absolute zonotope propagated through N1 (Source B). See §3.2 " *
               "(sec:std_zono) of advstd_techniques.tex. Strictly tighter than interval arithmetic; " *
               "preserves the integer optimum. Filename tag _stdBoost_zono."
        arg_type = Bool
        required = false
        default = false
        "--nn1_relax_threshold"
        help = "standard mode (Boosting Standard Mode): replace N1's org/pert ReLU binaries with a " *
               "triangle LP relaxation (no binary) when the triangle-gap-area of the intersected " *
               "per-copy bounds is ≤ this value (tiered per-copy rule). Sound: δ_relaxed ≥ δ_exact. " *
               "Default -1.0 disables. Requires --nn1_zono_bounds=true to populate Source B. " *
               "Requires --use_perturbed_intervals=true for soundness on the one-thin tier. " *
               "Filename tag _stdBoost_BTPR{τ}. See §3.3 (sec:std_btpr)."
        arg_type = Float64
        required = false
        default = -1.0
        "--nn1_sibling_gate"
        help = "standard mode (Boosting Standard Mode): augment Per-Copy Triangle Drop with the " *
               "sibling-gated conditional triangle (one-thin tier) and pre-activation coupling " *
               "(both-thin tier). Sound: δ_relaxed ≥ δ_exact. Requires --nn1_relax_threshold ≥ 0. " *
               "Filename tag _stdBoost_SibGate with per-tier counts. See §3.4 (sec:std_sibgate)."
        arg_type = Bool
        required = false
        default = false
        "--allow_relax_without_pi"
        help = "Ablation escape hatch: demote the two UNSOUND-COMBINATION hard-exits " *
               "(relax_threshold > 0 with --use_perturbed_intervals=false) to warnings and run " *
               "anyway. Still a sound over-approximation (δ_relaxed ≥ δ_exact: the emitted " *
               "triangle/SibGate cuts are exact-point-valid on their own — the chord uses the " *
               "exact encoding's (l, u), and SibGate's drift interval is a computed enclosure, " *
               "not a model constraint), but without the PI coupling the relaxed copy is only " *
               "loosely tied to the exact one, so the bound may be much looser and the solve " *
               "slower. Intended ONLY for perturbed-interval ablation runs; leave false otherwise."
        arg_type = Bool
        required = false
        default = false

        # ── HAR benchmark support (pretrained tabular net) ──
        "--internet_nets_benchmarks"
        help = "Master switch for the HAR benchmark net (pretrained tabular FC net). " *
               "Off ⇒ behavior identical to today. On ⇒ enables the har dataset, its " *
               "per-coordinate input box, and the har arch; required to run har."
        arg_type = Bool
        required = false
        default = false

    end
    return parse_args(s)
end

function main()
    args = parse_commandline()
    dataset = args["dataset"]
    global model_name = args["model_name"]   # global so mip_set_attr can gate the cnn2 StartNodeLimit cap
    model_path = args["model_path"]
    perturbation = args["perturbation"]
    name_to_save = args["name_to_save"]
    use_hyper_attack = args["use_hyper_attack"]
    global Threads_num = args["Threads_num"]
    global gurobi_seed = args["gurobi_seed"]
    if gurobi_seed != 0
        name_to_save = name_to_save * "_seed" * string(gurobi_seed)
    end
    perturbation_size = parse_numbers_to_Float64(args["perturbation_size"])
    mode = args["mode"]

    # ── HAR benchmark support (behind --internet_nets_benchmarks) ──
    global internet_nets_benchmarks = args["internet_nets_benchmarks"]
    if dataset == "har" && !internet_nets_benchmarks
        error("dataset \"$dataset\" requires --internet_nets_benchmarks true")
    end
    if internet_nets_benchmarks
        (input_width, input_height, input_channels, _num_classes) = get_dataset_params(dataset)
        input_box = get_input_box(dataset, model_path, input_width, input_height, input_channels)
        input_box === nothing &&
            error("--internet_nets_benchmarks is on but no input box was found for " *
                  "model_path=\"$model_path\" (expected a <model>_box.txt sidecar).")
        global input_box_lo = input_box[1]
        global input_box_hi = input_box[2]
        # The encoders use N1's input box for both networks; error if N2's own box differs.
        _mp2 = get(args, "model_path2", "")
        if _mp2 !== nothing && _mp2 != ""
            box2 = get_input_box(dataset, _mp2, input_width, input_height, input_channels)
            if box2 !== nothing && (box2[1] != input_box_lo || box2[2] != input_box_hi)
                error("model_path2's input box differs from model_path's; the " *
                      "encoders use a single box, so the two networks must share " *
                      "one normalization.")
            end
        end
    end

    if mode == "standard"
        # Single-network verification: the VHAGaR baseline as-is, or BLEND
        # without transfer when the boost flags (zonotope / conditional
        # triangle) are on. Also runs the delta_max normalizer (--perturbation max).
        main_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    elseif mode == "advanced_standard_n1"
        # Transfer, phase 1: verify N_pre and save what BLEND's incremental
        # verification consumes — N_pre's verified per-neuron bounds, its
        # optimal solution, and the N_pre->N difference-zonotope bounds.
        main_advanced_standard_n1(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    elseif mode == "advanced_standard_n2"
        # Transfer, phase 2: BLEND with transfer — verify the revised network
        # N, reusing N_pre's saved results (bound tightening + warm start).
        main_advanced_standard_n2(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    else
        error("unknown --mode \"$mode\"; expected standard | advanced_standard_n1 | advanced_standard_n2")
    end
end

function main_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_attack)
    c_tag_list = [args["ctag"]]
    activate_vaghgar_deps = args["activate_vaghgar_deps"]
    global optimizing_intervals = args["optimizing_intervals"]
    global geometric_intervals = args["geometric_intervals"]
    if geometric_intervals && !args["use_perturbed_intervals"]
        println("WARNING: --geometric_intervals requires --use_perturbed_intervals=true (it only affects the " *
                "perturbed-interval coupling). Ignoring it and falling back to geometric_intervals=off.")
        global geometric_intervals = false
    elseif geometric_intervals && !(perturbation in ("translation", "rotation"))
        println("WARNING: --geometric_intervals only applies to translation/rotation (pixel-relocation moves); " *
                "perturbation '$(perturbation)' is unaffected. Falling back to geometric_intervals=off.")
        global geometric_intervals = false
    end

    # ── Standard-mode boosts (Boosting Standard Mode, single-network N1) ─────
    # See advstd_techniques.tex §3 (sec:std). These flags are independent of
    # the advstd transfer machinery and live in standard mode only.
    nn1_use_zono_bounds   = args["nn1_zono_bounds"]
    nn1_relax_threshold   = args["nn1_relax_threshold"]
    nn1_use_sibling_gate  = args["nn1_sibling_gate"]
    # Reuse the existing advstd dispatch in core_ops.jl::relu() and
    # n2_relax_decision.jl: relu() activates the Per-Copy Triangle Drop
    # block when adv_std_n2_relax_threshold ≥ 0, and emits the SibGate
    # state cache when adv_std_n2_sibling_gate is true. Both globals are
    # already version-aware (org/perturbation), so wiring the standard-mode
    # flags through them gives N1 the same boost without code duplication.
    global adv_std_n2_relax_threshold = nn1_relax_threshold
    global adv_std_n2_sibling_gate    = nn1_use_sibling_gate


    if nn1_use_sibling_gate && nn1_relax_threshold < 0.0
        println("WARNING: --nn1_sibling_gate=true but --nn1_relax_threshold < 0; " *
                "the sibling-gated emission rides on the per-copy decision dict. " *
                "SibGate will be inactive (no neurons relaxed).")
    end
    if (nn1_relax_threshold >= 0.0 || nn1_use_sibling_gate) && !nn1_use_zono_bounds
        println("WARNING: --nn1_relax_threshold ≥ 0 or --nn1_sibling_gate=true without " *
                "--nn1_zono_bounds=true. Per-copy bounds will only come from the encoder's " *
                "default interval arithmetic (no Source B refinement). Decision/SibGate " *
                "will still run but will see looser bounds.")
    end

    println("Standard mode: boosts enabled:")
    println("  PerturbedIntervals:                 $(args["use_perturbed_intervals"])")
    println("  Absolute Zonotope (Source B on N1): $(nn1_use_zono_bounds)")
    println("  Per-Copy Triangle Drop τ:           $(nn1_relax_threshold)")
    println("  Sibling-Gated Refinement:           $(nn1_use_sibling_gate)")

    name_to_save_init = name_to_save
    for c_tag in c_tag_list
        results.str = ""
        c_targets = parse_numbers_to_Int64(args["ct"])
        results_path = args["output_dir"]
        timout = args["timout"]
        w, h, k, c = get_dataset_params( dataset )
        token_signature = string(now().instant.periods.value)
        nn = get_nn(model_path, model_name, w, h, k, c, dataset)

        # ── Zonotope Bound Tightening (paper: System, "Zonotope Bound
        # Tightening"), once per c_tag: propagate a zonotope over the input
        # domain and perturbation space to tighten the per-neuron bounds
        # [l,u] and [l^p,u^p]. relu() intersects them via n2_abs_*_bounds
        # (core_ops.jl:222). Without transfer, the N_pre-derived bounds
        # stay empty.
        clear_n2_abs_bounds()
        if nn1_use_zono_bounds
            input_dummy = zeros(Float64, 1, w, h, k)
            p_size_b = perturbation_size[1]
            # Full 4D (1,w,h,k) per-pixel L∞ box for both single- and multi-channel
            # inputs. The old size[4]>1 branch seeded a malformed (k,1) matrix that
            # collapsed the spatial dims, so conv bound-propagation computed a
            # negative output size on multi-channel (e.g. CIFAR-10) nets.
            I_pert_up_b   = p_size_b .* ones(Float64, size(input_dummy))
            I_pert_down_b = -p_size_b .* ones(Float64, size(input_dummy))
            println("Standard-mode boost: computing absolute zonotope bounds (Source B) on N1...")
            compute_n2_bounds_zonotope_with_n1_tighten(nn, I_pert_up_b, I_pert_down_b)
            println("  Source B bounds computed: $(length(n2_abs_up_bounds)) ReLU layers")
        end

        for c_target in c_targets
            name_to_save = name_to_save_init
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

            # Conditional Triangle inputs: per-neuron perturbation-difference intervals + per-copy pre-activation bounds.
            if nn1_use_sibling_gate && nn1_relax_threshold >= 0.0
                input_dummy_s = zeros(Float64, 1, w, h, k)
                p_size_s = perturbation_size[1]
                # Full 4D per-pixel box for single- and multi-channel inputs
                # (the size[4]>1 multichannel branch was malformed — see fix above).
                I_pert_up_s   = p_size_s .* ones(Float64, size(input_dummy_s))
                I_pert_down_s = -p_size_s .* ones(Float64, size(input_dummy_s))
                compute_n2_pert_relaxation_bounds(nn, I_pert_up_s, I_pert_down_s)
            end

            # Reset the per-pair relaxation counters (reported in the filename and result line).
            clear_n2_relaxed_counters!()
            clear_sibgate_tier_counters!()
            # Conditional Triangle decision: per neuron, relax the copies whose triangle-gap area is <= tau.
            if nn1_relax_threshold >= 0.0
                compute_n2_relax_decision!(nn1_relax_threshold)
            else
                clear_n2_relax_decision!()
            end

            # Encode the two network copies (on x and on x') as a MIP, computing each neuron's bounds [l,u] along the way.
            bounds_time = @elapsed begin
                merge!(d, get_model(w, h, k, perturbation, perturbation_size, nn, zeros(Float64, 1, w, h, k), optimizer,
                get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
            end
            d[:bounds_time] = bounds_time
            m = d[:Model]

            # Add the Conditional Triangle constraints: when one copy keeps its binary, its sibling's triangle is gated on it; when both are relaxed, the two copies are tied together by their perturbation-difference interval.

            apply_sibgate_constraints!(m)

            if use_hyper_attack
                hyper_attack_hints(m, token_signature, c_tag, c_target)
                name_to_save = name_to_save*"_HyperAttackHints"
            end
            if activate_vaghgar_deps
                name_to_save = name_to_save*"_VagharDeps_depGuardFix"
                perturbation_dependencies(m, nn, perturbation, perturbation_size, w, h, k;
                                          perturbation_var=d[:Perturbation])
            end
            if args["use_perturbed_intervals"]
                name_to_save = name_to_save*"_PertruebedIntervals"
                if geometric_intervals; name_to_save = name_to_save*"_geomInt"; end
                println("Adding perturbed interval constraints...")
                perturbed_interval_constraints(m, nn, "org", "perturbation")
            end

            # ── Standard-mode boost filename tags (kept distinct from advstd
            # tags via the _stdBoost_ prefix; advstd uses _N2_advStd_…). ──────
            if nn1_use_zono_bounds || nn1_relax_threshold >= 0.0 || nn1_use_sibling_gate
                name_to_save = name_to_save * "_stdBoost"
                if nn1_use_zono_bounds
                    name_to_save = name_to_save * "_zono"
                end
                if nn1_relax_threshold >= 0.0
                    name_to_save = name_to_save * "_BTPR" * string(nn1_relax_threshold)
                end
                if nn1_use_sibling_gate
                    name_to_save = name_to_save * "_SibGate"
                end
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
            # Append per-tier neuron counts at the very end, mirroring the
            # advstd _both/_orgDrop/_pertDrop convention so result-file
            # name matching can strip them with a single regex.
            if nn1_use_sibling_gate && nn1_relax_threshold >= 0.0
                name_to_save = name_to_save *
                               "_both"    * string(n_sibgate_both_thin) *
                               "_orgDrop" * string(n_sibgate_one_thin_org_dropped) *
                               "_pertDrop" * string(n_sibgate_one_thin_pert_dropped)
            end
            save_results(results_path, model_name, perturbation, perturbation_size, results.str, d, nn, c_tag-1, c_target-1, w, h, k,name_to_save*"_cTag"*string(c_tag),token_signature)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════
# Advanced Standard Mode (transfer)
# ═══════════════════════════════════════════════════════════════════════════
"""
    advstd_results_complete(results_path, n2_name_suffix, c_tag, c_targets) -> Bool

Check whether a completed results file already exists in `results_path` that
contains results for ALL requested (c_tag, c_target) pairs.  The file must
contain a `c_source=<c_tag-1>,c_target=<ct-1>` line for every c_target in
`c_targets` (skipping c_target == c_tag).  Returns true only when every
expected line is present.
"""
function advstd_results_complete(results_path::AbstractString, n2_name_suffix::AbstractString,
                                  c_tag::Int, c_targets::Vector{Int})
    if !isdir(results_path)
        return false
    end
    # Expected c_target set (0-indexed, excluding self)
    expected = Set(ct - 1 for ct in c_targets if ct != c_tag)
    if isempty(expected)
        return true
    end
    # Search for any .txt file whose name contains the n2_name suffix and the cTag marker.
    # Strip historical _elimOrg{N}_elimPert{N} counts before matching: the
    # retired N1-probe (Source C) injected them between the tech flags and
    # _HyperAttackHints in old result files, so those raw filenames do not
    # contain n2_name_suffix as a contiguous substring.
    ctag_marker = "_cTag" * string(c_tag)
    elim_re = r"_elimOrg\d+_elimPert\d+"
    for fname in readdir(results_path)
        if !endswith(fname, ".txt")
            continue
        end
        fname_for_match = replace(fname, elim_re => "")
        if !occursin(n2_name_suffix, fname_for_match) || !occursin(ctag_marker, fname_for_match)
            continue
        end
        # Parse the file: collect all (c_source, c_target) pairs present
        fpath = joinpath(results_path, fname)
        found = Set{Int}()
        for line in eachline(fpath)
            m_src = match(r"c_source=(\d+)", line)
            m_tgt = match(r"c_target=(\d+)", line)
            if m_src !== nothing && m_tgt !== nothing
                src = parse(Int, m_src.captures[1])
                tgt = parse(Int, m_tgt.captures[1])
                if src == c_tag - 1
                    push!(found, tgt)
                end
            end
        end
        if expected ⊆ found
            println("advstd_results_complete: found complete results in $fname — skipping run")
            return true
        end
    end
    return false
end

"""
    advstd_result_exists_for_pair(results_path, n2_name_suffix, c_tag, c_target) -> Bool

Check whether a results file in `results_path` already contains a completed
entry for a single (c_tag, c_target) pair.  Returns true if a matching
`c_source=<c_tag-1>,c_target=<c_target-1>` line is found.
"""
function advstd_result_exists_for_pair(results_path::AbstractString, n2_name_suffix::AbstractString,
                                       c_tag::Int, c_target::Int)
    if !isdir(results_path)
        return false
    end
    expected_src = c_tag - 1
    expected_tgt = c_target - 1
    ctag_marker = "_cTag" * string(c_tag)
    elim_re = r"_elimOrg\d+_elimPert\d+"
    for fname in readdir(results_path)
        if !endswith(fname, ".txt")
            continue
        end
        fname_for_match = replace(fname, elim_re => "")
        if !occursin(n2_name_suffix, fname_for_match) || !occursin(ctag_marker, fname_for_match)
            continue
        end
        fpath = joinpath(results_path, fname)
        for line in eachline(fpath)
            m_src = match(r"c_source=(\d+)", line)
            m_tgt = match(r"c_target=(\d+)", line)
            if m_src !== nothing && m_tgt !== nothing
                if parse(Int, m_src.captures[1]) == expected_src && parse(Int, m_tgt.captures[1]) == expected_tgt
                    return true
                end
            end
        end
    end
    return false
end

"""
    advstd_read_result_line(results_path, n2_name_suffix, c_tag, c_target) -> Union{String, Nothing}

Read and return the full result line for a specific (c_tag, c_target) pair from
an existing results file.  Returns nothing if not found.
"""
function advstd_read_result_line(results_path::AbstractString, n2_name_suffix::AbstractString,
                                  c_tag::Int, c_target::Int)
    if !isdir(results_path)
        return nothing
    end
    expected_src = string(c_tag - 1)
    expected_tgt = string(c_target - 1)
    ctag_marker = "_cTag" * string(c_tag)
    elim_re = r"_elimOrg\d+_elimPert\d+"
    for fname in readdir(results_path)
        if !endswith(fname, ".txt")
            continue
        end
        fname_for_match = replace(fname, elim_re => "")
        if !occursin(n2_name_suffix, fname_for_match) || !occursin(ctag_marker, fname_for_match)
            continue
        end
        fpath = joinpath(results_path, fname)
        for line in eachline(fpath)
            m_src = match(r"c_source=(\d+)", line)
            m_tgt = match(r"c_target=(\d+)", line)
            if m_src !== nothing && m_tgt !== nothing
                if m_src.captures[1] == expected_src && m_tgt.captures[1] == expected_tgt
                    return strip(line)
                end
            end
        end
    end
    return nothing
end

function main_advanced_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_attack)
    model_path2 = args["model_path2"]
    if model_path2 == "" || model_path2 == model_path
        error("advanced_standard mode requires --model_path2 pointing to N2 (different from N1)")
    end

    # Warm Start: hint each unstable neuron's phase from N_pre's solution, kept only where it agrees with the attack's hints (off | prev_pgd).
    var_hint_mode = parse_var_hint_mode(args["adv_std_var_hint"])
    # Transfer bound tightening: intersect N's per-neuron bounds with N_pre's verified bounds shifted by the difference bounds.
    use_bound_tightening = args["adv_std_bound_tightening"]
    # Zonotope Bound Tightening: propagate a zonotope through N to tighten the per-neuron bounds [l,u] and [l^p,u^p].
    use_zono_bounds = args["adv_std_zono_bounds"]
    # Ablation flag (zono_npre).
    zono_use_npre   = args["adv_std_zono_npre"]
    # Conditional Triangle: relax a copy's ReLU when its triangle-gap area is <= tau (read by relu() via this global).
    global adv_std_n2_relax_threshold = args["adv_std_n2_relax_threshold"]
    # Conditional Triangle emission: gate the triangle on the sibling copy's binary / couple both relaxed copies.
    global adv_std_n2_sibling_gate = args["adv_std_n2_sibling_gate"]
    if adv_std_n2_sibling_gate && adv_std_n2_relax_threshold < 0.0
        println("WARNING: --adv_std_n2_sibling_gate=true but --adv_std_n2_relax_threshold < 0; " *
                "Technique 4 inherits Technique 3's tiered decision rule. " *
                "SibGate will be inactive (no neurons relaxed).")
    end
    if adv_std_n2_relax_threshold >= 0.0 && !use_bound_tightening
        println("WARNING: --adv_std_n2_relax_threshold >= 0 but --adv_std_bound_tightening is " *
                "false. The relaxation block needs n1_neuron_bounds to be populated, which only " *
                "happens when bound_tightening is true — no neurons will be relaxed in this run.")
    end
    
    # Folder with N_pre's saved verification results (verified bounds, optimal solution, difference bounds) that transfer reuses.
    n1_state_dir = args["n1_state_dir"]

    c_tag_list = [args["ctag"]]
    # VHAGaR's dependency constraints between the two copies (equal / monotonic / anti-monotonic neurons).
    activate_vaghgar_deps = args["activate_vaghgar_deps"]
    # Tighter ReLU clipping inside the perturbation-difference interval propagation (uses each copy's stability).
    global optimizing_intervals = args["optimizing_intervals"]
    # Translation/rotation only: fold the exact (T-I) relocation map through the first layer instead of a per-pixel envelope.
    global geometric_intervals = args["geometric_intervals"]
    if geometric_intervals && !args["use_perturbed_intervals"]
        println("WARNING: --geometric_intervals requires --use_perturbed_intervals=true (it only affects the " *
                "perturbed-interval coupling). Ignoring it and falling back to geometric_intervals=off.")
        global geometric_intervals = false
    elseif geometric_intervals && !(perturbation in ("translation", "rotation"))
        println("WARNING: --geometric_intervals only applies to translation/rotation (pixel-relocation moves); " *
                "perturbation '$(perturbation)' is unaffected. Falling back to geometric_intervals=off.")
        global geometric_intervals = false
    end
    name_to_save_init = name_to_save

    println("Advanced-standard mode: techniques enabled:")
    println("  Technique 4 (Bound Tightening):    $(use_bound_tightening)")
    println("  Technique 4+ (Zono Bounds):        $(use_zono_bounds)")
    println("  Zono uses N_pre (Source A):        $(zono_use_npre)")
    println("  Technique 5 (Variable Hints):      $(var_hint_mode_label(var_hint_mode))")
    println("  Technique 6 (N2 Relax Threshold):  $(adv_std_n2_relax_threshold)")

    # Build the filename signature of this exact configuration (one tag per active technique); results whose
    # filename carries this signature were produced by the same configuration, so they can be skipped below.
    n2_check = name_to_save
    if occursin("_N2_advStd", n2_check)
        # already has technique flags
    else
        n2_check = n2_check * "_N2_advStd"
        # Technique 6 (BoundTightPertRelax) subsumes bound-tightening — when
        # τ ≥ 0 we emit _BoundTightPertRelax{τ} instead of _boundTight since
        # the relaxation logically depends on bound-tightening being active.
        if use_bound_tightening
            if args["adv_std_n2_relax_threshold"] >= 0.0
                n2_check = n2_check * "_BoundTightPertRelax" * string(args["adv_std_n2_relax_threshold"])
            else
                n2_check = n2_check * "_boundTight"
            end
        end
        if args["adv_std_n2_sibling_gate"]; n2_check = n2_check * "_SibGate"; end
        if use_zono_bounds;           n2_check = n2_check * "_zonoBounds"; end
        if use_zono_bounds && !zono_use_npre; n2_check = n2_check * "_noNpreZono"; end
        if var_hint_mode == VH_PREV_PGD;   n2_check = n2_check * "_varHintPrevPGD";   end
    end
    if use_hyper_attack; n2_check = n2_check * "_HyperAttackHints"; end
    if activate_vaghgar_deps;              n2_check = n2_check * "_VagharDeps_depGuardFix"; end  # must mirror the saved name's tag (see line ~666)
    if args["use_perturbed_intervals"]
        n2_check = n2_check * "_PerturbedIntervals"
    end
    if geometric_intervals;                 n2_check = n2_check * "_geomInt"; end

    # Check if ALL results already exist — if so, skip entirely
    for c_tag in c_tag_list
        c_targets = parse_numbers_to_Int64(args["ct"])
        results_path = args["output_dir"]
        if advstd_results_complete(results_path, n2_check, c_tag, c_targets)
            println("All results already present for c_tag=$c_tag — nothing to do.")
            return
        end
    end

    for c_tag in c_tag_list
        results.str = ""
        results_n2 = Results("")
        c_targets = parse_numbers_to_Int64(args["ct"])
        results_path = args["output_dir"]
        timout = args["timout"]
        w, h, k, c = get_dataset_params(dataset)
        token_signature = string(now().instant.periods.value)

        # Phase 2 always loads N1's state from disk; fail loudly if the state
        # dir vanished between the _n2 wrapper's check and here.
        (n1_state_dir != "" && isdir(n1_state_dir)) ||
            error("advanced_standard_n2 requires --n1_state_dir pointing to saved N1 state")
        nn2 = get_nn(model_path2, model_name, w, h, k, c, dataset)

        # Transfer bound tightening: load N_pre's difference bounds [d_lo, d_hi] and intersect N's zonotope bounds with N_pre's bounds shifted by them.
        clear_n2_abs_bounds()
        if use_bound_tightening
            input_dummy = zeros(Float64, 1, w, h, k)
            p_size = perturbation_size[1]
            I_pert_up_init = p_size .* ones(Float64, size(input_dummy))
            I_pert_down_init = -p_size .* ones(Float64, size(input_dummy))

            # Load the drift information — how far N can stray from N_pre ([d_lo, d_hi] + N_pre's pre-activation bounds); shared by all class pairs, shifts every reused N_pre quantity below.
            load_n1_diff_bounds!(n1_state_dir; require_preact=use_zono_bounds)

            # Zonotope Bound Tightening: propagate a zonotope through N, intersecting its bounds at each layer with N_pre's bounds shifted by the difference bounds.
            if use_zono_bounds
                println("Advanced-standard: computing N1-tightened absolute N2 zonotope (Source B)...")
                compute_n2_bounds_zonotope_with_n1_tighten(nn2, I_pert_up_init, I_pert_down_init;
                                                           use_n1_tighten=zono_use_npre)
            end

        end

        for c_target in c_targets
            name_to_save = name_to_save_init
            if c_tag == c_target
                continue
            end

            # ── Per-c_target skip: avoid re-solving if this pair already exists ──
            if advstd_result_exists_for_pair(results_path, n2_check, c_tag, c_target)
                println("Result already exists for c_tag=$c_tag, c_target=$c_target — skipping")
                # Preserve the existing result line in results_n2.str so
                # save_results (which overwrites the file) doesn't lose it.
                existing_line = advstd_read_result_line(results_path, n2_check, c_tag, c_target)
                if existing_line !== nothing
                    results_n2.str = results_n2.str * existing_line * "\n"
                end
                continue
            end

            # Load what N_pre proved and chose on this class pair — its verified per-neuron bounds (for bound tightening) and its optimal solution (for the warm-start hints).
            println("\n══ Advanced-standard: loading N1 state from $n1_state_dir (c_tag=$c_tag, c_target=$c_target) ══")
            n1_var_names, n1_var_values, n1_layers_info =
                load_n1_state(n1_state_dir, c_tag, c_target)

            # ═══ PASS 2: Solve N2 (accelerated standard) ═══════════════════
            println("\n══ Advanced-standard PASS 2: solving N2 with N1 info (c_tag=$c_tag, c_target=$c_target) ══")

            # Run hyper_attack for N2: the suboptimal solution supplies the
            # Gurobi Cutoff for branch pruning, and its per-neuron phase
            # hints are applied below via hyper_attack_hints.
            suboptimal_solution_n2, suboptimal_time_n2 = 0, 0
            if use_hyper_attack
                suboptimal_solution_n2, suboptimal_time_n2 = hyper_attack(dataset, c_tag, c_target, token_signature * "_n2", model_name, model_path2, perturbation, perturbation_size; force_cpu=args["force_cpu"])
            end
            d_n2 = Dict()
            d_n2[:TargetIndex] = get_target_indexes(c_target, c)
            d_n2[:SourceIndex] = get_target_indexes(c_tag, c)
            d_n2[:suboptimal_solution] = suboptimal_solution_n2
            d_n2[:suboptimal_time] = suboptimal_time_n2
            d_n2[:adv_std_flags] = (
                adv_std_bound_tightening     = args["adv_std_bound_tightening"],
                adv_std_zono_bounds          = args["adv_std_zono_bounds"],
                adv_std_n2_relax_threshold   = args["adv_std_n2_relax_threshold"],
                adv_std_var_hint             = var_hint_mode_label(var_hint_mode),   # "off" | "prev_pgd"
                gurobi_seed                  = args["gurobi_seed"],
            )
            optimizer = Gurobi.Optimizer
            mip_reset()

            # Transfer bound tightening: expose N_pre's verified per-neuron bounds so the encoder intersects each of N's bounds with them, shifted by the difference bounds.
            if use_bound_tightening
                set_n1_neuron_bounds(n1_layers_info)
            end

            # Reset the per-pair relaxation counters (reported in the filename and result line).
            clear_n2_relaxed_counters!()
            clear_sibgate_tier_counters!()

            # Conditional Triangle inputs: (compute) per-neuron perturbation-difference intervals + per-copy pre-activation bounds.
            if adv_std_n2_sibling_gate && adv_std_n2_relax_threshold >= 0.0 && use_bound_tightening
                compute_n2_pert_relaxation_bounds(nn2, I_pert_up_init, I_pert_down_init)
            end

            # Conditional Triangle decision: per neuron, relax the copies whose triangle-gap area (on the tightened bounds) is <= tau.
            if adv_std_n2_relax_threshold >= 0.0 && use_bound_tightening
                compute_n2_relax_decision!(adv_std_n2_relax_threshold)
            else
                clear_n2_relax_decision!()
            end

            bounds_time_n2 = @elapsed begin
                merge!(d_n2, get_model(w, h, k, perturbation, perturbation_size, nn2, zeros(Float64, 1, w, h, k), optimizer,
                get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
            end
            d_n2[:bounds_time] = bounds_time_n2
            m_n2 = d_n2[:Model]

            # Add the Conditional Triangle constraints: when one copy keeps its binary, its sibling's triangle is gated on it; when both are relaxed, the two copies are tied by their pre-activation difference interval.
            apply_sibgate_constraints!(m_n2)

            # Build N2 result filename with active technique flags
            # If name_to_save already has technique flags (set by sweep), use as-is
            if occursin("_N2_advStd", name_to_save)
                n2_name = name_to_save
            else
                n2_name = name_to_save * "_N2_advStd"
                # See n2_check builder above: _BoundTightPertRelax subsumes _boundTight.
                if use_bound_tightening
                    if args["adv_std_n2_relax_threshold"] >= 0.0
                        n2_name = n2_name * "_BoundTightPertRelax" * string(args["adv_std_n2_relax_threshold"])
                    else
                        n2_name = n2_name * "_boundTight"
                    end
                end
                if args["adv_std_n2_sibling_gate"]; n2_name = n2_name * "_SibGate"; end
                if use_zono_bounds;           n2_name = n2_name * "_zonoBounds"; end
                if use_zono_bounds && !zono_use_npre; n2_name = n2_name * "_noNpreZono"; end
                if var_hint_mode == VH_PREV_PGD;   n2_name = n2_name * "_varHintPrevPGD";   end
            end
            # Record in the filename how many neurons the Conditional Triangle relaxed: both copies (_both), only the clean copy (_orgDrop), only the perturbed copy (_pertDrop).
            if args["adv_std_n2_sibling_gate"]
                n2_name = n2_name *
                          "_both"     * string(n_sibgate_both_thin) *
                          "_orgDrop"  * string(n_sibgate_one_thin_org_dropped) *
                          "_pertDrop" * string(n_sibgate_one_thin_pert_dropped)
            end

            if use_hyper_attack
                hyper_attack_hints(m_n2, token_signature * "_n2", c_tag, c_target)
                n2_name = n2_name * "_HyperAttackHints"
            end
            if activate_vaghgar_deps
                n2_name = n2_name * "_VagharDeps_depGuardFix"
                perturbation_dependencies(m_n2, nn2, perturbation, perturbation_size, w, h, k;
                                          perturbation_var=d_n2[:Perturbation])
            end
            if args["use_perturbed_intervals"]
                n2_name = n2_name * "_PerturbedIntervals"
                if geometric_intervals; n2_name = n2_name * "_geomInt"; end
                perturbed_interval_constraints(m_n2, nn2, "org", "perturbation")
            end
            mip_set_delta_property(m_n2, perturbation, d_n2)
            set_optimizer(m_n2, optimizer)
            mip_set_attr(m_n2, perturbation, d_n2, timout)
            MOI.set(m_n2, Gurobi.CallbackFunction(), my_callback)
            # Warm Start: hint each unstable neuron's phase from N_pre's solution shifted by the difference bounds, kept only where it agrees with the attack's hints (which must already be set — hence after hyper_attack_hints).
            if var_hint_mode != VH_OFF
                apply_n1_var_hints!(m_n2, var_hint_mode, n1_var_names, n1_var_values, n1_layers_info)
            end
            optimize!(m_n2)
            mip_log(m_n2, d_n2)

            # Save N2 results (no mip_reuse_bounds — interleaved N1/N2 prevents safe reuse)
            clear_n1_neuron_bounds()
            results_n2.str = update_results_str(results_n2.str, c_tag, c_target, d_n2)
            save_results(results_path, model_name, perturbation, perturbation_size, results_n2.str, d_n2, nn2, c_tag-1, c_target-1, w, h, k, n2_name*"_cTag"*string(c_tag), token_signature * "_n2")
        end
    end
end

# Transfer, phase 1: verify N_pre once and save its results (verified bounds, optimal solution, difference bounds); advanced_standard_n2 runs then reuse them.
function main_advanced_standard_n1(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_attack)
    model_path2 = args["model_path2"]
    n1_state_dir = args["n1_state_dir"]
    if n1_state_dir == ""
        error("advanced_standard_n1 requires --n1_state_dir (where to save N1 state)")
    end

    c_tag_list = [args["ctag"]]
    activate_vaghgar_deps = args["activate_vaghgar_deps"]
    global optimizing_intervals = args["optimizing_intervals"]
    global geometric_intervals = args["geometric_intervals"]
    if geometric_intervals && !args["use_perturbed_intervals"]
        println("WARNING: --geometric_intervals requires --use_perturbed_intervals=true (it only affects the " *
                "perturbed-interval coupling). Ignoring it and falling back to geometric_intervals=off.")
        global geometric_intervals = false
    elseif geometric_intervals && !(perturbation in ("translation", "rotation"))
        println("WARNING: --geometric_intervals only applies to translation/rotation (pixel-relocation moves); " *
                "perturbation '$(perturbation)' is unaffected. Falling back to geometric_intervals=off.")
        global geometric_intervals = false
    end

    for c_tag in c_tag_list
        c_targets = parse_numbers_to_Int64(args["ct"])
        timout = args["timout"]
        w, h, k, c = get_dataset_params(dataset)
        token_signature = string(now().instant.periods.value)

        nn1 = get_nn(model_path, model_name, w, h, k, c, dataset)

        # Compute and save diff bounds (needs both nn1 and nn2)
        if model_path2 != ""
            nn2 = get_nn(model_path2, model_name, w, h, k, c, dataset)
            input_dummy = zeros(Float64, 1, w, h, k)
            p_size = perturbation_size[1]
            I_pert_up_init = p_size .* ones(Float64, size(input_dummy))
            I_pert_down_init = -p_size .* ones(Float64, size(input_dummy))
            # Difference zonotope: bound, per neuron, how far N's pre-activation can stray from N_pre's ([d_lo, d_hi]).
            println("Advanced-standard-N1: computing zonotope diff bounds between N1 and N2...")
            compute_diff_bounds_zonotope(nn1, nn2, I_pert_up_init, I_pert_down_init; optimizing_intervals=optimizing_intervals)
            # If this folder already has saved difference bounds (from an earlier, interrupted run), keep them:
            # recomputing gives the same numbers, and all class pairs in one folder must use the same bounds.
            diff_bounds_path = joinpath(n1_state_dir, "diff_bounds.bin")
            preact_path = joinpath(n1_state_dir, "n1_preact_bounds.bin")
            if !isfile(diff_bounds_path)
                save_n1_diff_bounds(n1_state_dir)
            elseif !isfile(preact_path) && !isempty(n1_preact_up_bounds)
                # Old folder from before N_pre's pre-activation bounds were saved: add just that missing file, keep the existing difference bounds.
                serialize(preact_path, (n1_preact_up_bounds, n1_preact_down_bounds))
                println("Advanced-standard-N1: preserved existing diff_bounds.bin; wrote new n1_preact_bounds.bin to $n1_state_dir")
            else
                println("Advanced-standard-N1: preserved existing diff_bounds.bin + n1_preact_bounds.bin at $n1_state_dir (partial-completion mode)")
            end
        end

        for c_target in c_targets
            if c_tag == c_target
                continue
            end

            println("\n══ Advanced-standard-N1: solving N1 (c_tag=$c_tag, c_target=$c_target) ══")
            suboptimal_solution_n1, suboptimal_time_n1 = 0, 0
            if use_hyper_attack
                suboptimal_solution_n1, suboptimal_time_n1 = hyper_attack(dataset, c_tag, c_target, token_signature, model_name, model_path, perturbation, perturbation_size; force_cpu=args["force_cpu"])
            end
            optimizer = Gurobi.Optimizer
            d_n1 = Dict()
            d_n1[:TargetIndex] = get_target_indexes(c_target, c)
            d_n1[:SourceIndex] = get_target_indexes(c_tag, c)
            d_n1[:suboptimal_solution] = suboptimal_solution_n1
            d_n1[:suboptimal_time] = suboptimal_time_n1
            mip_reset()
            clear_n1_neuron_bounds()
            bounds_time_n1 = @elapsed begin
                merge!(d_n1, get_model(w, h, k, perturbation, perturbation_size, nn1, zeros(Float64, 1, w, h, k), optimizer,
                get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
            end
            d_n1[:bounds_time] = bounds_time_n1
            m_n1 = d_n1[:Model]
            if use_hyper_attack
                hyper_attack_hints(m_n1, token_signature, c_tag, c_target)
            end
            if activate_vaghgar_deps
                perturbation_dependencies(m_n1, nn1, perturbation, perturbation_size, w, h, k;
                                          perturbation_var=d_n1[:Perturbation])
            end
            if args["use_perturbed_intervals"]
                perturbed_interval_constraints(m_n1, nn1, "org", "perturbation")
            end
            mip_set_delta_property(m_n1, perturbation, d_n1)
            set_optimizer(m_n1, optimizer)
            mip_set_attr(m_n1, perturbation, d_n1, timout)
            MOI.set(m_n1, Gurobi.CallbackFunction(), my_callback)
            optimize!(m_n1)
            mip_log(m_n1, d_n1)

            # Extract ALL solver info and save to disk
            n1_var_names, n1_var_values = extract_all_variable_values(m_n1)
            n1_layers_info = deepcopy(layers_info_dict)
            save_n1_state(n1_state_dir, c_tag, c_target, n1_var_names, n1_var_values, n1_layers_info)

            mip_reuse_bounds()
            m_n1 = nothing
        end
    end
    println("\n══ Advanced-standard-N1: done. State saved to $n1_state_dir ══")
end

# ═══════════════════════════════════════════════════════════════════════════
# Advanced Standard N2: load N1 state from disk and solve N2.
# Equivalent to advanced_standard --n1_state_dir, but as a dedicated mode.
# ═══════════════════════════════════════════════════════════════════════════
function main_advanced_standard_n2(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_attack)
    n1_state_dir = args["n1_state_dir"]
    if n1_state_dir == "" || !isdir(n1_state_dir)
        error("advanced_standard_n2 requires --n1_state_dir pointing to saved N1 state")
    end
    # Delegate to main_advanced_standard which already handles --n1_state_dir
    main_advanced_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_attack)
end


main()