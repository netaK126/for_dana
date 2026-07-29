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
include("utils/n1_probe_lp.jl")
include("utils/n2_relax_decision.jl")
include("utils/sibgate_emit.jl")

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--dataset", "-d"
        help = "mnist, fmnist, cifar10, or (with --internet_nets_benchmarks) acas, har"
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
        help = "standard, advanced_standard, advanced_standard_n1, or advanced_standard_n2"
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
               "(correct per-pixel envelope + first-layer composition of the move). Interval-only, no zonotope. " *
               "Sound: delta unchanged. Default false = today's behavior."
        arg_type = Bool
        required = false
        default = false
        "--activate_vaghgar_deps"
        help = "activate  vaghgar depandencies"
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
        help = "advanced_standard variable-hint mode (Technique 5): " *
               "off | prev | direct | direct_pgd | prev_pgd. " *
               "'prev' uses the previous §4.3 rule (shift N1's pre-activation by the diff " *
               "bound, clip to [l_n2, u_n2], take p from interval-length ratios). " *
               "'direct' derives p directly from Tech 2's tightened [l_n2, u_n2], without " *
               "re-introducing the ẑ^N1 midpoint proxy or diff-width noise. " *
               "'direct_pgd' computes p the same way as 'direct', but routes hint_val into " *
               "set_start_value() (Gurobi Start) with PGD consensus filtering: fill where PGD " *
               "is silent, leave where PGD agrees, withdraw where PGD disagrees. VarHintVal/" *
               "VarHintPri are NOT set under this mode. " *
               "'prev_pgd' is the same Start-consensus routing applied to 'prev's p_i. " *
               "Legacy 'true'/'false' still accepted (map to 'prev'/'off') with a deprecation warning."
        arg_type = String
        required = false
        default = "off"
        range_tester = x -> lowercase(strip(String(x))) in
                             ("off", "prev", "direct", "direct_pgd", "prev_pgd", "true", "false")
        "--adv_std_bound_tightening"
        help = "advanced_standard: tighten N2 bounds using N1 + compute_diff_and_comp_bounds (Technique 4)"
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
        "--adv_std_n1_probe"
        help = "advanced_standard: run post-Phase-1 LP probing against a joint N1+N2 LP " *
               "relaxation (triangle ReLU relaxations) to derive tighter per-neuron N2 " *
               "pre-activation bounds, eliminating more N2 binaries via the existing stable-flip " *
               "short-circuit. Values: off | lp. Requires --adv_std_bound_tightening true. " *
               "linf perturbation only in the first implementation; other perturbations skip with a warning."
        arg_type = String
        required = false
        default = "off"
        range_tester = x -> x in ("off", "lp")
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

        # ── ACAS/HAR benchmark support (pretrained tabular nets) ──
        "--internet_nets_benchmarks"
        help = "Master switch for the ACAS Xu / HAR benchmark nets (pretrained tabular FC nets " *
               "from the TwoSafe paper). Off ⇒ behavior identical to today. On ⇒ enables the acas/har " *
               "datasets, their per-coordinate input box, and the new archs; required to run acas/har."
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

    # ── ACAS/HAR benchmark support (behind --internet_nets_benchmarks) ──
    global internet_nets_benchmarks = args["internet_nets_benchmarks"]
    if dataset in ("acas", "har") && !internet_nets_benchmarks
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
        # N2 gets encoded over N1's box. That is right only if the two nets share
        # a normalization; assert it rather than silently reusing N1's.
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
        main_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    elseif mode == "advanced_standard"
        main_advanced_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    elseif mode == "advanced_standard_n1"
        main_advanced_standard_n1(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    elseif mode == "advanced_standard_n2"
        main_advanced_standard_n2(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save,use_hyper_attack)
    else
        error("unknown --mode \"$mode\"; expected standard | advanced_standard | advanced_standard_n1 | advanced_standard_n2")
    end
end

function main_standard(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_attack)
    c_tag_list = [args["ctag"]]
    activate_vaghgar_deps = args["activate_vaghgar_deps"]
    global use_relaxations = args["use_relaxations"]
    global relaxation_threshold = args["relaxation_threshold"]
    global optimizing_intervals = args["optimizing_intervals"]
    global relaxation_gap_area = args["relaxation_gap_area"]
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

    # Unsound-combination guard (mirrors run.jl:693 for advstd). The Per-Copy
    # Triangle Drop relies on perturbed-interval coupling to keep the relaxed
    # copy linked to its exact sibling; without it the LP can drift and
    # δ_exact is not guaranteed. τ = 0.0 emits no triangle (gap > 0 for every
    # split neuron), so the guard only fires for τ > 0.
    if nn1_relax_threshold > 0.0 && !args["use_perturbed_intervals"]
        if args["allow_relax_without_pi"]
            # Ablation mode: the emitted cuts are exact-point-valid on their own
            # (chord uses the exact encoding's (l, u); SibGate's drift interval is
            # a computed enclosure, not a model constraint), so δ_relaxed ≥ δ_exact
            # still holds — only tightness/speed degrade. See --allow_relax_without_pi.
            println("WARNING (--allow_relax_without_pi): running --nn1_relax_threshold > 0 " *
                    "WITHOUT --use_perturbed_intervals. Sound (δ_relaxed ≥ δ_exact) but the " *
                    "relaxed copy is only loosely tied to the exact one — expect a looser " *
                    "bound and a slower solve. Perturbed-interval ablation runs only.")
        else
            println("UNSOUND COMBINATION: --nn1_relax_threshold > 0 (Per-Copy Triangle Drop, " *
                    "standard-mode boost §3.3) requires --use_perturbed_intervals=true for soundness. " *
                    "Without the perturbed-interval coupling constraints the relaxed and exact " *
                    "copies can drift apart and δ_exact is not guaranteed. Exiting without " *
                    "encoding or optimizing. Set --use_perturbed_intervals true, or disable " *
                    "the relaxation with --nn1_relax_threshold off (or use τ = 0), or pass " *
                    "--allow_relax_without_pi true for a perturbed-interval ablation run.")
            return
        end
    end
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

        # ── Source B (absolute zonotope on N1) precompute, once per c_tag ──
        # n2_abs_*_bounds is the global the encoder's intersect_per_copy_bounds
        # consumes (see core_ops.jl:222). In standard mode we leave Source A
        # (n1_neuron_bounds / relu_diff_*) empty so only Source B intersects.
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
            global use_zonotope = true
            println("Standard-mode boost: computing absolute zonotope bounds (Source B) on N1...")
            compute_n2_bounds_zonotope_with_n1_tighten(nn, I_pert_up_b, I_pert_down_b)
            println("  Source B bounds computed: $(length(n2_abs_up_bounds)) ReLU layers")
        end

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

            # ── SibGate prerequisite: org-pert pre-activation diff bounds through N1.
            # compute_n2_pert_relaxation_bounds populates relu_n2pert_*_bounds
            # and n2_preact_*_bounds. The encoder side of SibGate (sibgate_emit.jl)
            # walks these arrays to build the conditional triangle / coupling line.
            # In standard mode the "second network" passed to this routine is N1 itself.
            if nn1_use_sibling_gate && nn1_relax_threshold >= 0.0
                input_dummy_s = zeros(Float64, 1, w, h, k)
                p_size_s = perturbation_size[1]
                # Full 4D per-pixel box for single- and multi-channel inputs
                # (the size[4]>1 multichannel branch was malformed — see fix above).
                I_pert_up_s   = p_size_s .* ones(Float64, size(input_dummy_s))
                I_pert_down_s = -p_size_s .* ones(Float64, size(input_dummy_s))
                compute_n2_pert_relaxation_bounds(nn, I_pert_up_s, I_pert_down_s)
            end

            # ── Per-Copy Triangle Drop decision dict (Tech 6 dispatch) ────────
            # Populated from Source B alone in standard mode (Source A absent).
            clear_n2_relaxed_counters!()
            clear_sibgate_tier_counters!()
            if nn1_relax_threshold >= 0.0
                compute_n2_relax_decision!(nn1_relax_threshold)
            else
                clear_n2_relax_decision!()
            end

            bounds_time = @elapsed begin
                merge!(d, get_model(w, h, k, perturbation, perturbation_size, nn, zeros(Float64, 1, w, h, k), optimizer,
                get_default_tightening_options(optimizer), DEFAULT_TIGHTENING_ALGORITHM))
            end
            d[:bounds_time] = bounds_time
            m = d[:Model]

            # ── SibGate emission (post-encoding pass) ────────────────────────
            # apply_sibgate_constraints! walks n2_relax_decision + n2_relu_state
            # cache and emits the conditional triangles / coupling lines.
            # No-op when adv_std_n2_sibling_gate=false.
            apply_sibgate_constraints!(m)

            if use_hyper_attack
                hyper_attack_hints(m, token_signature, c_tag, c_target)
                name_to_save = name_to_save*"_HyperAttackHints"
            end
            if activate_vaghgar_deps
                # _depGuardFix marks the fixed dep encoder (has_a_o && has_a_p
                # guard); the paper-tables pipeline keeps binary-dropping runs
                # only when this tag is present, so it MUST follow _VagharDeps.
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
            if args["use_relaxations"]
                name_to_save = name_to_save*"_Relaxations"*string(args["relaxation_threshold"])
                if args["relaxation_gap_area"]
                    name_to_save = name_to_save*"_GapArea"
                end
                println("Applying conditional triangle relaxations with threshold $(args["relaxation_threshold"]) (gap_area=$(args["relaxation_gap_area"]))...")
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
            if args["use_relaxations"]
                name_to_save = name_to_save * "_RelaxCount" * string(relaxation_condition_count)
            end
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
# Advanced Standard Mode
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
    # Strip the probe's _elimOrg{N}_elimPert{N} counts before matching: when
    # adv_std_n1_probe=lp those counts are injected between the tech flags and
    # _HyperAttackHints, so the raw filename no longer contains n2_name_suffix
    # as a contiguous substring. n2_name_suffix itself never has elim counts.
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

    # Technique 5: 5-valued mode (off/prev/direct/direct_pgd/prev_pgd); see parse_var_hint_mode.
    var_hint_mode = parse_var_hint_mode(args["adv_std_var_hint"])
    use_bound_tightening = args["adv_std_bound_tightening"]
    use_zono_bounds = args["adv_std_zono_bounds"]
    zono_use_npre   = args["adv_std_zono_npre"]
    # Technique 6: N1-gated N2/N2p triangle LP relaxation. Propagated to the
    # core_ops.jl::relu() consumer via the `adv_std_n2_relax_threshold` global.
    global adv_std_n2_relax_threshold = args["adv_std_n2_relax_threshold"]
    # Technique 4 (SibGate) — read by core_ops.jl::relu() to switch from
    # the simple Wong-Kolter triangle (Technique 3) to the conditional
    # triangle gated on the sibling binary (one-thin tier) and the
    # pre-activation coupling line (both-thin tier).
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
    # Unsound-combination guard: Technique 4 (BoundTightPertRelax) relies on the
    # perturbed-interval coupling constraints to link a relaxed copy's post-ReLU
    # value to the exact one (see §4.4 of advstd_techniques.tex). Without those
    # constraints the two copies can drift apart in the joint feasible set and
    # δ_exact is no longer guaranteed to be reached. Refuse to run rather than
    # silently produce unsound results.
    #
    # τ == 0.0 is treated as "effectively disabled" and allowed through: the gap
    # formula u·|l| / (2(u-l)) is strictly positive for every split neuron
    # (l < 0 < u), so gap ≤ 0 is only reachable when the neuron is stable — in
    # which case Tech 2 has already removed the binary and n2_relax_decision's
    # stable-both short-circuit skips it. No actual triangle relaxation is
    # emitted at τ = 0, so no coupling constraint is needed.
    if adv_std_n2_relax_threshold > 0.0 && !args["use_perturbed_intervals"]
        if args["allow_relax_without_pi"]
            # Ablation mode: the emitted cuts are exact-point-valid on their own
            # (chord uses the exact encoding's (l, u); SibGate's drift interval is
            # a computed enclosure, not a model constraint), so δ_relaxed ≥ δ_exact
            # still holds — only tightness/speed degrade. See --allow_relax_without_pi.
            println("WARNING (--allow_relax_without_pi): running --adv_std_n2_relax_threshold > 0 " *
                    "WITHOUT --use_perturbed_intervals. Sound (δ_relaxed ≥ δ_exact) but the " *
                    "relaxed copy is only loosely tied to the exact one — expect a looser " *
                    "bound and a slower solve. Perturbed-interval ablation runs only.")
        else
            println("UNSOUND COMBINATION: --adv_std_n2_relax_threshold > 0 (Technique 4, " *
                    "BoundTightPertRelax) requires --use_perturbed_intervals=true for soundness. " *
                    "Without the perturbed-interval coupling constraints the relaxed and exact " *
                    "copies can drift apart and δ_exact is not guaranteed. Exiting without " *
                    "encoding or optimizing. Set --use_perturbed_intervals true, or disable " *
                    "Technique 4 with --adv_std_n2_relax_threshold off (or use τ = 0, which " *
                    "emits no actual relaxations), or pass --allow_relax_without_pi true " *
                    "for a perturbed-interval ablation run.")
            return
        end
    end
    n1_state_dir = args["n1_state_dir"]

    c_tag_list = [args["ctag"]]
    activate_vaghgar_deps = args["activate_vaghgar_deps"]
    global use_relaxations = args["use_relaxations"]
    global relaxation_threshold = args["relaxation_threshold"]
    global optimizing_intervals = args["optimizing_intervals"]
    global relaxation_gap_area = args["relaxation_gap_area"]
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
    println("  Technique 4+ (N1 Probe):           $(args["adv_std_n1_probe"])")
    println("  Technique 5 (Variable Hints):      $(var_hint_mode_label(var_hint_mode))")
    println("  Technique 6 (N2 Relax Threshold):  $(adv_std_n2_relax_threshold)")

    # ── Pre-flight: build n2_check suffix for skip detection ──
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
        if args["adv_std_n1_probe"] == "lp"; n2_check = n2_check * "_n1ProbeLP"; end
        # Mode-specific varHint filename tag. Keep the legacy "_varHintFixed"
        # token for VH_PREV (so historical result files keep comparing cleanly);
        # VH_DIRECT emits "_varHintDirect"; VH_DIRECT_PGD emits "_varHintDirectPGD";
        # VH_PREV_PGD emits "_varHintPrevPGD".
        if var_hint_mode == VH_PREV;       n2_check = n2_check * "_varHintFixed";     end
        if var_hint_mode == VH_DIRECT;     n2_check = n2_check * "_varHintDirect";    end
        if var_hint_mode == VH_DIRECT_PGD; n2_check = n2_check * "_varHintDirectPGD"; end
        if var_hint_mode == VH_PREV_PGD;   n2_check = n2_check * "_varHintPrevPGD";   end
    end
    if use_hyper_attack; n2_check = n2_check * "_HyperAttackHints"; end
    if activate_vaghgar_deps;              n2_check = n2_check * "_VagharDeps_depGuardFix"; end  # must mirror the saved name's tag (see line ~666)
    if args["use_perturbed_intervals"]
        n2_check = n2_check * "_PerturbedIntervals"
    else
        # Perturbed-interval ablation runs (--allow_relax_without_pi): stamp an
        # explicit _noPI so these files never alias the PI runs of the same combo
        # (the sweep's skip-check globs the tag tail with a wildcard).
        n2_check = n2_check * "_noPI"
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

        load_n1_from_disk = n1_state_dir != "" && isdir(n1_state_dir)
        # nn1 is needed for Technique 4 diff-bound computation and for the
        # N1-probe LP (Source C). In the state-dir load path we normally
        # skip loading nn1 to save time, but if the probe is enabled we
        # need nn1's weights to build the joint LP.
        nn1_needed = !load_n1_from_disk || args["adv_std_n1_probe"] == "lp"
        nn1 = nn1_needed ? get_nn(model_path, model_name, w, h, k, c, dataset) : nothing
        nn2 = get_nn(model_path2, model_name, w, h, k, c, dataset)

        # ── Technique 4: precompute sound per-neuron diff bounds ──────────
        clear_n2_abs_bounds()
        if use_bound_tightening
            input_dummy = zeros(Float64, 1, w, h, k)
            p_size = perturbation_size[1]
            # Full 4D (1,w,h,k) per-pixel L∞ box for both single- and multi-channel
            # inputs; the old size[4]>1 branch seeded a malformed (k,1) matrix that
            # collapsed spatial dims and broke conv bound-propagation on multi-
            # channel (e.g. CIFAR-10) nets.
            I_pert_up_init = p_size .* ones(Float64, size(input_dummy))
            I_pert_down_init = -p_size .* ones(Float64, size(input_dummy))

            if load_n1_from_disk
                # Phase-2 loads from disk. diff_bounds.bin is mandatory;
                # n1_preact_bounds.bin is mandatory iff zono (Source B) is active.
                # Both loads crash-on-miss rather than silently degrading.
                load_n1_diff_bounds!(n1_state_dir; require_preact=use_zono_bounds)
            elseif use_zono_bounds
                # No state dir — compute diff bounds on the fly (no Phase 1 to load from).
                global use_zonotope = true
                println("Advanced-standard: computing zonotope diff bounds (Source A) between N1 and N2...")
                compute_diff_bounds_zonotope(nn1, nn2, I_pert_up_init, I_pert_down_init; optimizing_intervals=optimizing_intervals)
                println("  Source A diff bounds computed: $(length(relu_diff_up_bounds)) ReLU layers")
            else
                println("Advanced-standard: computing diff bounds between N1 and N2...")
                compute_diff_and_comp_bounds(nn1, nn2, I_pert_up_init, I_pert_down_init; optimizing_intervals=optimizing_intervals)
                println("  diff bounds computed: $(length(relu_diff_up_bounds)) ReLU layers")
            end

            # Source B: absolute N2 zonotope, intersected with Source A per layer.
            # Needs nn2 (always available here) and Source A's globals
            # (relu_diff_*_bounds, n1_preact_*_bounds). Skips internally with a
            # warning if those globals are empty, e.g. legacy n1_state_dir load.
            if use_zono_bounds
                println("Advanced-standard: computing N1-tightened absolute N2 zonotope (Source B)...")
                compute_n2_bounds_zonotope_with_n1_tighten(nn2, I_pert_up_init, I_pert_down_init;
                                                           use_n1_tighten=zono_use_npre)
            end

            # Source C: joint N1+N2 LP probe via OBBT (--adv_std_n1_probe=lp).
            # Builds a fresh LP-only model with triangle-relaxed ReLUs for
            # both N1 and N2 sharing v_in/v_x0, runs per-neuron min/max on
            # every N2 pre-activation. Output is strictly tighter than
            # Source A/B because the LP polytope captures cross-neuron
            # correlations that scalar/zonotope projections lose.
            clear_n2_probe_bounds()
            if args["adv_std_n1_probe"] == "lp"
                println("Advanced-standard: running N1-probe LP (Source C) to tighten N2 bounds...")
                probe_ran = compute_n2_bounds_n1_probe_lp(
                    nn1, nn2, perturbation, perturbation_size,
                    w, h, k, Gurobi.Optimizer)
                if !probe_ran
                    println("ERROR: --adv_std_n1_probe=lp was requested but Source C could not run " *
                            "(see the compute_n2_bounds_n1_probe_lp message above). Aborting this " *
                            "Phase 2 invocation without running any MIP solves — no result files " *
                            "will be written. Re-run with compatible prerequisites (supported " *
                            "perturbation + populated n1_preact/relu_diff bounds + matching ReLU " *
                            "layer counts) or drop --adv_std_n1_probe.")
                    return
                end
            end
        end

        for c_target in c_targets
            name_to_save = name_to_save_init
            global relaxation_condition_count = 0
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

            # ═══ PASS 1: Get N1 solver state (solve or load from disk) ═══
            n1_var_names, n1_var_values = String[], Float64[]
            n1_layers_info = Dict{Tuple{Int,Int}, Tuple{Float64,Float64,Int}}()

            if load_n1_from_disk
                println("\n══ Advanced-standard: loading N1 state from $n1_state_dir (c_tag=$c_tag, c_target=$c_target) ══")
                n1_var_names, n1_var_values, n1_layers_info =
                    load_n1_state(n1_state_dir, c_tag, c_target)
            else
                println("\n══ Advanced-standard PASS 1: solving N1 (c_tag=$c_tag, c_target=$c_target) ══")
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

                # Extract ALL N1 solver info (regardless of which techniques are enabled,
                # so the state is complete for any N2 job)
                n1_var_names, n1_var_values = extract_all_variable_values(m_n1)
                n1_layers_info = deepcopy(layers_info_dict)
                m_n1 = nothing
            end

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
                adv_std_n1_probe             = args["adv_std_n1_probe"],
                adv_std_n2_relax_threshold   = args["adv_std_n2_relax_threshold"],
                adv_std_var_hint             = var_hint_mode_label(var_hint_mode),   # "off" | "prev" | "direct"
                gurobi_seed                  = args["gurobi_seed"],
            )
            optimizer = Gurobi.Optimizer
            mip_reset()

            # Technique 4: set N1 neuron bounds (consumed by relu() during get_model)
            if use_bound_tightening
                set_n1_neuron_bounds(n1_layers_info)
            end

            # Technique 6: reset the relaxed-binary counters per c_target so
            # the filename reflects the counts for *this specific* MIP build.
            clear_n2_relaxed_counters!()
            # Technique 4 (SibGate) per-tier counters reset per c_target.
            clear_sibgate_tier_counters!()

            # Technique 4 (SibGate): populate the org↔pert pre-activation
            # diff bound through N2 alone (relu_n2pert_*_bounds) and N2's
            # per-copy pre-activation bound (n2_preact_*_bounds). These are
            # the "[l_int, u_int]" and "[l_pre, u_pre]" inputs of the
            # conditional-triangle / pre-act-coupling derivation in
            # advstd_techniques.tex §Tech 4. Reuses the existing routine
            # also used by --no_n1_binaries_and_relaxtions_only_on_n2.
            if adv_std_n2_sibling_gate && adv_std_n2_relax_threshold >= 0.0 && use_bound_tightening
                compute_n2_pert_relaxation_bounds(nn2, I_pert_up_init, I_pert_down_init)
            end

            # Technique 6 (BoundTightPertRelax): precompute the per-copy
            # relaxation decision using N2's own tightened bounds (Sources
            # A/B/C, now populated above). relu() consults the resulting
            # n2_relax_decision dict instead of re-scoring from N1's bounds.
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

            # Technique 4 (SibGate): both org and pert copies are now
            # encoded inside m_n2. Walk n2_relax_decision and emit:
            #   • coupling lines for both-thin neurons
            #   • conditional triangles (gated on the sibling binary)
            #     for one-thin neurons
            # No-op when --adv_std_n2_sibling_gate=false.
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
                if args["adv_std_n1_probe"] == "lp"; n2_name = n2_name * "_n1ProbeLP"; end
                # See n2_check builder above: VH_PREV keeps the legacy tag,
                # VH_DIRECT emits "_varHintDirect", VH_DIRECT_PGD emits "_varHintDirectPGD",
                # VH_PREV_PGD emits "_varHintPrevPGD".
                if var_hint_mode == VH_PREV;       n2_name = n2_name * "_varHintFixed";     end
                if var_hint_mode == VH_DIRECT;     n2_name = n2_name * "_varHintDirect";    end
                if var_hint_mode == VH_DIRECT_PGD; n2_name = n2_name * "_varHintDirectPGD"; end
                if var_hint_mode == VH_PREV_PGD;   n2_name = n2_name * "_varHintPrevPGD";   end
            end
            # Append the probe's binary-elimination counts (org + pert split)
            # as the LAST suffix so pre-flight substring matching (which uses
            # n2_check without these counts) still finds already-completed files.
            if args["adv_std_n1_probe"] == "lp"
                n2_name = n2_name *
                          "_elimOrg" * string(n_probe_eliminated_binaries_org) *
                          "_elimPert" * string(n_probe_eliminated_binaries_pert)
            end
            # Technique 4 (SibGate): per-tier neuron counts so the filename
            # encodes how the relaxation actually played out:
            #   _both<N>     : neurons with BOTH binaries dropped (Tier 1).
            #   _orgDrop<N>  : neurons with only org binary dropped (Tier 2,
            #                  pert binary survives → conditional triangle on org
            #                  gated on a^pert).
            #   _pertDrop<N> : neurons with only pert binary dropped (Tier 2,
            #                  org binary survives → conditional triangle on pert
            #                  gated on a^org).
            # Counters set by compute_n2_relax_decision!; pre-flight skip uses
            # n2_check (without these counts), so historical files still match.
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
                # _depGuardFix marks the fixed dep encoder; must follow
                # _VagharDeps so paper-tables keeps this binary-dropping run.
                n2_name = n2_name * "_VagharDeps_depGuardFix"
                perturbation_dependencies(m_n2, nn2, perturbation, perturbation_size, w, h, k;
                                          perturbation_var=d_n2[:Perturbation])
            end
            if args["use_perturbed_intervals"]
                n2_name = n2_name * "_PerturbedIntervals"
                if geometric_intervals; n2_name = n2_name * "_geomInt"; end
                perturbed_interval_constraints(m_n2, nn2, "org", "perturbation")
            else
                # Mirror the n2_check pre-flight tag: mark PI-ablation results
                # so they stay distinguishable from same-combo PI runs.
                n2_name = n2_name * "_noPI"
            end
            mip_set_delta_property(m_n2, perturbation, d_n2)
            set_optimizer(m_n2, optimizer)
            mip_set_attr(m_n2, perturbation, d_n2, timout)
            MOI.set(m_n2, Gurobi.CallbackFunction(), my_callback)
            # Technique 5: Variable hints.
            # Mode ∈ {off, prev, direct, direct_pgd, prev_pgd}; apply_n1_var_hints!
            # early-returns on VH_OFF. The *_PGD modes additionally consume PGD's
            # Start values (set by hyper_attack_hints above), so they must run
            # after the hyper_attack_hints call at line ~999.
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

# ═══════════════════════════════════════════════════════════════════════════
# Advanced Standard N1: solve N1 only and save solver state to disk.
# Run once; then run advanced_standard_n2 (or advanced_standard --n1_state_dir)
# multiple times in parallel with different technique flags.
# ═══════════════════════════════════════════════════════════════════════════
function main_advanced_standard_n1(args, dataset, model_name, model_path, perturbation, perturbation_size, name_to_save, use_hyper_attack)
    model_path2 = args["model_path2"]
    n1_state_dir = args["n1_state_dir"]
    if n1_state_dir == ""
        error("advanced_standard_n1 requires --n1_state_dir (where to save N1 state)")
    end

    c_tag_list = [args["ctag"]]
    activate_vaghgar_deps = args["activate_vaghgar_deps"]
    global use_relaxations = args["use_relaxations"]
    global relaxation_threshold = args["relaxation_threshold"]
    global optimizing_intervals = args["optimizing_intervals"]
    global relaxation_gap_area = args["relaxation_gap_area"]
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
            # Full 4D (1,w,h,k) per-pixel L∞ box for both single- and multi-channel
            # inputs; the old size[4]>1 branch seeded a malformed (k,1) matrix that
            # collapsed spatial dims and broke conv bound-propagation on multi-
            # channel (e.g. CIFAR-10) nets.
            I_pert_up_init = p_size .* ones(Float64, size(input_dummy))
            I_pert_down_init = -p_size .* ones(Float64, size(input_dummy))
            if args["adv_std_zono_bounds"]
                global use_zonotope = true
                println("Advanced-standard-N1: computing zonotope diff bounds between N1 and N2...")
                compute_diff_bounds_zonotope(nn1, nn2, I_pert_up_init, I_pert_down_init; optimizing_intervals=optimizing_intervals)
            else
                println("Advanced-standard-N1: computing diff bounds between N1 and N2...")
                compute_diff_and_comp_bounds(nn1, nn2, I_pert_up_init, I_pert_down_init; optimizing_intervals=optimizing_intervals)
            end
            # Preserve existing on-disk diff_bounds when completing a partial
            # state dir: per-pair n1_vars/n1_layers/... files already on disk
            # were produced against those saved bounds, and overwriting with
            # freshly-recomputed bounds (which can differ in the last ULPs due
            # to Gurobi LP threading) would desync old vs new pair files.
            # In-memory globals are still populated above so this c_tag's
            # per-c_target N1 MIPs get the comp-bound tightening.
            diff_bounds_path = joinpath(n1_state_dir, "diff_bounds.bin")
            preact_path = joinpath(n1_state_dir, "n1_preact_bounds.bin")
            if !isfile(diff_bounds_path)
                save_n1_diff_bounds(n1_state_dir)
            elseif !isfile(preact_path) && !isempty(n1_preact_up_bounds)
                # Diff bounds on disk, but n1_preact upgrade available.
                # Write only the new preact file; leave diff_bounds.bin alone.
                serialize(preact_path, (n1_preact_up_bounds, n1_preact_down_bounds))
                println("Advanced-standard-N1: preserved existing diff_bounds.bin; wrote new n1_preact_bounds.bin to $n1_state_dir")
            else
                println("Advanced-standard-N1: preserved existing diff_bounds.bin + n1_preact_bounds.bin at $n1_state_dir (partial-completion mode)")
            end
        end

        for c_target in c_targets
            global relaxation_condition_count = 0
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