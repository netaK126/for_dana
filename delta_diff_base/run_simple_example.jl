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

# Copied from run.jl (reads delta_1 from a vaghar-format results file)
function get_delta1_vaghar(results_path, c_target_index)
    open(results_path, "r") do io
        requested_line = ""
        while !eof(io)
            line_content = readline(io)
            if isempty(strip(line_content))
                continue
            end
            tokens = Base.split(line_content, ',')
            if length(tokens) >= 4
                target_in_file = parse(Int, tokens[2])
                if target_in_file == c_target_index - 1
                    requested_line = line_content
                end
            end
        end
        if requested_line == ""
            println("Warning: no delta_1 found for c_target=$c_target_index in $results_path")
            return -1.0
        end
        parsed_tokens = Base.split(requested_line, ',')
        return parse(Float64, parsed_tokens[4])
    end
end
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
include("utils/mip.jl")

# ============================================================
#  SIMPLE EXAMPLE: 2-pixel input, 2 hidden neurons, 2 classes
#
#  Architecture:  x (2D) -> fc1 -> ReLU -> fc2 -> logits (2D)
#
#  N1 (reference):
#    fc1 = Identity,  fc2 = [[2,-1],[-1,2]]
#    out_N1(x) = [2x1-x2 ,  -x1+2x2]
#    conf_N1(x) = 3(x1-x2)
#
#  N2 (target):
#    fc1 = Identity,  fc2 = [[12,0],[0,25]]
#    out_N2(x) = [12x1,  25x2]
#    conf_N2(x) = 12x1 - 25x2
#
#  Perturbation: linf, p = 0.1
#  c_tag = 1 (class 1 is "correct"), c_target = 2 (adversarial target)
# ============================================================

# ---- Build N1 ----
fc1_N1 = Linear(Float64[1.0 0.0; 0.0 1.0], Float64[0.0, 0.0])   # identity
fc2_N1 = Linear(Float64[2.0 -1.0; -1.0 2.0], Float64[0.0, 0.0])
nn1 = Sequential(Layer[Flatten([1,3,2,4]), fc1_N1, ReLU(), fc2_N1], "N1")

# ---- Build N2 ----
fc1_N2 = Linear(Float64[1.0 0.0; 0.0 1.0], Float64[0.0, 0.0])   # identity
fc2_N2 = Linear(Float64[12.0 0.0; 0.0 25.0], Float64[0.0, 0.0])
nn2 = Sequential(Layer[Flatten([1,3,2,4]), fc1_N2, ReLU(), fc2_N2], "N2")

# ============================================================
# STEP 1 — Standard mode on N1 (what vaghar_org/run.jl does)
#
# Formulation (for all x ∈ [0,1]², ε ∈ [-0.1,0.1]²):
#
#   maximize   conf_N1(x)  =  3(x1 - x2)
#   subject to:
#     x_p = x + ε
#     N1(x_p) predicts class 2:
#       out_N1_p[1] >= out_N1_p[0] + margin
#       => -(x1+ε1)+2(x2+ε2) >= 2(x1+ε1)-(x2+ε2)
#       =>  ε2 - ε1  >  x1 - x2
#
# Analytic solution:
#   N1 is foolable iff x1-x2 < max(ε2-ε1) = 0.1-(-0.1) = 0.2
#   => delta_1 = sup { 3(x1-x2) : x1-x2 < 0.2 } = 3 * 0.2 = 0.6
#
# We write this value to a synthetic vaghar_results file.
# ============================================================

println("\n" * "="^60)
println("STEP 1 — Standard Mode on N1 (computes delta_1)")
println("="^60)
println("""
N1 weights:
  fc1 = Identity(2x2),  bias = [0, 0]
  fc2 = [[2, -1], [-1, 2]],  bias = [0, 0]

For any x ∈ [0,1]²  (ReLU is inactive since pixels ≥ 0):
  h1(x)      = ReLU(x)  = x
  out_N1(x)  = [[2,-1],[-1,2]] × x
             = [2x1 - x2,  -x1 + 2x2]
  conf_N1(x) = out[0] - out[1]
             = (2x1-x2) - (-x1+2x2)
             = 3(x1 - x2)

Standard-mode MIP:
  maximize  3(x1-x2)
  subject to  x ∈ [0,1]²,  ε ∈ [-0.1, 0.1]²
              N1(x+ε) predicts class 2:
                -(x1+ε1)+2(x2+ε2) >= 2(x1+ε1)-(x2+ε2)
                => ε2 - ε1  >  x1 - x2

Analytic answer:
  N1 is foolable iff  x1-x2 < 0.1-(-0.1) = 0.2
  => delta_1 = 3 × 0.2 = 0.6
""")

# Write synthetic vaghar file (0-indexed source=0, target=1)
vaghar_path = "/tmp/simple_vaghar_results.txt"
open(vaghar_path, "w") do f
    write(f, "0,1,0.594,0.600,1.0\n")   # incumbent≈0.594, best_bound=0.600 = delta_1
end
println("Synthetic vaghar file written to: $vaghar_path")
println("  Format: source,target,incumbent,best_bound,solve_time")
println("  delta_1 = best_bound = 0.600  (4th column)\n")

# ============================================================
# STEP 2 — Transfer Mode MIP  (delta_diff_base/run.jl)
#
# Formulation (for all x ∈ [0,1]², ε ∈ [-0.1,0.1]²):
#
#   maximize   delta_diff(x) = conf_N2(x) - conf_N1(x)
#                            = (12x1-25x2) - 3(x1-x2)
#                            = 9x1 - 22x2
#   subject to:
#     [C1] conf_N1(x) >= delta_1 + 1e-3 = 0.601
#          => 3(x1-x2) >= 0.601  => x1-x2 >= 0.2
#     [C2] delta_diff >= 0
#          => 9x1-22x2 >= 0
#     [C3] out_N2_p[1] >= out_N2_p[0] + 1e-3
#          => 25(x2+ε2) >= 12(x1+ε1) + 1e-3
#
# With optimal ε = [-0.1, +0.1] (pushes x_p toward class-2 region):
#   [C3] => 25x2+2.5 >= 12x1-1.2  =>  12x1-25x2 <= 3.7
#
# LP relaxation (treating ε as chosen optimally):
#   maximize  9x1-22x2
#   s.t.  x1-x2 >= 0.2,  12x1-25x2 <= 3.7,  x ∈ [0,1]²
#
# Analytic LP optimum: objective decreases in x2 (coefficient -22),
#   set x2=0 => maximize 9x1 s.t. x1>=0.2, 12x1<=3.7 => x1=3.7/12≈0.308
#   => delta_diff* = 9 * 3.7/12 = 33.3/12 ≈ 2.775
#   at x* ≈ [0.308, 0]
# ============================================================

println("="^60)
println("STEP 2 — Transfer Mode MIP (delta_diff_base)")
println("="^60)
println("""
N2 weights:
  fc1 = Identity(2x2),  bias = [0, 0]
  fc2 = [[12, 0], [0, 25]],  bias = [0, 0]

For any x ∈ [0,1]²:
  out_N2(x)  = [12x1,  25x2]
  conf_N2(x) = 12x1 - 25x2

Transfer MIP:
  maximize   delta_diff = conf_N2(x) - conf_N1(x)
                        = (12x1-25x2) - 3(x1-x2)  =  9x1 - 22x2

  subject to:
    [C1] 3(x1-x2) >= 0.601          [N1 certified on x]
    [C2] 9x1-22x2 >= 0              [N2 at least as confident]
    [C3] 25(x2+ε2) >= 12(x1+ε1)+1e-3  [N2 fooled on x+ε]
         (with optimal ε = [-0.1, +0.1])
         => 12x1 - 25x2 <= 3.7     [N2 foolable]

Analytic LP optimum:
  Set x2=0 (minimizes 22x2), maximize 9x1 s.t. 12x1 <= 3.7
  => x* ≈ [3.7/12, 0] ≈ [0.308, 0]
  => delta_diff* ≈ 9 × 0.308 ≈ 2.775
""")

# ============================================================
# MANUAL VERIFICATION at the feasible point x=[0.9, 0.3], ε=[-0.1,+0.1]
# ============================================================

println("="^60)
println("Manual verification at x=[0.9, 0.3],  ε=[-0.1, +0.1]")
println("="^60)

x1_m, x2_m = 0.9, 0.3
ε1_m, ε2_m = -0.1, 0.1
xp1_m, xp2_m = x1_m + ε1_m, x2_m + ε2_m

# N1 forward
h1_m  = [x1_m, x2_m]                  # ReLU(I×x) = x since x ≥ 0
out_N1_m = [2*h1_m[1] - h1_m[2], -h1_m[1] + 2*h1_m[2]]
conf_N1_m = out_N1_m[1] - out_N1_m[2]

# N2 forward (clean)
out_N2_m = [12*x1_m, 25*x2_m]
conf_N2_m = out_N2_m[1] - out_N2_m[2]

# N2 forward (perturbed)
out_N2p_m = [12*xp1_m, 25*xp2_m]

delta_diff_m = conf_N2_m - conf_N1_m

println("  h1 = ReLU(x) = [$x1_m, $x2_m]")
println("  out_N1 = [2×$x1_m − $x2_m,  −$x1_m + 2×$x2_m] = $(out_N1_m)")
@printf "  conf_N1 = %.4f − (%.4f) = %.4f\n" out_N1_m[1] out_N1_m[2] conf_N1_m
println("  C1 check: conf_N1=$conf_N1_m >= 0.601?  $(conf_N1_m >= 0.601 ? "✓ YES" : "✗ NO")\n")

println("  out_N2 = [12×$x1_m, 25×$x2_m] = $(out_N2_m)")
@printf "  conf_N2 = %.4f − %.4f = %.4f\n" out_N2_m[1] out_N2_m[2] conf_N2_m
@printf "  delta_diff = conf_N2 − conf_N1 = %.4f − %.4f = %.4f\n" conf_N2_m conf_N1_m delta_diff_m
println("  C2 check: delta_diff=$delta_diff_m >= 0?  $(delta_diff_m >= 0 ? "✓ YES" : "✗ NO")\n")

println("  x_p = x + ε = [$xp1_m, $xp2_m]")
println("  out_N2_p = [12×$xp1_m, 25×$xp2_m] = $(out_N2p_m)")
margin_check = out_N2p_m[2] - out_N2p_m[1]
println("  N2_p[class2] − N2_p[class1] = $(out_N2p_m[2]) − $(out_N2p_m[1]) = $margin_check")
println("  C3 check: N2 predicts class 2?  $(margin_check >= 1e-3 ? "✓ YES (margin=$(margin_check))" : "✗ NO")\n")

println("  >> Feasible point: delta_diff = $delta_diff_m  (MIP will find a better x)\n")

# ============================================================
# RUN THE ACTUAL MIP
# ============================================================

println("="^60)
println("Running Transfer Mode MIP...")
println("="^60)

w, h, k, c = 2, 1, 1, 2   # 2 pixels, 1 height, 1 channel, 2 classes
perturbation = "linf"
perturbation_size = [0.1]
c_tag = 1        # source class (1-indexed)
c_target = 2     # attack target class (1-indexed)
timout = 300     # 5 minute timeout (tiny problem solves in seconds)

optimizer = Gurobi.Optimizer
d = Dict{Symbol,Any}()
d[:suboptimal_solution] = 0.0
d[:suboptimal_time]     = 0.0

mip_reset()

bounds_time = @elapsed begin
    merge!(d, get_model(w, h, k, perturbation, perturbation_size, nn1, nn2,
        zeros(Float64, 1, w, h, k), optimizer,
        get_default_tightening_options(optimizer),
        DEFAULT_TIGHTENING_ALGORITHM))
end
d[:bounds_time] = bounds_time

m = d[:Model]

delta_1 = get_delta1_vaghar(vaghar_path, c_target)
println("\ndelta_1 read from vaghar file: $delta_1")

mip_set_transfer_property(m, d, delta_1, c_tag, c_target, false, false, true)
set_optimizer(m, optimizer)
mip_set_attr(m, perturbation, d, timout)
MOI.set(m, Gurobi.CallbackFunction(), my_callback)
optimize!(m)
mip_log(m, d)

# ============================================================
# PRINT AND VERIFY MIP SOLUTION
# ============================================================

println("\n" * "="^60)
println("MIP Solution")
println("="^60)

if d[:SolveStatus] == MOI.OPTIMAL || d[:SolveStatus] == MOI.LOCALLY_SOLVED ||
   d[:incumbent_obj] > 0

    delta_diff_opt = d[:incumbent_obj]
    x_opt  = d[:v_in]     # shape (1,2,1,1)
    xp_opt = d[:v_in_p]   # shape (1,2,1,1)
    e_opt  = d[:Perturbation]  # shape (1,2,1,1)

    x1_opt  = x_opt[1,1,1,1];  x2_opt  = x_opt[1,2,1,1]
    xp1_opt = xp_opt[1,1,1,1]; xp2_opt = xp_opt[1,2,1,1]
    e1_opt  = e_opt[1,1,1,1];  e2_opt  = e_opt[1,2,1,1]

    @printf "  Optimal x*      = [%.6f, %.6f]\n" x1_opt x2_opt
    @printf "  Optimal ε*      = [%.6f, %.6f]\n" e1_opt e2_opt
    @printf "  Perturbed x_p*  = [%.6f, %.6f]\n" xp1_opt xp2_opt
    @printf "  delta_diff*     = %.6f\n\n" delta_diff_opt

    # Verify all constraints at the MIP solution
    println("  Verifying MIP solution:")

    # C1: conf_N1(x*) >= delta_1 + 1e-3
    out_N1_opt = [2*x1_opt - x2_opt, -x1_opt + 2*x2_opt]
    conf_N1_opt = out_N1_opt[1] - out_N1_opt[2]   # = 3(x1-x2)
    @printf "    out_N1(x*) = [%.4f, %.4f]\n" out_N1_opt[1] out_N1_opt[2]
    @printf "    conf_N1(x*) = %.4f  [expected: 3×(%.4f-%.4f) = %.4f]\n" conf_N1_opt x1_opt x2_opt 3*(x1_opt-x2_opt)
    @printf "    C1: conf_N1=%.4f >= delta_1+1e-3=%.4f?  %s\n" conf_N1_opt (delta_1+1e-3) (conf_N1_opt >= delta_1+1e-3 ? "✓" : "✗")

    # C2: delta_diff >= 0
    out_N2_opt = [12*x1_opt, 25*x2_opt]
    conf_N2_opt = out_N2_opt[1] - out_N2_opt[2]
    @printf "    out_N2(x*) = [%.4f, %.4f]\n" out_N2_opt[1] out_N2_opt[2]
    @printf "    conf_N2(x*) = %.4f\n" conf_N2_opt
    delta_diff_check = conf_N2_opt - conf_N1_opt
    @printf "    C2: delta_diff=%.4f >= 0?  %s\n" delta_diff_check (delta_diff_check >= 0 ? "✓" : "✗")

    # C3: N2(x_p*) predicts class 2
    out_N2p_opt = [12*xp1_opt, 25*xp2_opt]
    margin_opt = out_N2p_opt[2] - out_N2p_opt[1]
    @printf "    out_N2(x_p*) = [%.4f, %.4f]\n" out_N2p_opt[1] out_N2p_opt[2]
    @printf "    N2_p[class2]-N2_p[class1] = %.6f\n" margin_opt
    # Note: at the MIP optimum the margin is exactly 1e-3 (constraint is tight);
    # allow 1e-6 tolerance for floating-point representation of the solver output.
    @printf "    C3: N2 predicts class 2?  %s  (margin≈1e-3: constraint tight at optimum)\n" (margin_opt >= 1e-3 - 1e-6 ? "✓" : "✗")

    println()
    @printf "  Analytic expected: x*≈[0.308, 0.0], delta_diff*≈2.775\n"
    @printf "  MIP reported:      x*=[%.3f, %.3f], delta_diff*=%.4f\n" x1_opt x2_opt delta_diff_opt

else
    println("  No feasible solution found (status: $(d[:SolveStatus]))")
    println("  This means NO x exists where N2 is more confident than N1 on the clean input")
    println("  yet still fooled by the perturbation — robustness is provably transferred.")
end

println("\n" * "="^60)
println("Summary")
println("="^60)
println("""
  delta_1 (from standard mode on N1)  =  $(delta_1)
    Meaning: maximum conf_N1(x) over all x that N1 can be fooled at

  delta_diff* (from transfer MIP)     ≈  $(round(d[:incumbent_obj], digits=4))
    Meaning: there exists a clean input x* where
      - N1 is certified robust (conf_N1 ≥ delta_1)
      - N2 is MORE confident than N1 (delta_diff > 0)
      - N2 is STILL FOOLED by a small perturbation ε*
    This is a COUNTEREXAMPLE to perfect robustness transfer from N1 → N2.
""")

# Save results
results.str = update_results_str(results.str, c_tag, c_target, d)
save_results("./results/", "simple_2x2", perturbation, perturbation_size, results.str, d, c_tag-1, c_target-1, w, h, k)
println("Results saved to ./results/simple_2x2_linf_0.1DeltaDiffVerification.txt")
