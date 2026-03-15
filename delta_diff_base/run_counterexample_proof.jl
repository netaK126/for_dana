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
            println("Warning: no delta_1 found for c_target=$c_target_index")
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
include("utils/models.jl")
include("utils/mip.jl")

# ============================================================
#  COUNTEREXAMPLE NETWORKS  (2-pixel input, 2-class output)
#
#  N1: logits = [x1,  x2]       conf_N1(x) = x1 - x2
#  N2: logits = [2x1, 5x2]      conf_N2(x) = 2x1 - 5x2
#
#  Perturbation: linf, p = 0.1,  c_tag=1, c_target=2
#
#  Analytic results:
#    delta_1(N1) = 0.199
#    delta_1(N2) = 0.699
#    delta_diff* = 0.3495
#    delta_1(N1) + delta_diff* = 0.5485  <  0.699 = delta_1(N2)
# ============================================================

println("\n" * "="^60)
println("COUNTEREXAMPLE: delta_1(N1)+delta_diff < delta_1(N2)")
println("="^60)
println("""
Network definitions  (x = (x1,x2) ∈ [0,1]², p = 0.1, η = 0.001):
  N1: logits = [x1, x2]           → conf_N1(x) = x1 - x2
  N2: logits = [2·x1, 5·x2]       → conf_N2(x) = 2x1 - 5x2

Foolability (analytic):
  N1-foolable iff ε2-ε1 ≥ (x1-x2)+η  and  max(ε2-ε1)=0.2
                 → x1-x2 ≤ 0.199
  N2-foolable iff 5ε2-2ε1 ≥ (2x1-5x2)+η  and  max(5ε2-2ε1)=0.7
                 → 2x1-5x2 ≤ 0.699

Expected MIP results:
  delta_1(N1) = 0.199   (standard mode on N1)
  delta_1(N2) = 0.699   (standard mode on N2)
  delta_diff* = 0.3495  (transfer MIP)
  N1+diff     = 0.5485  <  0.699  →  inequality FAILS
""")

# ---- Build networks ----
fc1_N1 = Linear(Float64[1.0 0.0; 0.0 1.0], Float64[0.0, 0.0])
fc2_N1 = Linear(Float64[1.0 0.0; 0.0 1.0], Float64[0.0, 0.0])   # identity → logits=[x1,x2]
nn1 = Sequential(Layer[Flatten([1,3,2,4]), fc1_N1, ReLU(), fc2_N1], "N1")

fc1_N2 = Linear(Float64[1.0 0.0; 0.0 1.0], Float64[0.0, 0.0])
fc2_N2 = Linear(Float64[2.0 0.0; 0.0 5.0], Float64[0.0, 0.0])   # diag(2,5) → logits=[2x1,5x2]
nn2 = Sequential(Layer[Flatten([1,3,2,4]), fc1_N2, ReLU(), fc2_N2], "N2")

w, h, k = 2, 1, 1
c_tag    = 1
c_target = 2
p_size   = 0.1
margin   = 1e-3
timeout  = 120
optimizer = Gurobi.Optimizer
tight_opts = get_default_tightening_options(optimizer)
input_dummy = zeros(Float64, 1, w, h, k)

# ============================================================
# STANDARD-MODE MIP (single network)
#   maximize  conf_N(x) = logit[c_tag] - logit[c_target]
#   subject to:
#     x ∈ [0,1]²,  ε ∈ [-p,p]²,  xp = x+ε ∈ [0,1]²
#     N(xp) predicts c_target:  logit_N(xp)[c_target] ≥ logit_N(xp)[c_tag] + η
# ============================================================
function run_standard_mode(nn, c_tag_, c_target_, dummy, p_, margin_, optimizer_, tight_opts_)
    mip_reset()
    m = Model(optimizer_with_attributes(optimizer_, tight_opts_...))
    # Use interval arithmetic for bound tightening (fast; no binary vars for linear nets)
    m.ext[:MIPVerify] = MIPVerifyExt(interval_arithmetic)

    # Decision variables: clean input x, perturbation ε, perturbed input xp
    v_x  = map(_ -> @variable(m, lower_bound=0.0, upper_bound=1.0), CartesianIndices(size(dummy)))
    v_e  = map(_ -> @variable(m, lower_bound=-p_, upper_bound=p_),   CartesianIndices(size(dummy)))
    v_xp = map(_ -> @variable(m, lower_bound=0.0, upper_bound=1.0), CartesianIndices(size(dummy)))
    @constraint(m, v_xp .== v_x + v_e)

    # Encode both network passes through MIPVerify
    v_out_clean = v_x  |> nn     # N(x)
    v_out_pert  = v_xp |> nn     # N(x+ε)

    # Foolability constraint: perturbed input is misclassified as c_target
    @constraint(m, v_out_pert[c_target_] - v_out_pert[c_tag_] >= margin_)

    # Objective: maximize confidence margin on the clean input
    @objective(m, Max, v_out_clean[c_tag_] - v_out_clean[c_target_])

    set_optimizer(m, optimizer_)
    set_optimizer_attribute(m, "OutputFlag", 0)
    set_optimizer_attribute(m, "TimeLimit", timeout)
    set_optimizer_attribute(m, "MIPGap", 1e-6)
    optimize!(m)

    status = JuMP.termination_status(m)
    delta1 = JuMP.objective_bound(m)    # best_bound  (= true optimum for LP)
    incumb = try JuMP.objective_value(m) catch; NaN end
    x_sol  = JuMP.value.(v_x)
    xp_sol = JuMP.value.(v_xp)
    e_sol  = JuMP.value.(v_e)
    return delta1, incumb, status, x_sol, xp_sol, e_sol, v_out_clean, v_out_pert, m
end

# ============================================================
# PART A — Standard Mode on N1  →  delta_1(N1)
# ============================================================
println("="^60)
println("PART A: Standard Mode on N1")
println("="^60)

d1_N1, inc_N1, st_N1, x_N1, xp_N1, e_N1, vout_N1, voutp_N1, m_N1 =
    run_standard_mode(nn1, c_tag, c_target, input_dummy, p_size, margin, optimizer, tight_opts)

x1_N1  = x_N1[1,1,1,1];  x2_N1  = x_N1[1,2,1,1]
xp1_N1 = xp_N1[1,1,1,1]; xp2_N1 = xp_N1[1,2,1,1]
e1_N1  = e_N1[1,1,1,1];  e2_N1  = e_N1[1,2,1,1]

conf_N1_clean = JuMP.value(vout_N1[c_tag]  - vout_N1[c_target])
conf_N1_pert  = JuMP.value(voutp_N1[c_tag] - voutp_N1[c_target])

@printf "  Status:             %s\n" st_N1
@printf "  Optimal x*       = [%.6f, %.6f]\n" x1_N1 x2_N1
@printf "  Optimal ε*       = [%.6f, %.6f]\n" e1_N1 e2_N1
@printf "  Perturbed x_p*   = [%.6f, %.6f]\n" xp1_N1 xp2_N1
@printf "  conf_N1(x*)      = %.6f  (analytic: x1-x2 = %.6f)\n" conf_N1_clean (x1_N1-x2_N1)
@printf "  conf_N1(xp*) [perturbed] = %.6f  (should be ≤ -η = -0.001)\n" conf_N1_pert
@printf "  N1-foolable at x*? %s\n" (conf_N1_pert <= -margin + 1e-6 ? "YES ✓" : "NO ✗")
@printf "\n  delta_1(N1) [best_bound] = %.6f  (analytic: 0.199)\n" d1_N1

# Write to vaghar-format file
vaghar_path = "/tmp/counterexample_vaghar_N1.txt"
open(vaghar_path, "w") do f
    write(f, "$(c_tag-1),$(c_target-1),$(inc_N1),$(d1_N1),1.0\n")
end
println("  Vaghar file written: source=0, target=1, incumbent=$(round(inc_N1,digits=6)), best_bound=$(round(d1_N1,digits=6))")

# ============================================================
# PART B — Standard Mode on N2  →  delta_1(N2)
# ============================================================
println("\n" * "="^60)
println("PART B: Standard Mode on N2")
println("="^60)

d1_N2, inc_N2, st_N2, x_N2, xp_N2, e_N2, vout_N2, voutp_N2, m_N2 =
    run_standard_mode(nn2, c_tag, c_target, input_dummy, p_size, margin, optimizer, tight_opts)

x1_N2  = x_N2[1,1,1,1];  x2_N2  = x_N2[1,2,1,1]
xp1_N2 = xp_N2[1,1,1,1]; xp2_N2 = xp_N2[1,2,1,1]
e1_N2  = e_N2[1,1,1,1];  e2_N2  = e_N2[1,2,1,1]

conf_N2_clean = JuMP.value(vout_N2[c_tag]  - vout_N2[c_target])
conf_N2_pert  = JuMP.value(voutp_N2[c_tag] - voutp_N2[c_target])

@printf "  Status:             %s\n" st_N2
@printf "  Optimal x*       = [%.6f, %.6f]\n" x1_N2 x2_N2
@printf "  Optimal ε*       = [%.6f, %.6f]\n" e1_N2 e2_N2
@printf "  Perturbed x_p*   = [%.6f, %.6f]\n" xp1_N2 xp2_N2
@printf "  conf_N2(x*)      = %.6f  (analytic: 2x1-5x2 = %.6f)\n" conf_N2_clean (2*x1_N2-5*x2_N2)
@printf "  conf_N2(xp*) [perturbed] = %.6f  (should be ≤ -η = -0.001)\n" conf_N2_pert
@printf "  N2-foolable at x*? %s\n" (conf_N2_pert <= -margin + 1e-6 ? "YES ✓" : "NO ✗")
@printf "\n  delta_1(N2) [best_bound] = %.6f  (analytic: 0.699)\n" d1_N2

# ============================================================
# PART C — Transfer MIP  →  delta_diff*
# ============================================================
println("\n" * "="^60)
println("PART C: Transfer MIP  (delta_diff_base)")
println("="^60)

delta_1_from_file = get_delta1_vaghar(vaghar_path, c_target)
println("  delta_1(N1) read from vaghar file: $delta_1_from_file")

d_transfer = Dict{Symbol,Any}()
d_transfer[:suboptimal_solution] = 0.0
d_transfer[:suboptimal_time]     = 0.0

mip_reset()
bounds_time = @elapsed begin
    merge!(d_transfer, get_model(w, h, k, "linf", [p_size], nn1, nn2,
        input_dummy, optimizer, tight_opts, DEFAULT_TIGHTENING_ALGORITHM))
end
d_transfer[:bounds_time] = bounds_time

m_transfer = d_transfer[:Model]
mip_set_transfer_property(m_transfer, d_transfer, delta_1_from_file,
                          c_tag, c_target, false, false, true)
set_optimizer(m_transfer, optimizer)
mip_set_attr(m_transfer, "linf", d_transfer, timeout)
MOI.set(m_transfer, Gurobi.CallbackFunction(), my_callback)
optimize!(m_transfer)
mip_log(m_transfer, d_transfer)

ddiff_star  = d_transfer[:incumbent_obj]
x_tr  = d_transfer[:v_in]
xp_tr = d_transfer[:v_in_p]
e_tr  = d_transfer[:Perturbation]
x1_tr = x_tr[1,1,1,1]; x2_tr = x_tr[1,2,1,1]
xp1_tr= xp_tr[1,1,1,1];xp2_tr= xp_tr[1,2,1,1]
e1_tr = e_tr[1,1,1,1]; e2_tr = e_tr[1,2,1,1]

conf_N1_tr = x1_tr - x2_tr          # = conf_N1(x*) for our linear N1
conf_N2_tr = 2*x1_tr - 5*x2_tr      # = conf_N2(x*) for our linear N2
diff_check = conf_N2_tr - conf_N1_tr

# Perturbed N2 output
conf_N2_pert_tr = 2*xp1_tr - 5*xp2_tr

@printf "\n  Optimal x*        = [%.6f, %.6f]\n" x1_tr x2_tr
@printf "  Optimal ε*        = [%.6f, %.6f]\n" e1_tr e2_tr
@printf "  Perturbed x_p*    = [%.6f, %.6f]\n" xp1_tr xp2_tr
@printf "  conf_N1(x*)       = %.6f  (analytic: x1-x2  = %.6f)\n" conf_N1_tr (x1_tr-x2_tr)
@printf "  conf_N2(x*)       = %.6f  (analytic: 2x1-5x2 = %.6f)\n" conf_N2_tr (2*x1_tr-5*x2_tr)
@printf "  delta_diff at x*  = conf_N2-conf_N1 = %.6f\n" diff_check

@printf "\n  C1: conf_N1(x*)=%.4f >= delta_1(N1)+1e-3=%.4f?  %s\n" conf_N1_tr (delta_1_from_file+1e-3) (conf_N1_tr >= delta_1_from_file+1e-3-1e-6 ? "✓" : "✗")
@printf "  C2: delta_diff=%.4f >= 0?  %s\n" diff_check (diff_check >= -1e-6 ? "✓" : "✗")
@printf "  C3: conf_N2(xp*)=%.6f <= -η?  %s\n" conf_N2_pert_tr (conf_N2_pert_tr <= -margin+1e-6 ? "✓" : "✗")
@printf "\n  delta_diff* [MIP] = %.6f  (analytic: 0.3495)\n" ddiff_star

# ============================================================
# FINAL COMPARISON
# ============================================================
println("\n" * "="^60)
println("FINAL COMPARISON")
println("="^60)

lhs = d1_N1 + ddiff_star    # claimed upper bound
rhs = d1_N2                 # true delta_1(N2)
gap = rhs - lhs

@printf "\n  delta_1(N1)              = %.6f  (analytic: 0.199)\n" d1_N1
@printf "  delta_diff*              = %.6f  (analytic: 0.3495)\n" ddiff_star
@printf "  delta_1(N1)+delta_diff*  = %.6f  (analytic: 0.5485)\n" lhs
@printf "  delta_1(N2)              = %.6f  (analytic: 0.699)\n" rhs
@printf "\n  Gap = delta_1(N2) - (delta_1(N1)+delta_diff*) = %.6f\n" gap

if gap > 1e-6
    println("\n  CONCLUSION: delta_1(N1)+delta_diff* < delta_1(N2)")
    println("  The claimed inequality  delta_1(N1)+delta_diff* >= delta_1(N2)  FAILS. ✗")
    println("\n  The CORRECT (provable) inequality is the reverse:")
    println("  delta_1(N2) >= delta_1(N1)+delta_diff*  ✓")
    @printf "  Proof witness: x* = [%.4f, %.4f]\n" x1_tr x2_tr
    @printf "  conf_N2(x*) = %.4f = conf_N1(x*) + delta_diff* = %.4f + %.4f\n" conf_N2_tr conf_N1_tr ddiff_star
    @printf "  Since conf_N1(x*) = %.4f > %.4f = delta_1(N1),\n" conf_N1_tr d1_N1
    @printf "  we get delta_1(N2) >= conf_N2(x*) = %.4f > %.4f = delta_1(N1)+delta_diff*\n" conf_N2_tr lhs
else
    println("\n  Inequality HOLDS (unexpected for this example).")
end
