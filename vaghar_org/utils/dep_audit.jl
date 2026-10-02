# Audit of the perturbation-dependency signs: for each neuron the dependency code would
# probe, decide whether the sign it stamps is actually valid. The pre-fix code read
# objective_value, which bounds the diff only from the side the search reached, so a
# timed-out probe could stamp a sign that does not hold. Here the sign is refuted by a
# witness (an input pair that violates it, re-checked by a forward pass) or proved by
# objective_bound, and nothing is added to the model.

mutable struct DepAuditConf
    enabled::Bool
    io::Union{IOStream,Nothing}
    out_dir::String
    tag::String
    tol::Float64
    budgets::Vector{Float64}   # tried in order until a direction is decided
    threads::Int
    meta::Vector{String}        # dataset, arch, model, perturbation, size
    nn::Any
    v_in::Any
    v_in_p::Any
    n_witness::Int
end

const dep_audit = DepAuditConf(false, nothing, "", "", 1e-4, [10.0, 60.0], 1,
                               String[], nothing, nothing, nothing, 0)

const DEP_AUDIT_HEADER = "dataset,arch,model,perturbation,size,layer,neuron," *
                         "l_org,u_org,l_pert,u_pert,guard_ge,guard_le," *
                         "min_bound,min_val,max_bound,max_val," *
                         "verdict_ge,verdict_le,old_stamp_ge,old_stamp_le," *
                         "old_val_min,old_val_max,fired_ge,fired_le," *
                         "witness_ok,witness_z,witness_zp,probe_sec,witness_file"

"""
    dep_audit_begin!(out_dir, tag, meta; tol, budgets, threads)

Open the per-job CSV. `meta` is the report key: dataset, arch, model, perturbation, size.
"""
function dep_audit_begin!(out_dir::String, tag::String, meta::Vector{String};
                          tol::Float64 = 1e-4, budgets = [10.0, 60.0], threads::Int = 1)
    mkpath(out_dir)
    dep_audit.enabled = true
    dep_audit.out_dir = out_dir
    dep_audit.tag = tag
    dep_audit.tol = tol
    dep_audit.budgets = collect(Float64, budgets)
    dep_audit.threads = threads
    # The size field is comma-separated upstream (e.g. "5,5,5"), which would split the row.
    dep_audit.meta = [replace(s, "," => "-") for s in meta]
    dep_audit.n_witness = 0
    dep_audit.io = open(joinpath(out_dir, tag * ".csv"), "w")
    println(dep_audit.io, DEP_AUDIT_HEADER)
    flush(dep_audit.io)
    return nothing
end

"Hand the audit the network and the two input variable arrays, for witness extraction."
function dep_audit_set_model!(nn, v_in, v_in_p)
    dep_audit.nn = nn
    dep_audit.v_in = v_in
    dep_audit.v_in_p = v_in_p
    return nothing
end

function dep_audit_end!()
    if dep_audit.io !== nothing
        close(dep_audit.io)
        dep_audit.io = nothing
    end
    dep_audit.enabled = false
    return nothing
end

# ── one direction ────────────────────────────────────────────────────────────────────

"Post-activation vector after the `activation_cnt`-th ReLU, flattened the way
`encode_dependencies` flattens phi_dep, so neuron indices line up."
function _relu_layer_output(nn, x, activation_cnt::Int)
    cnt = 0
    cur = x
    for l in nn.layers
        cur = cur |> l
        if occursin("ReLU", string(typeof(l)))
            cnt += 1
            if cnt == activation_cnt
                return length(size(cur)) == 4 ? (cur |> Flatten([1, 2, 3, 4])) : cur
            end
        end
    end
    return nothing
end

"""
    _audit_direction(m, v_org, v_pert, is_min) -> (bound, val, point)

Solve `min`/`max` of `z - z^p` under both of the paper's early stops, escalating through
the configured budgets until the sign is settled. Returns the proved bound, the best
point's objective (NaN when none), and that point's input pair (nothing when none).
"""
function _audit_direction(m, v_org, v_pert, is_min::Bool)
    tol = dep_audit.tol
    v_obj = @variable(m)
    @constraint(m, v_obj == v_org - v_pert)
    if is_min
        @objective(m, Min, v_obj)
        set_optimizer_attribute(m, "BestObjStop", -tol)
    else
        @objective(m, Max, v_obj)
        set_optimizer_attribute(m, "BestObjStop", tol)
    end
    set_optimizer_attribute(m, "BestBdStop", 0.0)
    # Clear the Cutoff the old-rule replay leaves behind, or it would prune this search.
    set_optimizer_attribute(m, "Cutoff", is_min ? 1e100 : -1e100)
    if dep_audit.threads > 0
        set_optimizer_attribute(m, "Threads", dep_audit.threads)
        set_optimizer_attribute(m, "Seed", 0)
    end

    bnd = NaN
    val = NaN
    point = nothing
    for budget in dep_audit.budgets
        set_optimizer_attribute(m, "TimeLimit", budget)
        optimize!(m)
        st = JuMP.termination_status(m)
        if st == MOI.OPTIMIZE_NOT_CALLED || st == MOI.INVALID_MODEL ||
           st == MOI.NUMERICAL_ERROR || st == MOI.OTHER_ERROR
            break
        end
        bnd = try JuMP.objective_bound(m) catch; NaN end
        if result_count(m) > 0
            val = JuMP.objective_value(m)
            refuted = is_min ? (val <= -tol) : (val >= tol)
            if refuted
                point = try (JuMP.value.(dep_audit.v_in), JuMP.value.(dep_audit.v_in_p))
                        catch; nothing end
            end
        end
        proved  = isfinite(bnd) && (is_min ? bnd > -tol : bnd < tol)
        refuted = isfinite(val) && (is_min ? val <= -tol : val >= tol)
        (proved || refuted) && break
    end
    return bnd, val, point
end

"""
    _old_rule_probe(m, v_org, v_pert, is_min) -> (stamped, value)

The pre-fix solve, replayed exactly: Cutoff 0, the inherited 5 s limit, and the verdict
read off `objective_value`. `stamped` is whether that code would have added the relation.
Guard-true neurons are rare, so this costs one extra solve on a handful of neurons and
turns "the sign is false" into "the old rule actually stamped the false sign".
"""
function _old_rule_probe(m, v_org, v_pert, is_min::Bool)
    tol = dep_audit.tol
    v_obj = @variable(m)
    @constraint(m, v_obj == v_org - v_pert)
    is_min ? @objective(m, Min, v_obj) : @objective(m, Max, v_obj)
    set_optimizer_attribute(m, "Cutoff", 0)
    set_optimizer_attribute(m, "TimeLimit", 5.0)
    # Disable the audit's early stops so this is the old search, not a shortened one.
    set_optimizer_attribute(m, "BestObjStop", is_min ? -1e100 : 1e100)
    set_optimizer_attribute(m, "BestBdStop", is_min ? 1e100 : -1e100)
    optimize!(m)
    # Inf is the sentinel the pre-fix code left l_diff/u_diff at when nothing was found.
    val = result_count(m) > 0 ? JuMP.objective_value(m) : Inf
    stamped = is_min ? ((val != Inf) && (val > -tol)) : (val < tol)
    return stamped, val
end

"Re-evaluate a witness on the network itself, so the claim does not rest on the MIP."
function _check_witness(point, activation_cnt::Int, n::Int)
    point === nothing && return (false, NaN, NaN)
    try
        x, xp = point
        zc = _relu_layer_output(dep_audit.nn, Array{Float64}(x), activation_cnt)
        zp = _relu_layer_output(dep_audit.nn, Array{Float64}(xp), activation_cnt)
        (zc === nothing || zp === nothing) && return (false, NaN, NaN)
        (n > length(zc) || n > length(zp)) && return (false, NaN, NaN)
        return (true, Float64(zc[n]), Float64(zp[n]))
    catch
        return (false, NaN, NaN)
    end
end

function _verdict(bnd, val, opp_bnd, is_min::Bool, tol::Float64)
    # "ge" asks whether z >= z^p holds; its refutation is a point with z - z^p <= -tol,
    # and the opposite sign is proved by the max side being below tol.
    refuted = isfinite(val) && (is_min ? val <= -tol : val >= tol)
    proved  = isfinite(bnd) && (is_min ? bnd > -tol : bnd < tol)
    if refuted
        opp_proved = isfinite(opp_bnd) && (is_min ? opp_bnd < tol : opp_bnd > -tol)
        return opp_proved ? "FLIPPED" : "WRONG"
    end
    proved && return "SAFE"
    return "UNDECIDED"
end

"""
    dep_audit_neuron!(m, v_org, v_pert, activation_cnt, n, guard_ge, guard_le)

Probe both directions on one neuron, classify each sign the pre-fix code could stamp, and
append a CSV row. Adds no dependency constraint: the relations under test must not
prejudge one another.
"""
function dep_audit_neuron!(m, v_org, v_pert, activation_cnt::Int, n::Int,
                           guard_ge::Bool, guard_le::Bool, bnds)
    tol = dep_audit.tol
    if !guard_ge && !guard_le
        # Neither interval guard held, so the pre-fix code solved nothing and stamped
        # nothing here: the neuron is clean by construction and needs no probe.
        _dep_audit_row(activation_cnt, n, bnds, guard_ge, guard_le,
                       NaN, NaN, NaN, NaN, "n/a", "n/a", false, false, NaN, NaN,
                       "n/a", "n/a", false, NaN, NaN, 0.0, "")
        return nothing
    end
    t0 = time()
    min_bnd, min_val, min_pt = _audit_direction(m, v_org, v_pert, true)
    max_bnd, max_val, max_pt = _audit_direction(m, v_org, v_pert, false)
    probe_sec = time() - t0

    v_ge = guard_ge ? _verdict(min_bnd, min_val, max_bnd, true,  tol) : "n/a"
    v_le = guard_le ? _verdict(max_bnd, max_val, min_bnd, false, tol) : "n/a"

    # Replay the pre-fix solve on the same neuron to see whether it really stamped.
    old_ge, old_vmin = guard_ge ? _old_rule_probe(m, v_org, v_pert, true)  : (false, NaN)
    old_le, old_vmax = guard_le ? _old_rule_probe(m, v_org, v_pert, false) : (false, NaN)
    wrongish(v) = (v == "FLIPPED" || v == "WRONG")
    fired_ge = guard_ge ? string(old_ge && wrongish(v_ge)) : "n/a"
    fired_le = guard_le ? string(old_le && wrongish(v_le)) : "n/a"

    point = nothing
    if v_ge == "FLIPPED" || v_ge == "WRONG"
        point = min_pt
    elseif v_le == "FLIPPED" || v_le == "WRONG"
        point = max_pt
    end
    ok, wz, wzp = _check_witness(point, activation_cnt, n)

    wfile = ""
    if point !== nothing
        dep_audit.n_witness += 1
        wfile = string(dep_audit.tag, "_w", dep_audit.n_witness, ".bin")
        try
            serialize(joinpath(dep_audit.out_dir, wfile), point)
        catch
            wfile = ""
        end
    end

    _dep_audit_row(activation_cnt, n, bnds, guard_ge, guard_le, min_bnd, min_val, max_bnd,
                   max_val, v_ge, v_le, old_ge, old_le, old_vmin, old_vmax,
                   fired_ge, fired_le, ok, wz, wzp, probe_sec, wfile)
    return nothing
end

function _dep_audit_row(activation_cnt, n, bnds, guard_ge, guard_le, min_bnd, min_val,
                        max_bnd, max_val, v_ge, v_le, old_ge, old_le, old_vmin, old_vmax,
                        fired_ge, fired_le, ok, wz, wzp, probe_sec, wfile)
    dep_audit.io === nothing && return nothing
    l_o, u_o, l_p, u_p = bnds
    println(dep_audit.io, join(vcat(dep_audit.meta,
        [string(activation_cnt), string(n),
         string(l_o), string(u_o), string(l_p), string(u_p),
         string(guard_ge), string(guard_le),
         string(min_bnd), string(min_val), string(max_bnd), string(max_val),
         v_ge, v_le, string(old_ge), string(old_le),
         string(old_vmin), string(old_vmax), fired_ge, fired_le,
         string(ok), string(wz), string(wzp),
         string(round(probe_sec, digits = 3)), wfile]), ","))
    flush(dep_audit.io)
    return nothing
end
