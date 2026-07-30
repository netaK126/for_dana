function get_dataset_params( dataset )
    if dataset == "mnist"
        w_,h_,k_=28,28,1
        c_ = 10
    elseif dataset == "fmnist"
        w_,h_,k_=28,28,1
        c_ = 10
    elseif dataset == "cifar10"
        w_,h_,k_=32,32,3
        c_ = 10
    elseif dataset == "har"
        # Human Activity Recognition (Anguita et al. 2013): 561 sensor
        # features, 6 activity classes.
        w_,h_,k_=561,1,1
        c_ = 6
    end
    return w_,h_,k_,c_
end


# Per-coordinate input box for the verification region, returned as two arrays
# of shape (1, w_, h_, k_) so they index directly with the CartesianIndices
# used by the perturbation encoders.
#
# The box is NOT hard-coded: it is read from the `<model>_box.txt` sidecar that
# utils/nnet_to_pickle.py derives from the source .nnet (two comma-separated
# lines: lo then hi). HAR uses [-1,1]^561 (Paulsen et al., ICSE'20 / TwoSafe).
# Image models have no sidecar, so this returns `nothing` and callers keep the
# historical [0,1] path.
function get_input_box( dataset, model_path, w_, h_, k_ )
    box_path = splitext(model_path)[1] * "_box.txt"
    if !isfile(box_path)
        return nothing
    end
    lines = readlines(box_path)
    # NB: `Base.split` is qualified because MIPVerify's `@enum ReLUType split …`
    # (core_ops.jl) shadows `split` with an enum value in this scope.
    lo_vec = parse.(Float64, Base.split(strip(lines[1]), ","))
    hi_vec = parse.(Float64, Base.split(strip(lines[2]), ","))
    n = w_ * h_ * k_
    @assert length(lo_vec) == n "input box lo length $(length(lo_vec)) != w*h*k=$n ($box_path)"
    @assert length(hi_vec) == n "input box hi length $(length(hi_vec)) != w*h*k=$n ($box_path)"
    lo = reshape(lo_vec, (1, w_, h_, k_))
    hi = reshape(hi_vec, (1, w_, h_, k_))
    return lo, hi
end
