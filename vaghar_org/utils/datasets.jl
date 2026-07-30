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


# The network's input domain — one [lo, hi] per input coordinate — read from the model_box.txt next to the weights (HAR: [-1,1]^561); image models have no such file, so callers keep [0,1].
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
