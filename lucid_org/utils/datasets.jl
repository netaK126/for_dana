function get_dataset_params( dataset )
    if dataset == "adult"
        dim = 21
        c_ = 2
    elseif dataset == "crypto"
        dim = 7
        c_ = 2
    elseif dataset == "twitter"
        dim = 15
        c_ = 2
     elseif dataset == "credit"
        dim = 23
        c_ = 2
    end
    return dim,c_
end


