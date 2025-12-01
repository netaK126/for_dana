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
    elseif dataset == "mnist"
        w,h,k=28,28,1
        dim = w*h*k
        c_ = 10
    elseif dataset == "fmnist"
        w,h,k=28,28,1
        dim = w*h*k
        c_ = 10
    elseif dataset == "cifar10"
        w,h,k=32,32,3
        dim = w*h*k
        c_ = 10
    
    end
    return dim,c_
end

