
global did_change
global after_moderation_dict
did_change = false

function add_random_noise_to_dict!(dict_to_modify::Dict, e::Float64)
    global did_change
    global after_moderation_dict
    if did_change
        dict_to_modify = deepcopy(after_moderation_dict)
    else
        for (key, matrix) in dict_to_modify
            if occursin("bias",key)
                continue
            end
            noise = 2 * e * rand(Float32, size(matrix)) .- e
            dict_to_modify[key] .= matrix .+ noise
        end
        after_moderation_dict = deepcopy(dict_to_modify)
        did_change = true
    end
    return dict_to_modify
end

function create_modified_nn(model_path, model_name, dim, c, dataset)
    if model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 10
    elseif model_name == "5x10"
        is_conv = false
        stride = 0
        layer_number = 5
        layers_n = 10
    elseif model_name == "10x10"
        is_conv = false
        stride = 0
        layer_number = 10
        layers_n = 10
    end
    noise = 0.000002
    model_pth = myunpickle(model_path)
    if occursin("3x",model_name)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        dict = add_random_noise_to_dict!(dict, noise)
        model_pth[1]=dict["fc1/weight"]
        model_pth[2]=vec(dict["fc1/bias"])
        model_pth[3]=dict["fc2/weight"]
        model_pth[4]=vec(dict["fc2/bias"])
        model_pth[5]=dict["fc3/weight"]
        model_pth[6]=vec(dict["fc3/bias"])
    elseif occursin("5x",model_name)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
        "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))))
        dict = add_random_noise_to_dict!(dict, noise)
        model_pth[1]=dict["fc1/weight"]
        model_pth[2]=vec(dict["fc1/bias"])
        model_pth[3]=dict["fc2/weight"]
        model_pth[4]=vec(dict["fc2/bias"])
        model_pth[5]=dict["fc3/weight"]
        model_pth[6]=vec(dict["fc3/bias"])
        model_pth[7]=dict["fc4/weight"]
        model_pth[8]=vec(dict["fc4/bias"])
        model_pth[9]=dict["fc5/weight"]
        model_pth[10]=vec(dict["fc5/bias"])
    elseif occursin("10x",model_name)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
        "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
        "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
        "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
        "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
        "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
        "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        dict = add_random_noise_to_dict!(dict, noise)
        model_pth[1]=dict["fc1/weight"]
        model_pth[2]=vec(dict["fc1/bias"])
        model_pth[3]=dict["fc2/weight"]
        model_pth[4]=vec(dict["fc2/bias"])
        model_pth[5]=dict["fc3/weight"]
        model_pth[6]=vec(dict["fc3/bias"])
        model_pth[7]=dict["fc4/weight"]
        model_pth[8]=vec(dict["fc4/bias"])
        model_pth[9]=dict["fc5/weight"]
        model_pth[10]=vec(dict["fc5/bias"])
        model_pth[11]=dict["fc6/weight"]
        model_pth[12]=vec(dict["fc6/bias"])
        model_pth[13]=dict["fc7/weight"]
        model_pth[14]=vec(dict["fc7/bias"])
        model_pth[15]=dict["fc8/weight"]
        model_pth[16]=vec(dict["fc8/bias"])
        model_pth[17]=dict["fc9/weight"]
        model_pth[18]=vec(dict["fc9/bias"])
        model_pth[19]=dict["fc10/weight"]
        model_pth[20]=vec(dict["fc10/bias"])
    end
    py_state_dict = Dict()
    torch = pyimport("torch")
    # pickle = pyimport("pickle")
    for (key, value) in dict
        new_key = replace(key, "/" => ".")
        if endswith(new_key, "weight")
            # transpose weight matrix
            py_state_dict[new_key] = torch.tensor(value')  # ' means transpose in Julia
        elseif endswith(new_key, "bias")
            # flatten bias vector
            py_state_dict[new_key] = torch.tensor(vec(value))
        end
    end

    # Now save the dict instead of model_pth (list)
    modifier_model_path = joinpath(dirname(model_path), splitext(basename(model_path))[1])*"_modified7.pth"
    open(joinpath(dirname(model_path), splitext(basename(model_path))[1])*"_modified7.p","w") do f
        pickle.dump(model_pth, f)
    end
    torch.save(py_state_dict, modifier_model_path)
end

function get_nn(model_path, model_name, dim, c, dataset)

    if model_name == "2x10"
        is_conv = false
        stride = 0
        layer_number = 2
        layers_n = 10
    elseif model_name == "4x10"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 10
    elseif model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 10
    elseif model_name == "2x50"
        is_conv = false
        stride = 0
        layer_number = 2
        layers_n = 50
    elseif model_name == "4x50"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 50
    elseif model_name == "4x25"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 25
    elseif model_name == "4x30"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 30
    elseif model_name == "2x100"
        is_conv = false
        stride = 0
        layer_number = 2
        layers_n = 100
    elseif model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 10
    elseif model_name == "5x10"
        is_conv = false
        stride = 0
        layer_number = 5
        layers_n = 10
    elseif model_name == "10x10"
        is_conv = false
        stride = 0
        layer_number = 10
        layers_n = 10
    end
    model_pth = myunpickle(model_path)
    if occursin("3x",model_name)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        # dict = add_random_noise_to_dict!(dict, 0.001)
        fc1 = get_matrix_params(dict, "fc1", (dim, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ], "nn",)
    elseif occursin("4x",model_name)
        dict = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
        # dict = add_random_noise_to_dict!(dict, 0.001)
        fc1 = get_matrix_params(dict, "fc1", (dim, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, layers_n))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ], "nn",)
    elseif occursin("5x",model_name)
        w,h,k=28,28,1
        is_conv = false
        stride = 0
        layer_number = 5
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[3],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))))
        total_noise = Dict()
        total_noise["fc1/weight"] = 0
        total_noise["fc2/weight"] = 0
        total_noise["fc3/weight"] = 0
#         if weight_addition != 0
#             total_noise = add_random_noise_to_dict!(dict, weight_addition)
#         end
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, c))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, c))
#         fc3 = get_matrix_params_mod(dict, "fc3", (layers_n, c),weight_addition)
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5,], "nn",)
    elseif occursin("10x",model_name)
        w,h,k=28,28,1
        is_conv = false
        stride = 0
        layer_number = 10
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, c))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, c))
        fc6 = get_matrix_params(dict, "fc6", (layers_n, c))
        fc7 = get_matrix_params(dict, "fc7", (layers_n, c))
        fc8 = get_matrix_params(dict, "fc8", (layers_n, c))
        fc9 = get_matrix_params(dict, "fc9", (layers_n, c))
        fc10 = get_matrix_params(dict, "fc10", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5,
                    ReLU(), fc6, ReLU(), fc7, ReLU(), fc8, ReLU(), fc9, ReLU(), fc10], "nn",)
    elseif occursin("3x",model_name)
        w,h,k=28,28,1
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3], "nn",)
    elseif occursin("2x",model_name)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        fc1 = get_matrix_params(dict, "fc1", (dim, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ], "nn",)
    elseif occursin("4x",model_name)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
             "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
             "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))))
        fc1 = get_matrix_params(dict, "fc1", (dim, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, layers_n))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, layers_n))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5, ], "nn",)
    elseif occursin("conv0",model_name)
        is_conv = true
        stride_to_use_1 = 1
        stride_to_use_2 = 1
        conv_filters = 6
        conv_filters2 = 6

        if dataset == "credit"
            flatten_num = 114
        elseif dataset == "adult"
            flatten_num = 102
        elseif dataset == "twitter"
            flatten_num = 66
        elseif dataset == "crypto"
            flatten_num = 18
        end


        model_pth = myunpickle(model_path)
        dict1 = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc1/weight"=>model_pth[5],"fc1/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc2/weight"=>model_pth[7], "fc2/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
        conv1 = get_conv_params(dict1, "conv1", (3, 1, 1, conv_filters), expected_stride = stride_to_use_1)
        conv2 = get_conv_params(dict1, "conv2", (3, 1, conv_filters, conv_filters2), expected_stride = stride_to_use_2)
        fc1 = get_matrix_params(dict1, "fc1", (flatten_num, 2))
        fc2 = get_matrix_params(dict1, "fc2", (2, 2))
        nn = Sequential( [ conv1,ReLU(),conv2,ReLU(), Flatten([1, 2, 3, 4]), fc1, ReLU(), fc2, ],"nn", )

    end
    return nn, is_conv
end


function get_nn_preffix(model_path, model_name, dim, c, dataset, test_neta = false)
    global separation_index
    if model_name == "10x10"
        is_conv = false
        stride = 0
        layer_number = separation_index-1
        layers_n = 10
    elseif model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = separation_index-1
        layers_n = 3
    end
    model_pth = myunpickle(model_path)

    if occursin("10x",model_name)
        w,h,k=28,28,1
        is_conv = false
        stride = 0
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        # println("get_nn_preffix")
        # println(typeof(dict["fc1/weight"]))
        # println(sizeof(dict["fc1/weight"]))
        # exit()
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, c))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, c))
        fc6 = get_matrix_params(dict, "fc6", (layers_n, c))
        fc7 = get_matrix_params(dict, "fc7", (layers_n, c))
        fc8 = get_matrix_params(dict, "fc8", (layers_n, c))
        fc9 = get_matrix_params(dict, "fc9", (layers_n, c))
        fc10 = get_matrix_params(dict, "fc10", (layers_n, c))
        all_layers = [fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10]
        fc_list = []
        push!(fc_list, Flatten([1, 3, 2, 4]))
        for i in 1:10
            if i < separation_index
                push!(fc_list, all_layers[i])  # eval(:fc3) returns the variable fc3
                push!(fc_list, ReLU())
            end
        end

        nn = Sequential(fc_list, "nn",)
    elseif occursin("3x",model_name)
        w,h,k=28,28,1
        is_conv = false
        stride = 0
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        # println("get_nn_preffix")
        # println(typeof(dict["fc1/weight"]))
        # println(sizeof(dict["fc1/weight"]))
        # exit()
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        all_layers = [fc1, fc2, fc3]
        fc_list = []
        push!(fc_list, Flatten([1, 3, 2, 4]))
        for i in 1:3
            if i < separation_index
                push!(fc_list, all_layers[i])  # eval(:fc3) returns the variable fc3
                push!(fc_list, ReLU())
            end
        end

        nn = Sequential(fc_list, "nn",)
    end
    return nn, is_conv
end


function get_nn_suffix(model_path, model_name, dim, c, dataset)
    global separation_index
    if model_name == "10x10"
        is_conv = false
        stride = 0
        layer_number = 10 - separation_index
        layers_n = 10
    elseif model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = 3 - separation_index
        layers_n = 10
    end
    model_pth = myunpickle(model_path)

    if occursin("10x",model_name)
        w,h,k=28,28,1
        is_conv = false
        stride = 0
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, c))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, c))
        fc6 = get_matrix_params(dict, "fc6", (layers_n, c))
        fc7 = get_matrix_params(dict, "fc7", (layers_n, c))
        fc8 = get_matrix_params(dict, "fc8", (layers_n, c))
        fc9 = get_matrix_params(dict, "fc9", (layers_n, c))
        fc10 = get_matrix_params(dict, "fc10", (layers_n, c))
        all_layers = [fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10]
        fc_list = []
        push!(fc_list, Flatten([1, 3, 2, 4]))
        for i in 1:10
            if i >= separation_index
                push!(fc_list, all_layers[i]) 
                if i!=10
                    push!(fc_list, ReLU())
                end
            end
            
        end

        nn = Sequential(fc_list, "nn",)
    elseif occursin("3x",model_name)
        w,h,k=28,28,1
        is_conv = false
        stride = 0
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        all_layers = [fc1, fc2, fc3]
        fc_list = []
        push!(fc_list, Flatten([1, 3, 2, 4]))
        for i in 1:3
            if i >= separation_index
                push!(fc_list, all_layers[i]) 
                if i!=3
                    push!(fc_list, ReLU())
                end
            end
            
        end

        nn = Sequential(fc_list, "nn",)
    end
    return nn, is_conv
end


function get_nn_hyper(model_path, model_name, dim, c, dataset, hypers_dir_path, is_deps, chunks = 0)

    if model_name == "2x10"
        is_conv = false
        stride = 0
        layer_number = 2
        layers_n = 10
    elseif model_name == "4x10"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 10
    elseif model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 10
    elseif model_name == "2x50"
        is_conv = false
        stride = 0
        layer_number = 2
        layers_n = 50
    elseif model_name == "4x50"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 50
    elseif model_name == "4x25"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 25
    elseif model_name == "4x30"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 30
    elseif model_name == "2x100"
        is_conv = false
        stride = 0
        layer_number = 2
        layers_n = 100
    elseif model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 10
    elseif model_name == "10x10"
        is_conv = false
        stride = 0
        layer_number = 10
        layers_n = 10
    end

    model_pth = myunpickle(model_path)
    if occursin("3x",model_name)
        dicto = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
            # dicto = add_random_noise_to_dict!(dicto, 0.001)
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))

        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (dim, layers_n))
        deps1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        deps1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (layers_n, layers_n))
        deps2In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        deps2 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        fc3 = get_hyper_network_params(dict_min, dict_max, "fc3", (layers_n, c))
        if is_deps==1
            nn_second = Sequential( [ Flatten([1, 3, 2, 4]),fc1, deps1In, ReLU(), deps1, fc2, deps2In, ReLU(), deps2, fc3, ], "nn",)
        else
            nn_second = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ], "nn",)
        end
    elseif occursin("4x",model_name)
        dicto = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
            # dicto = add_random_noise_to_dict!(dicto, 0.001)
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))

        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (dim, layers_n))
        deps1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        deps1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (layers_n, layers_n))
        deps2In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        deps2 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        fc3 = get_hyper_network_params(dict_min, dict_max, "fc3", (layers_n, c))
        deps3In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        deps3 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        fc4 = get_hyper_network_params(dict_min, dict_max, "fc4", (layers_n, c))
        if is_deps==1
            nn_second = Sequential( [ Flatten([1, 3, 2, 4]),fc1, deps1In, ReLU(), deps1, fc2, deps2In, ReLU(), deps2, fc3, deps3In, ReLU(), deps3, fc4], "nn",)
        else
            nn_second = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4], "nn",)
        end

    elseif occursin("10x",model_name)
        dicto = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}(
            "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))

        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (dim, layers_n))
        deps1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        deps1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (layers_n, layers_n))
        deps2In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        deps2 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        fc3 = get_hyper_network_params(dict_min, dict_max, "fc3", (layers_n, c))
        deps3In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        deps3 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        fc4 = get_hyper_network_params(dict_min, dict_max, "fc4", (layers_n, c))
        deps4In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc4", (layers_n, layers_n))
        deps4 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc4", (layers_n, layers_n))
        fc5 = get_hyper_network_params(dict_min, dict_max, "fc5", (layers_n, c))
        deps5In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc5", (layers_n, layers_n))
        deps5 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc5", (layers_n, layers_n))
        fc6 = get_hyper_network_params(dict_min, dict_max, "fc6", (layers_n, c))
        deps6In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc6", (layers_n, layers_n))
        deps6 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc6", (layers_n, layers_n))
        fc7 = get_hyper_network_params(dict_min, dict_max, "fc7", (layers_n, c))
        deps7In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc7", (layers_n, layers_n))
        deps7 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc7", (layers_n, layers_n))
        fc8 = get_hyper_network_params(dict_min, dict_max, "fc8", (layers_n, c))
        deps8In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc8", (layers_n, layers_n))
        deps8 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc8", (layers_n, layers_n))
        fc9 = get_hyper_network_params(dict_min, dict_max, "fc9", (layers_n, c))
        deps9In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc9", (layers_n, layers_n))
        deps9 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc9", (layers_n, layers_n))
        fc10 = get_hyper_network_params(dict_min, dict_max, "fc10", (layers_n, c))

        if is_deps==1
            # nn_second = Sequential( [ Flatten([1, 3, 2, 4]),fc1, deps1In, ReLU(), deps1, fc2, deps2In, ReLU(), deps2, fc3, ], "nn",)

            nn_second = Sequential( [ Flatten([1, 3, 2, 4])
                        , fc1, deps1In, ReLU(), deps1
                        , fc2, deps2In, ReLU(), deps2
                        , fc3, deps3In, ReLU(), deps3
                        , fc4, deps4In, ReLU(), deps4
                        , fc5, deps5In, ReLU(), deps5
                        , fc6, deps6In, ReLU(), deps6
                        , fc7, deps7In, ReLU(), deps7
                        , fc8, deps8In, ReLU(), deps8
                        , fc9, deps9In, ReLU(), deps9
                        , fc10], "nn",)
        else
            nn_second = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU()
                        , fc3, ReLU(), fc4, ReLU(), fc5, ReLU()
                        , fc6, ReLU(), fc7, ReLU()
                        , fc8, ReLU(), fc9, ReLU()
                        , fc10], "nn",)
        end
    elseif occursin("2x",model_name)
        dicto = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))

        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))

        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (dim, layers_n))
        deps1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        deps1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (layers_n, layers_n))
        deps2In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        deps2 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        fc3 = get_hyper_network_params(dict_min, dict_max, "fc3", (layers_n, c))
        if is_deps==1
            nn_second = Sequential( [ Flatten([1, 3, 2, 4]),fc1, deps1In, ReLU(), deps1, fc2, deps2In, ReLU(), deps2, fc3, ], "nn",)
        else
            nn_second = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ], "nn",)
        end
    elseif occursin("4x",model_name)
        dicto = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
             "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
             "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))))

        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
             "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
             "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))))

        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
             "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
             "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))))

        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (dim, layers_n))
        deps1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        deps1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (layers_n, layers_n))
        deps2In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        deps2 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        fc3 = get_hyper_network_params(dict_min, dict_max, "fc3", (layers_n, layers_n))
        deps3In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        deps3 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        fc4 = get_hyper_network_params(dict_min, dict_max, "fc4", (layers_n, layers_n))
        deps4In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc4", (layers_n, layers_n))
        deps4 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc4", (layers_n, layers_n))
        fc5 = get_hyper_network_params(dict_min, dict_max, "fc5", (layers_n, c))
        if is_deps ==1
            nn_second = Sequential( [ Flatten([1, 3, 2, 4]),fc1, deps1In, ReLU(), deps1, fc2, deps2In, ReLU(), deps2, fc3, deps3In, ReLU(), deps3, fc4, deps4In, ReLU(), deps4, fc5, ], "nn",)
        else
            nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5, ], "nn",)
        end
    elseif occursin("conv0",model_name)
        is_conv = true
        stride_to_use_1 = 1
        stride_to_use_2 = 1
        conv_filters = 6
        conv_filters2 = 6

        if dataset == "credit"
            flatten_num = 114
        elseif dataset == "adult"
            flatten_num = 102
        elseif dataset == "twitter"
            flatten_num = 66
        elseif dataset == "crypto"
            flatten_num = 18
        end

        model_pth = myunpickle(model_path)
        dicto = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc1/weight"=>model_pth[5],"fc1/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc2/weight"=>model_pth[7], "fc2/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))

        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc1/weight"=>model_pth[5],"fc1/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc2/weight"=>model_pth[7], "fc2/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))

        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc1/weight"=>model_pth[5],"fc1/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc2/weight"=>model_pth[7], "fc2/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))

        conv1 = get_hyper_network_conv_params(dict_min, dict_max, "conv1", (3, 1, 1, conv_filters), expected_stride = stride_to_use_1)
        deps1In = get_hyper_network_conv_deps_paramsIn(dict_min, dict_max, dicto, "conv1", (3, 1, 1, conv_filters), expected_stride = stride_to_use_1)
        deps1   = get_hyper_network_conv_deps_params(dict_min, dict_max, dicto, "conv1", (3, 1, 1, conv_filters), expected_stride = stride_to_use_1)
        conv2 = get_hyper_network_conv_params(dict_min, dict_max, "conv2", (3, 1, conv_filters, conv_filters2), expected_stride = stride_to_use_2)
        deps2In = get_hyper_network_conv_deps_paramsIn(dict_min, dict_max, dicto, "conv2", (3, 1, conv_filters, conv_filters2), expected_stride = stride_to_use_2)
        deps2   = get_hyper_network_conv_deps_params(dict_min, dict_max, dicto, "conv2", (3, 1, conv_filters, conv_filters2), expected_stride = stride_to_use_2)
        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (flatten_num, 2))
        depsfc1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (flatten_num, 2))
        depsfc1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (flatten_num, 2))

        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (2, 2))
        if is_deps == 1
            nn_second = Sequential( [ conv1, deps1In, ReLU(), deps1, conv2, deps2In, ReLU(), deps2, Flatten([1, 2, 3, 4]), fc1, depsfc1In, ReLU(), depsfc1, fc2], "nn",)
        else
            nn_second = Sequential( [ conv1, ReLU(), conv2, ReLU(), Flatten([1, 2, 3, 4]), fc1, ReLU(), fc2], "nn",)
        end
    end
    return nn_second
end



function get_nn_hyper_suffix(model_path, model_name, dim, c, dataset, hypers_dir_path, is_deps, chunks = 0)
    global separation_index
    if model_name == "10x10"
        is_conv = false
        stride = 0
        layer_number = separation_index-1
        layers_n = 10
    elseif model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = separation_index-1
        layers_n = 10
    end

    model_pth = myunpickle(model_path)
    if occursin("10x",model_name)
        dicto = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))

        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (dim, layers_n))
        deps1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        deps1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (layers_n, layers_n))
        deps2In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        deps2 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        fc3 = get_hyper_network_params(dict_min, dict_max, "fc3", (layers_n, c))
        deps3In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        deps3 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        fc4 = get_hyper_network_params(dict_min, dict_max, "fc4", (layers_n, c))
        deps4In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc4", (layers_n, layers_n))
        deps4 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc4", (layers_n, layers_n))
        fc5 = get_hyper_network_params(dict_min, dict_max, "fc5", (layers_n, c))
        deps5In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc5", (layers_n, layers_n))
        deps5 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc5", (layers_n, layers_n))
        fc6 = get_hyper_network_params(dict_min, dict_max, "fc6", (layers_n, c))
        deps6In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc6", (layers_n, layers_n))
        deps6 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc6", (layers_n, layers_n))
        fc7 = get_hyper_network_params(dict_min, dict_max, "fc7", (layers_n, c))
        deps7In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc7", (layers_n, layers_n))
        deps7 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc7", (layers_n, layers_n))
        fc8 = get_hyper_network_params(dict_min, dict_max, "fc8", (layers_n, c))
        deps8In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc8", (layers_n, layers_n))
        deps8 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc8", (layers_n, layers_n))
        fc9 = get_hyper_network_params(dict_min, dict_max, "fc9", (layers_n, c))
        deps9In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc9", (layers_n, layers_n))
        deps9 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc9", (layers_n, layers_n))
        fc10 = get_hyper_network_params(dict_min, dict_max, "fc10", (layers_n, c))

        all_layers_fc = [fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10]
        all_layers_deps = [deps1, deps2, deps3, deps4, deps5, deps6, deps7, deps8, deps9]
        all_layers_depsIn = [deps1In, deps2In, deps3In, deps4In, deps5In, deps6In, deps7In, deps8In, deps9In]
        fc_list = []
        push!(fc_list, Flatten([1, 3, 2, 4]))
        for i in 1:10
            if i >= separation_index
                push!(fc_list, all_layers_fc[i])
                if i!=10
                    if is_deps==1
                        push!(fc_list, all_layers_depsIn[i]) 
                    end
                    push!(fc_list, ReLU())
                    if is_deps==1
                        push!(fc_list, all_layers_deps[i]) 
                    end
                end
            end
            
        end
        nn_second = Sequential(fc_list,"nn")

    elseif occursin("3x",model_name)
        dicto = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))

        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (dim, layers_n))
        deps1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        deps1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (layers_n, layers_n))
        deps2In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        deps2 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        fc3 = get_hyper_network_params(dict_min, dict_max, "fc3", (layers_n, c))

        all_layers_fc = [fc1, fc2, fc3]
        all_layers_deps = [deps1, deps2]
        all_layers_depsIn = [deps1In, deps2In]
        fc_list = []
        push!(fc_list, Flatten([1, 3, 2, 4]))
        for i in 1:3
            if i >= separation_index
                push!(fc_list, all_layers_fc[i])
                if i!=3
                    if is_deps==1
                        push!(fc_list, all_layers_depsIn[i]) 
                    end
                    push!(fc_list, ReLU())
                    if is_deps==1
                        push!(fc_list, all_layers_deps[i]) 
                    end
                end
            end
            
        end
        nn_second = Sequential(fc_list,"nn")
    end
    return nn_second
end


function get_nn_hyper_preffix(main_model_path, model_name, dim, c, dataset, hypers_dir_path, is_deps, test_neta = false, chunks = 0)
    global separation_index
    if model_name == "10x10"
        is_conv = false
        stride = 0
        layer_number = separation_index-1
        layers_n = 10
    elseif model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = separation_index-1
        layers_n = 10
    end

    model_pth = myunpickle(main_model_path)
    if occursin("10x",model_name)
        dicto = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))),
            "fc10/weight"=>model_pth[19],"fc10/bias" => reshape(model_pth[20],(1,length(model_pth[20]))))

        # println("hyper_preffix")
        # println(typeof(dict_min["fc1/bias"]))
        # println(sizeof(dict_min["fc1/bias"]))
        # exit()
        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (dim, layers_n))
        deps1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        deps1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (layers_n, layers_n))
        deps2In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        deps2 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        fc3 = get_hyper_network_params(dict_min, dict_max, "fc3", (layers_n, c))
        deps3In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        deps3 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc3", (layers_n, layers_n))
        fc4 = get_hyper_network_params(dict_min, dict_max, "fc4", (layers_n, c))
        deps4In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc4", (layers_n, layers_n))
        deps4 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc4", (layers_n, layers_n))
        fc5 = get_hyper_network_params(dict_min, dict_max, "fc5", (layers_n, c))
        deps5In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc5", (layers_n, layers_n))
        deps5 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc5", (layers_n, layers_n))
        fc6 = get_hyper_network_params(dict_min, dict_max, "fc6", (layers_n, c))
        deps6In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc6", (layers_n, layers_n))
        deps6 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc6", (layers_n, layers_n))
        fc7 = get_hyper_network_params(dict_min, dict_max, "fc7", (layers_n, c))
        deps7In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc7", (layers_n, layers_n))
        deps7 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc7", (layers_n, layers_n))
        fc8 = get_hyper_network_params(dict_min, dict_max, "fc8", (layers_n, c))
        deps8In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc8", (layers_n, layers_n))
        deps8 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc8", (layers_n, layers_n))
        fc9 = get_hyper_network_params(dict_min, dict_max, "fc9", (layers_n, c))
        deps9In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc9", (layers_n, layers_n))
        deps9 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc9", (layers_n, layers_n))
        fc10 = get_hyper_network_params(dict_min, dict_max, "fc10", (layers_n, c))

        all_layers_fc = [fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9, fc10]
        all_layers_deps = [deps1, deps2, deps3, deps4, deps5, deps6, deps7, deps8, deps9]
        all_layers_depsIn = [deps1In, deps2In, deps3In, deps4In, deps5In, deps6In, deps7In, deps8In, deps9In]
        fc_list = []
        push!(fc_list, Flatten([1, 3, 2, 4]))
        for i in 1:10
            if i < separation_index
                push!(fc_list, all_layers_fc[i])
                if is_deps==1
                    push!(fc_list, all_layers_depsIn[i]) 
                end
                push!(fc_list, ReLU())
                if is_deps==1
                    push!(fc_list, all_layers_deps[i]) 
                end
            end
            
        end
        nn_second = Sequential(fc_list,"nn")

    elseif occursin("3x",model_name)
        dicto = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_min_box.p")
        dict_min = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        model_pth = myunpickle(hypers_dir_path*"/hypernetwork_max_box.p")
        dict_max = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))

        # println("hyper_preffix")
        # println(typeof(dict_min["fc1/bias"]))
        # println(sizeof(dict_min["fc1/bias"]))
        # exit()
        fc1 = get_hyper_network_params(dict_min, dict_max, "fc1", (dim, layers_n))
        deps1In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        deps1 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc1", (layers_n, layers_n))
        fc2 = get_hyper_network_params(dict_min, dict_max, "fc2", (layers_n, layers_n))
        deps2In = get_hyper_network_deps_paramsIn(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        deps2 = get_hyper_network_deps_params(dict_min, dict_max, dicto, "fc2", (layers_n, layers_n))
        fc3 = get_hyper_network_params(dict_min, dict_max, "fc3", (layers_n, c))

        all_layers_fc = [fc1, fc2, fc3]
        all_layers_deps = [deps1, deps2]
        all_layers_depsIn = [deps1In, deps2In]
        fc_list = []
        push!(fc_list, Flatten([1, 3, 2, 4]))
        for i in 1:3
            if i < separation_index
                push!(fc_list, all_layers_fc[i])
                if is_deps==1
                    push!(fc_list, all_layers_depsIn[i]) 
                end
                push!(fc_list, ReLU())
                if is_deps==1
                    push!(fc_list, all_layers_deps[i]) 
                end
            end
            
        end
        nn_second = Sequential(fc_list,"nn")
    end
    return nn_second
end