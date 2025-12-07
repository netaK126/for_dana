
function get_nn(model_path, model_name, w, h, k, c, dataset)

    if model_name == "3x10"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ], "nn",)
    elseif model_name == "10x10"
        is_conv = false
        stride = 0
        layer_number = 5
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
#         fc3 = get_matrix_params_mod(dict, "fc3", (layers_n, c),weight_addition)
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5,
                    ReLU(), fc6, ReLU(), fc7, ReLU(), fc8, ReLU(), fc9, ReLU(), fc10,], "nn",)
    elseif model_name == "cnn0"
        is_conv = true
        stride_to_use_1 = 4
        stride_to_use_2 = 4
        conv_filters = 3
        conv_filters2 = 3
        flatten_num = 12
        model_pth = myunpickle(model_path)
        dict1 = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc1/weight"=>model_pth[5],"fc1/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc2/weight"=>model_pth[7], "fc2/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
        conv1 = get_conv_params(dict1, "conv1", (4, 4, k, conv_filters), expected_stride = stride_to_use_1)
        conv2 = get_conv_params(dict1, "conv2", (3, 3, conv_filters, conv_filters2), expected_stride = stride_to_use_2)
        fc1 = get_matrix_params(dict1, "fc1", (flatten_num, 10))
        fc2 = get_matrix_params(dict1, "fc2", (10, c))
        nn = Sequential( [ conv1,ReLU(),conv2,ReLU(), Flatten([1, 2, 3, 4]), fc1, ReLU(), fc2, ],"nn", )
    end

    if dataset == "mnist"
        mnist = read_datasets("mnist")
        compute_acc(mnist, nn, is_conv, w, h, k)
    end

    return nn
end