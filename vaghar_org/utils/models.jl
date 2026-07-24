
function get_nn(model_path, model_name, w, h, k, c, dataset)

    if model_name == "2x10"
        is_conv = false
        stride = 0
        layer_number = 2
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ], "nn",)
    elseif model_name == "3x10"
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
    elseif model_name == "3x50"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 50
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ], "nn",)
    elseif model_name == "3x100"
        is_conv = false
        stride = 0
        layer_number = 3
        layers_n = 100
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
             "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ], "nn",)
    elseif model_name == "4x10"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, layers_n))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4 ], "nn",)
    elseif model_name == "5x10"
        is_conv = false
        stride = 0
        layer_number = 5
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, layers_n))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, layers_n))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5, ], "nn",)
    elseif model_name == "6x10"
        is_conv = false
        stride = 0
        layer_number = 6
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, layers_n))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, layers_n))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, layers_n))
        fc6 = get_matrix_params(dict, "fc6", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5, ReLU(), fc6, ], "nn",)
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
    elseif model_name == "4x10"
        is_conv = false
        stride = 0
        layer_number = 4
        layers_n = 10
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, layers_n))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4 ], "nn",)
    elseif model_name == "6x100"
        is_conv = false
        stride = 0
        layer_number = 6
        layers_n = 100
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, layers_n))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, layers_n))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, layers_n))
        fc6 = get_matrix_params(dict, "fc6", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5, ReLU(), fc6, ], "nn",)
    elseif model_name == "9x200"
        is_conv = false
        stride = 0
        layer_number = 9
        layers_n = 200
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1],"fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3],"fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5],"fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7],"fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9],"fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
            "fc8/weight"=>model_pth[15],"fc8/bias" => reshape(model_pth[16],(1,length(model_pth[16]))),
            "fc9/weight"=>model_pth[17],"fc9/bias" => reshape(model_pth[18],(1,length(model_pth[18]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, layers_n))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, layers_n))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, layers_n))
        fc6 = get_matrix_params(dict, "fc6", (layers_n, layers_n))
        fc7 = get_matrix_params(dict, "fc7", (layers_n, layers_n))
        fc8 = get_matrix_params(dict, "fc8", (layers_n, layers_n))
        fc9 = get_matrix_params(dict, "fc9", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5, ReLU(), fc6, ReLU(), fc7, ReLU(), fc8, ReLU(), fc9, ], "nn",)
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
    elseif model_name == "cnn1"
        is_conv = true
        stride_to_use_1 = 3
        stride_to_use_2 = 3
        conv_filters = 6
        conv_filters2 = 6
        flatten_num = 54
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
    elseif model_name == "cnn2"
        is_conv = true
        stride_to_use_1 = 1
        stride_to_use_2 = 3
        conv_filters = 3
        conv_filters2 = 3
        # Flattened conv-output size derived from the input geometry (valid
        # padding: out = div(in - kernel, stride) + 1), so cnn2 adapts across
        # datasets: 28x28 -> 192 (MNIST/Fashion-MNIST), 32x32 -> 243 (CIFAR-10).
        w1 = div(w - 4, stride_to_use_1) + 1
        h1 = div(h - 4, stride_to_use_1) + 1
        w2 = div(w1 - 3, stride_to_use_2) + 1
        h2 = div(h1 - 3, stride_to_use_2) + 1
        flatten_num = conv_filters2 * w2 * h2
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
    elseif model_name == "cnn3"
        is_conv = true
        stride_to_use = 3
        conv_filters = 6
        flatten_num = 54
        model_pth = myunpickle(model_path)
        dict1 = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "conv3/weight"=>model_pth[5], "conv3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "conv4/weight"=>model_pth[7], "conv4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
        "conv5/weight"=>model_pth[9], "conv5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
        "fc1/weight"=>model_pth[11],"fc1/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
        "fc2/weight"=>model_pth[13], "fc2/bias" => reshape(model_pth[14],(1,length(model_pth[14]))),
        "fc3/weight"=>model_pth[15], "fc3/bias" => reshape(model_pth[16],(1,length(model_pth[16]))))
        conv1 = get_conv_params(dict1, "conv1", (4, 4, k, conv_filters), expected_stride = stride_to_use)
        conv2 = get_conv_params(dict1, "conv2", (3, 3, conv_filters, conv_filters), expected_stride = stride_to_use)
        conv3 = get_conv_params(dict1, "conv3", (3, 3, conv_filters, conv_filters), expected_stride = stride_to_use)
        conv4 = get_conv_params(dict1, "conv4", (3, 3, conv_filters, conv_filters), expected_stride = stride_to_use)
        conv5 = get_conv_params(dict1, "conv5", (3, 3, conv_filters, conv_filters), expected_stride = stride_to_use)
        fc1 = get_matrix_params(dict1, "fc1", (flatten_num, 10))
        fc2 = get_matrix_params(dict1, "fc2", (10, 10))
        fc3 = get_matrix_params(dict1, "fc3", (10, c))
        nn = Sequential( [ conv1,ReLU(),conv2,ReLU(),conv3,ReLU(),conv4,ReLU(),conv5,ReLU(), Flatten([1, 2, 3, 4]), fc1, ReLU(), fc2, ReLU(), fc3, ],"nn", )
    elseif model_name == "cnn4"
        is_conv = true
        stride_to_use_1 = 1
        stride_to_use_2 = 3
        conv_filters = 8
        conv_filters2 = 8
        # Wider sibling of cnn2 (8 conv channels instead of 3), 2 CONV + 2 FC.
        # Flattened conv-output size derived from the input geometry (valid
        # padding: out = div(in - kernel, stride) + 1), so cnn4 adapts across
        # datasets: 28x28 -> 512 (MNIST/Fashion-MNIST), 32x32 -> 648 (CIFAR-10).
        w1 = div(w - 4, stride_to_use_1) + 1
        h1 = div(h - 4, stride_to_use_1) + 1
        w2 = div(w1 - 3, stride_to_use_2) + 1
        h2 = div(h1 - 3, stride_to_use_2) + 1
        flatten_num = conv_filters2 * w2 * h2
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
    elseif model_name == "cnn5"
        is_conv = true
        stride_to_use_1 = 1
        stride_to_use_2 = 4
        conv_filters = 10
        conv_filters2 = 10
        # Larger 2-CONV + 2-FC net (10 channels), sized so the CIFAR-10 (32x32x3)
        # instance has ~8910 hidden ReLU neurons. Flattened conv-output size is
        # derived from the input geometry (valid padding: out = div(in-kernel,stride)+1)
        # so cnn5 adapts across datasets: 32x32 -> 490 (CIFAR-10), 28x28 -> 360.
        w1 = div(w - 4, stride_to_use_1) + 1
        h1 = div(h - 4, stride_to_use_1) + 1
        w2 = div(w1 - 4, stride_to_use_2) + 1
        h2 = div(h1 - 4, stride_to_use_2) + 1
        flatten_num = conv_filters2 * w2 * h2
        model_pth = myunpickle(model_path)
        dict1 = Dict{String,Any}("conv1/weight"=>model_pth[1], "conv1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
        "conv2/weight"=>model_pth[3], "conv2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
        "fc1/weight"=>model_pth[5],"fc1/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
        "fc2/weight"=>model_pth[7], "fc2/bias" => reshape(model_pth[8],(1,length(model_pth[8]))))
        conv1 = get_conv_params(dict1, "conv1", (4, 4, k, conv_filters), expected_stride = stride_to_use_1)
        conv2 = get_conv_params(dict1, "conv2", (4, 4, conv_filters, conv_filters2), expected_stride = stride_to_use_2)
        fc1 = get_matrix_params(dict1, "fc1", (flatten_num, 10))
        fc2 = get_matrix_params(dict1, "fc2", (10, c))
        nn = Sequential( [ conv1,ReLU(),conv2,ReLU(), Flatten([1, 2, 3, 4]), fc1, ReLU(), fc2, ],"nn", )
    elseif model_name == "acas"
        # ACAS Xu (Julian et al. 2016): 5 inputs -> 6 hidden ReLU layers of 50
        # -> 5 outputs (300 hidden neurons). Weights from the standard
        # ACASXU_run2a_*.nnet, converted by utils/nnet_to_pickle.py (output
        # layer negated so argmax == ACAS's native argmin advisory). The saved
        # net consumes normalized inputs; the verification box lives in
        # get_input_box(...).
        is_conv = false
        stride = 0
        layer_number = 7
        layers_n = 50
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1], "fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3], "fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))),
            "fc3/weight"=>model_pth[5], "fc3/bias" => reshape(model_pth[6],(1,length(model_pth[6]))),
            "fc4/weight"=>model_pth[7], "fc4/bias" => reshape(model_pth[8],(1,length(model_pth[8]))),
            "fc5/weight"=>model_pth[9], "fc5/bias" => reshape(model_pth[10],(1,length(model_pth[10]))),
            "fc6/weight"=>model_pth[11],"fc6/bias" => reshape(model_pth[12],(1,length(model_pth[12]))),
            "fc7/weight"=>model_pth[13],"fc7/bias" => reshape(model_pth[14],(1,length(model_pth[14]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, layers_n))
        fc3 = get_matrix_params(dict, "fc3", (layers_n, layers_n))
        fc4 = get_matrix_params(dict, "fc4", (layers_n, layers_n))
        fc5 = get_matrix_params(dict, "fc5", (layers_n, layers_n))
        fc6 = get_matrix_params(dict, "fc6", (layers_n, layers_n))
        fc7 = get_matrix_params(dict, "fc7", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ReLU(), fc3, ReLU(), fc4, ReLU(), fc5, ReLU(), fc6, ReLU(), fc7, ], "nn",)
    elseif model_name == "har"
        # Human Activity Recognition (Anguita et al. 2013): 561 inputs -> one
        # hidden ReLU layer of 500 -> 6 classes. Weights from the ReluDiff
        # HAR.nnet (Paulsen et al., ICSE'20), converted by
        # utils/nnet_to_pickle.py. Verification box is [-1,1]^561.
        is_conv = false
        stride = 0
        layer_number = 2
        layers_n = 500
        model_pth = myunpickle(model_path)
        dict = Dict{String,Any}( "fc1/weight"=>model_pth[1], "fc1/bias" => reshape(model_pth[2],(1,length(model_pth[2]))),
            "fc2/weight"=>model_pth[3], "fc2/bias" => reshape(model_pth[4],(1,length(model_pth[4]))))
        fc1 = get_matrix_params(dict, "fc1", (w*h*k, layers_n))
        fc2 = get_matrix_params(dict, "fc2", (layers_n, c))
        nn = Sequential( [ Flatten([1, 3, 2, 4]),fc1, ReLU(), fc2, ], "nn",)
    end

    if dataset == "mnist"
        mnist = read_datasets("mnist")
        compute_acc(mnist, nn, is_conv, w, h, k)
    end

    return nn
end