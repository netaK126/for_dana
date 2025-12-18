
function get_nn(model_path, model_name, dim, c, dataset)

    if model_name == "2x10"
        is_conv = false
        stride = 0
        layer_number = 2
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
    end
    model_pth = myunpickle(model_path)
    if occursin("2x",model_name)
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

function get_nn_hyper(model_path, model_name, dim, c, dataset, hypers_dir_path, is_deps, chunks = 0)

    if model_name == "2x10"
        is_conv = false
        stride = 0
        layer_number = 2
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
    end

    model_pth = myunpickle(model_path)
    if occursin("2x",model_name)
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
