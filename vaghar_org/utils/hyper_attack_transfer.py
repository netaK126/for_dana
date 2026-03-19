import argparse
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split
import os
import sys
from models import *
from tqdm import tqdm


def confidence_margin(output, c):
    """C(N, x, c) = N(x)[c] - max_{k!=c} N(x)[k]"""
    output_tmp = output.clone()
    output_tmp[:, c] = float('-inf')
    max_not_c, _ = output_tmp.max(dim=1)
    return output[:, c] - max_not_c


def update_attack(X, eps_pgd, alpha, size_, perturbation_type, dims):
    if perturbation_type == "brightness" or perturbation_type == "contrast":
        eps_pgd += alpha * eps_pgd.grad.sign()
        eps_pgd = torch.clamp(eps_pgd, 0, size_[0])
        eps_pgd.requires_grad = True
    elif perturbation_type == "linf" or perturbation_type == "patch":
        eps_pgd += alpha * eps_pgd.grad.sign()
        eps_pgd = torch.clamp(eps_pgd, -size_[0], size_[0])
        eps_pgd.requires_grad = True
    return eps_pgd


def create_attacked(X, eps, perturbation_type, size_, dims):
    if perturbation_type == "linf":
        Xout = torch.clamp(X + eps, 0, 1)
    elif perturbation_type == "brightness":
        Xout = X + eps
    elif perturbation_type == "contrast":
        Xout = X * (1 + eps)
    elif perturbation_type == "patch":
        row_start = int(size_[1]) - 1
        col_start = int(size_[2]) - 1
        length = int(size_[3])
        Xout = X + 0.0
        Xout[:, :, row_start:row_start+length, col_start:col_start+length] = torch.clamp(
            X[:, :, row_start:row_start+length, col_start:col_start+length] + eps, 0, 1)
    else:
        Xout = torch.clamp(X + eps, 0, 1)
    return Xout


def define_attack(perturbation_type, size_, M, dims, device):
    if perturbation_type == "patch":
        length = int(size_[3])
        eps_pgd = torch.Tensor(M, dims[0], length, length).to(device)
        eps_pgd = eps_pgd - eps_pgd + size_[0] / 2
        eps_pgd.requires_grad = True
    elif perturbation_type == "linf":
        eps_pgd = torch.Tensor(M, dims[0], dims[1], dims[2]).to(device)
        eps_pgd = eps_pgd - eps_pgd + size_[0] / 2
        eps_pgd.requires_grad = True
    else:
        eps_pgd = torch.Tensor(M, 1, 1, 1).to(device)
        eps_pgd = eps_pgd - eps_pgd + size_[0] / 2
        eps_pgd.requires_grad = True
    return eps_pgd


def load_dataset(dataset):
    if dataset == "mnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.MNIST(root='./data/', train=False, transform=transform, download=True)
    elif dataset == "fmnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.FashionMNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.FashionMNIST(root='./data/', train=False, transform=transform, download=True)
    elif dataset == "svhn":
        h_dim, w_dim, k_dim = 32, 32, 3
        transform = transforms.Compose([transforms.ToTensor()])
        dataset_obj = dsets.SVHN(root='./data/', transform=transform, download=True)
        val_size = 12000
        train_size = len(dataset_obj) - 12000
        trainset, testset = random_split(dataset_obj, [train_size, val_size])
    elif dataset == "cifar10":
        h_dim, w_dim, k_dim = 32, 32, 3
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.CIFAR10(root='./data/', train=False, transform=transform, download=True)
    return trainset, testset, (k_dim, h_dim, w_dim)


def load_model(model_arch, model_path, device):
    if model_arch == "2x10":
        model = FNN_2_10()
    elif model_arch == "3x10":
        model = FNN_3_10()
    elif model_arch == "4x10":
        model = FNN_4_10()
    elif model_arch == "3x50":
        model = FNN_3_50()
    elif model_arch == "10x10":
        model = FNN_10_10()
    elif model_arch == "5x10":
        model = FNN_5_10()
    elif model_arch == "6x10":
        model = FNN_6_10()
    elif model_arch == "cnn0":
        model = CNN0()
    elif model_arch == "cnn1":
        model = CNN1()
    elif model_arch == "cnn2":
        model = CNN2()
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    return model


def extract_activations(model, x, model_name):
    """Extract per-ReLU-layer activations (post-ReLU, pre-sign for hint generation)."""
    activations = []
    if "4x" in model_name:
        x_flat = x.reshape(-1, 784)
        x1 = F.relu(model.fc1(x_flat))
        x2 = F.relu(model.fc2(x1))
        x3 = F.relu(model.fc3(x2))
        activations = [x1, x2, x3]
    elif "3x" in model_name and "cnn" not in model_name:
        x_flat = x.reshape(-1, 784)
        x1 = F.relu(model.fc1(x_flat))
        x2 = F.relu(model.fc2(x1))
        activations = [x1, x2]
    elif "5x" in model_name:
        x_flat = x.reshape(-1, 784)
        x1 = F.relu(model.fc1(x_flat))
        x2 = F.relu(model.fc2(x1))
        x3 = F.relu(model.fc3(x2))
        x4 = F.relu(model.fc4(x3))
        activations = [x1, x2, x3, x4]
    elif "6x" in model_name:
        x_flat = x.reshape(-1, 784)
        x1 = F.relu(model.fc1(x_flat))
        x2 = F.relu(model.fc2(x1))
        x3 = F.relu(model.fc3(x2))
        x4 = F.relu(model.fc4(x3))
        x5 = F.relu(model.fc5(x4))
        activations = [x1, x2, x3, x4, x5]
    elif "10x" in model_name:
        x_flat = x.reshape(-1, 784)
        x1 = F.relu(model.fc1(x_flat))
        x2 = F.relu(model.fc2(x1))
        x3 = F.relu(model.fc3(x2))
        x4 = F.relu(model.fc4(x3))
        x5 = F.relu(model.fc5(x4))
        x6 = F.relu(model.fc6(x5))
        x7 = F.relu(model.fc7(x6))
        x8 = F.relu(model.fc8(x7))
        x9 = F.relu(model.fc9(x8))
        activations = [x1, x2, x3, x4, x5, x6, x7, x8, x9]
    elif "2x" in model_name:
        x_flat = x.reshape(-1, 784)
        x1 = F.relu(model.fc1(x_flat))
        activations = [x1]
    elif "cnn0" in model_name:
        dims = (model.k, model.w, model.h)
        x_r = x.reshape(-1, dims[0], dims[1], dims[2])
        x1 = F.relu(model.conv1(x_r))
        x2 = F.relu(model.conv2(x1))
        x3 = F.relu(model.fc1(model.flatten1(x2)))
        activations = [model.flatten1(x1), model.flatten1(x2), x3]
    elif "cnn1" in model_name:
        dims = (model.k, model.w, model.h)
        x_r = x.reshape(-1, dims[0], dims[1], dims[2])
        x1 = F.relu(model.conv1(x_r))
        x2 = F.relu(model.conv2(x1))
        x3 = F.relu(model.fc1(model.flatten1(x2)))
        activations = [model.flatten1(x1), model.flatten1(x2), x3]
    return activations


def build_str_transfer(layer_data, rel_layer, abs_layer, network_version, th=0.01):
    """Generate variable names matching JuMP naming:
    {network_version}a_layerCount{rel_layer}_neuronCount{i+1}_{abs_layer}_{i+1}
    """
    bools = ""
    strings = ""
    for i_c, c in enumerate(layer_data):
        if c.item() > 1 - th:
            bools += "1,"
        elif c.item() < th:
            bools += "0,"
        else:
            bools += "-1,"
        strings += network_version + "a_layerCount" + str(rel_layer) + \
                   "_neuronCount" + str(i_c + 1) + "_" + \
                   str(abs_layer) + "_" + str(i_c + 1) + ","
    return bools, strings


def create_hyper_input(model1, source, trainset, testset, M, dims, device, delta1):
    """Sample inputs classified as source by model1 with sufficient confidence."""
    train_images = [image for image, _ in trainset]
    train_images = torch.stack(train_images).to(device)
    test_images = [image for image, _ in testset]
    test_images = torch.stack(test_images).to(device)
    random_images = torch.rand(len(trainset) + len(testset), dims[0], dims[1], dims[2]).to(device)
    all_samples = torch.cat((random_images, train_images, test_images), dim=0).to(device)

    with torch.no_grad():
        classification = model1(all_samples)
    _, predicted_labels = torch.max(classification, dim=1)
    indices_of_s = (predicted_labels == source).nonzero().squeeze()

    if indices_of_s.dim() == 0:
        indices_of_s = indices_of_s.unsqueeze(0)

    source_samples_classification = classification[indices_of_s]
    c_margins = confidence_margin(source_samples_classification, source)

    # Sort by confidence margin (descending)
    _, sorted_indices = c_margins.sort(descending=True)
    sorted_indices_of_s = indices_of_s[sorted_indices]

    step_size = max(1, len(sorted_indices_of_s) // M)
    uniform_indices = sorted_indices_of_s[::step_size][:M]
    hyper_input = all_samples[uniform_indices]
    return hyper_input


def attack_transfer(model1, model2, X, source_, target_, device, token_signature,
                    model_name, dims, delta1, c_tag_mode,
                    type_="linf", size_=[0.05], iterations=500, alpha=0.01,
                    lambda_0=1.01, K_max=500, model_name2=None, delta_diff_positive=True):
    model1.eval()
    model2.eval()
    M = len(X)
    X_pgd = X.clone().detach()
    X_pgd.requires_grad = True
    eps_pgd = define_attack(type_, size_, M, dims, device)

    c_pert = source_ if c_tag_mode else target_

    for t in tqdm(range(iterations), desc="Transfer Attack"):
        # Clean outputs
        out_n1 = model1(X_pgd)
        out_n2 = model2(X_pgd)
        c_n1_x = confidence_margin(out_n1, source_)
        c_n2_x = confidence_margin(out_n2, source_)
        delta_diff = c_n2_x - c_n1_x

        # Perturbed outputs
        X_pert = create_attacked(X_pgd, eps_pgd, type_, size_, dims)
        out_n2_p = model2(X_pert)
        c_n2_xp = confidence_margin(out_n2_p, c_pert)
        gap_pert = c_n2_xp
        if N1_P_VERSION:
            out_n1_p = model1(X_pert)
            c_n1_xp = confidence_margin(out_n1_p, c_pert)
            gap_pert = gap_pert - c_n1_xp

        # Handle NaNs
        nan_mask = torch.isnan(gap_pert) | torch.isnan(delta_diff)
        delta_diff[nan_mask] = 0
        gap_pert[nan_mask] = 0

        # Adaptive lambda scaling
        lambdas_ = torch.tensor(
            (torch.absolute(delta_diff) / (torch.absolute(gap_pert) + 1e-9)).detach().cpu().numpy()
        ).to(device)
        lambdas_.requires_grad = False

        # Loss: maximize delta_diff while satisfying gap constraint
        if c_tag_mode:
            # gap_pert must be positive: maximize both delta_diff and gap_pert
            loss = delta_diff + lambda_0 * lambdas_ * gap_pert
        else:
            # gap_pert must be negative: maximize delta_diff, minimize gap_pert
            loss = delta_diff - lambda_0 * lambdas_ * gap_pert

        total_loss = torch.sum(loss)
        model1.zero_grad()
        model2.zero_grad()
        total_loss.backward()

        if t == iterations - 1:
            break

        with torch.no_grad():
            X_pgd += alpha * X_pgd.grad.sign()
            X_pgd = torch.clamp(X_pgd, 0, 1)
            X_pgd.requires_grad = True
            eps_pgd = update_attack(X_pgd, eps_pgd, alpha, size_, type_, dims)

    # Feasibility filter
    with torch.no_grad():
        out_n1 = model1(X_pgd)
        out_n2 = model2(X_pgd)
        c_n1_x = confidence_margin(out_n1, source_)
        c_n2_x = confidence_margin(out_n2, source_)
        delta_diff_final = c_n2_x - c_n1_x

        X_pert = create_attacked(X_pgd, eps_pgd, type_, size_, dims)
        out_n2_p = model2(X_pert)
        c_n2_xp = confidence_margin(out_n2_p, c_pert)
        gap_pert_final = c_n2_xp
        if N1_P_VERSION:
            out_n1_p = model1(X_pert)
            c_n1_xp = confidence_margin(out_n1_p, c_pert)
            gap_pert_final = gap_pert_final - c_n1_xp

        # Check N1 classifies as source
        _, max_labels_n1 = out_n1.max(dim=1)
        n1_correct = (max_labels_n1 == source_)

        # Feasibility conditions
        feasible = n1_correct & (c_n1_x >= delta1)
        if delta_diff_positive:
            feasible = feasible & (delta_diff_final >= 0)
        if c_tag_mode:
            feasible = feasible & (gap_pert_final < 0)
        else:
            feasible = feasible & (gap_pert_final > 0)

    s_indices = feasible.nonzero()
    k_to_use = min(K_max, len(s_indices))
    best_val = torch.Tensor([0])

    if k_to_use > 0:
        if k_to_use > 1:
            s_indices = s_indices.squeeze()
        values, indices = torch.topk(delta_diff_final[s_indices], k=k_to_use)
        best_val = values[0]
        best_idx = s_indices[indices[0]]

        # Extract activations for the best sample
        best_x = X_pgd[best_idx].unsqueeze(0)
        best_xp = create_attacked(best_x, eps_pgd[best_idx].unsqueeze(0), type_, size_, dims)
        

        mn2 = model_name2 if model_name2 else model_name
        act_n1_x = extract_activations(model1, best_x, model_name)
        act_n2_x = extract_activations(model2, best_x, mn2)
        act_n2_xp = extract_activations(model2, best_xp, mn2)
        K1 = len(act_n1_x)  # ReLU layers in N1
        K2 = len(act_n2_x)  # ReLU layers in N2
        # Absolute layer offsets in layers_info_dict:
        # N1(x): 0, N2(x): K1, N1(x_p): K1+K2, N2(x_p): K1+K2+K1 (or K1+K2 without n1_p)
        n2_pert_offset = (K1 + K2 + K1) if N1_P_VERSION else (K1 + K2)
        list_to_iterate = [
            ("n1_org", act_n1_x, 0),
            ("n2_org", act_n2_x, K1),
            ("n2_pert", act_n2_xp, n2_pert_offset),
        ]
        if N1_P_VERSION:
            act_n1_xp = extract_activations(model1, best_xp, model_name)
            list_to_iterate.append(("n1_pert", act_n1_xp, K1 + K2))

        # Build hint strings for all 4 network copies
        bools = ""
        strings = ""
        for nv, acts, abs_offset in list_to_iterate:
            for rel_layer_idx, layer_data in enumerate(acts):
                rel_layer = rel_layer_idx + 1
                abs_layer = abs_offset + rel_layer
                # Mean over batch dim (should be 1 here), then sign
                layer_mean = torch.mean(torch.sign(layer_data), dim=0)
                b, s = build_str_transfer(layer_mean, rel_layer, abs_layer, nv)
                bools += b
                strings += s

        # Remove trailing commas
        bools = bools[:-1]
        strings = strings[:-1]

        with open("/tmp/strings_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as f:
            f.write(strings)
        with open("/tmp/booleans_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as f:
            f.write(bools)
    else:
        # No feasible solution found
        with open("/tmp/strings_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as f:
            f.write("")
        with open("/tmp/booleans_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as f:
            f.write("")
        with open("/tmp/fail_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as f:
            f.write("")

    return best_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper Attack for Transfer Proof',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="mnist", help='dataset')
    parser.add_argument('--source', type=float, default=0, help='source class (0-indexed)')
    parser.add_argument('--target', type=float, default=1, help='target class (0-indexed)')
    parser.add_argument('--token', type=str, default="04082021", help='token signature')
    parser.add_argument('--model', type=str, default="4x10", help='model architecture for N1')
    parser.add_argument('--model2', type=str, default="", help='model architecture for N2 (if different from N1)')
    parser.add_argument('--model_path', type=str, default="./models/model1.pth", help='path to N1 weights')
    parser.add_argument('--model_path2', type=str, default="./models/model2.pth", help='path to N2 weights')
    parser.add_argument('--perturbation', type=str, default="linf", help='perturbation type')
    parser.add_argument('--perturbation_size', type=str, default="0.05", help='perturbation size')
    parser.add_argument('--delta1', type=float, default=0.0, help='delta_1 from VHAGaR')
    parser.add_argument('--c_tag_mode', type=str, default="true", help='true or false')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--M', type=int, default=1000, help='Number of samples')
    parser.add_argument('--itr', type=int, default=100, help='Number of PGD iterations')
    parser.add_argument('--alpha', type=float, default=0.01, help='PGD step size')
    parser.add_argument('--n1_p_mode', type=bool, default=True, help='n1_p_mode')
    parser.add_argument('--delta_diff_positive', type=bool, default=True, help='force delta_diff to be positive')
    args = parser.parse_args()
    N1_P_VERSION = args.n1_p_mode
    source = int(args.source)
    target = int(args.target)
    token_signature = args.token
    model_arch = args.model
    model_arch2 = args.model2 if args.model2 else args.model
    model_path = args.model_path
    model_path2 = args.model_path2
    perturbation_type = args.perturbation
    perturbation_size_to_parse = args.perturbation_size.split(",")
    perturbation_size = [float(i) for i in perturbation_size_to_parse]
    delta1 = args.delta1
    c_tag_mode = args.c_tag_mode.lower() == "true"
    dataset = args.dataset
    M = args.M
    iterations = args.itr
    alpha = args.alpha

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cpu")#torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Transfer attack - source:", source, "target:", target,
          "model1:", model_arch, "model2:", model_arch2,
          "perturbation:", perturbation_type, "size:", perturbation_size,
          "delta1:", delta1, "c_tag_mode:", c_tag_mode)

    trainset, testset, dims = load_dataset(dataset)
    model1 = load_model(model_arch, model_path, device)
    model2 = load_model(model_arch2, model_path2, device)
    X = create_hyper_input(model1, source, trainset, testset, M, dims, device, delta1)

    best_val = attack_transfer(model1, model2, X, source, target, device,
                               token_signature, model_arch, dims, delta1, c_tag_mode,
                               perturbation_type, perturbation_size, iterations, alpha,
                               model_name2=model_arch2, delta_diff_positive=args.delta_diff_positive)

    print("best_val", best_val.item())
    with open("/tmp/best_val_" + str(source) + "_" + str(target) + "_" + str(token_signature) + ".txt", "w") as f:
        f.write(str(best_val.item()))
