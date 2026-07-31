import argparse
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split
import os
from models import *
from tqdm import tqdm


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


def create_attacked(X, eps, perturbation_type,size_,dims):
    if perturbation_type == "occ":
        row_start = int(size_[1])-1
        col_start = int(size_[2])-1
        length = int(size_[3])
        Xout = X + size_[0]
        Xout[:, :, row_start:row_start+length, col_start:col_start+length] = 0
    elif perturbation_type == "patch":
        row_start = int(size_[1])-1
        col_start = int(size_[2])-1
        length = int(size_[3])
        Xout = X+0.0
        Xout[:, :, row_start:row_start+length, col_start:col_start+length] = torch.clamp(X[:, :, row_start:row_start+length, col_start:col_start+length]+eps,0,1)
    elif perturbation_type == "rotation":
        angle = int(size_[1])
        Xout = X - X # + size_[0]
        height, width = dims[1],dims[2]
        center = (width // 2, height // 2)
        for i in range(height):
            for j in range(width):
                j_c = j - center[0]
                i_c = i - center[1]
                j_r = j_c * np.cos(angle * np.pi / 180) - i_c * np.sin(angle * np.pi / 180) + center[0]
                i_r = j_c * np.sin(angle * np.pi / 180) + i_c * np.cos(angle * np.pi / 180) + center[1]
                if np.floor(j_r) >= 0 and np.ceil(j_r) < width and np.floor(i_r) >= 0 and np.ceil(i_r) < height:
                    di = i_r-np.floor(i_r)
                    dj = j_r-np.floor(j_r)
                    Xout[:, :, i, j] = (1-di)*(1-dj) * X[:, :, int(np.floor(i_r)), int(np.floor(j_r))]+\
                                       di * (1-dj) * X[:, :, int(np.ceil(i_r)), int(np.floor(j_r))]+\
                                       (1-di) * dj * X[:, :, int(np.floor(i_r)), int(np.ceil(j_r))] +\
                                       di * dj * X[:, :, int(np.ceil(i_r)), int(np.ceil(j_r))]
    elif perturbation_type == "brightness":
        Xout = X+eps
    elif perturbation_type == "translation":
        m = int(size_[1])
        k = int(size_[2])
        padded_img = F.pad(X, (k, 0, m, 0), mode='constant', value=0)
        if m == 0:
            Xout = padded_img[:, :, :, :-k]
        elif k == 0:
            Xout = padded_img[:, :, :-m, :]
        else:
            Xout = padded_img[:, :, :-m, :-k]
    elif perturbation_type == "max":
        # delta_max maximises the source-class margin over the input region with
        # NO perturbation (mip.jl skips set_max_indexes for "max"), so the
        # "attacked" copy is the clean input itself.
        Xout = X + 0.0
    elif perturbation_type == "linf":
        if BENCH_BOX is None:
            Xout = torch.clamp(X+eps, 0, 1)
        else:
            lo, hi = BENCH_BOX
            lo_t = torch.as_tensor(lo, dtype=X.dtype, device=X.device).view(1, *X.shape[1:])
            hi_t = torch.as_tensor(hi, dtype=X.dtype, device=X.device).view(1, *X.shape[1:])
            Xout = torch.max(torch.min(X + eps, hi_t), lo_t)
    elif perturbation_type == "contrast":
        Xout = X*(1+eps)
    return Xout


def define_attack(perturbation_type, size_, M, dims, device):
    if perturbation_type == "patch":
        length =  int(size_[3])
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


def build_str(layer_data, layer_number, num_relu_layers, th=0.01):
    bools = ""
    strings = ""
    if layer_number <= num_relu_layers:
        version = "org"
        lc = layer_number
    else:
        version = "perturbation"
        lc = layer_number - num_relu_layers
    for i_c, c in enumerate(layer_data):
        if c.item() > 1 - th:
            bools += "1,"
        elif c.item() < th:
            bools += "0,"
        else:
            bools += "-1,"
        strings += version + "a_layerCount" + str(lc) + "_neuronCount0_" + str(layer_number) + "_" + str(i_c + 1) + ","
    return bools, strings


def attack(model, X, source_, target_, device, token_signature,\
           model_name, dims, type_="brightness", size_=1.0, iterations=500, alpha=0.01, lambda_0 = 1.01, K_max=500):
    model.eval()
    M = len(X)
    X_pgd = X.clone().detach()
    X_pgd.requires_grad = True
    eps_pgd = define_attack(type_, size_, M, dims, device)
    tt = target_
    ss = source_

    for t in tqdm(range(iterations), desc="Attack"):
        output = model(X_pgd)
        output2 = model(create_attacked(X_pgd,eps_pgd, type_, size_,dims))
        nan_indices = torch.isnan(output2)
        nan_rows = torch.any(nan_indices, dim=1)
        output2[nan_rows] = 0
        output_tmp = output.clone()
        output_tmp[torch.arange(M), ss] = float('-inf')
        max_not_ss, max_labels_ss = output_tmp.max(dim=1)
        diff1 = output[torch.arange(M), ss] - output[torch.arange(M), max_labels_ss]
        max_scores, max_labels = output2.max(dim=1)
        diff2 = output2[torch.arange(M), tt] - output2[torch.arange(M), max_labels]
        if type_ == "max":
            # No perturbation copy exists, so output2 == output and diff2 would
            # push the TARGET to be the argmax, fighting diff1's push for the
            # SOURCE. delta_max's objective is the source margin alone.
            diff = diff1
        else:
            lambdas_ = torch.tensor((torch.absolute(diff1) / (torch.absolute(diff2) + 1e-9)).detach().cpu().numpy()).to(device)
            lambdas_.requires_grad = False
            diff = diff1 + lambda_0 * lambdas_ * diff2
        loss = torch.sum(diff)
        model.zero_grad()
        loss.backward()
        max_vals, max_inds = torch.topk(output, k=2, dim=1)
        max_labels_1 = max_inds[:, 0]
        max_vals, max_inds = torch.topk(output2, k=2, dim=1)
        max_labels_2 = max_inds[:, 0]
        if type_ == "max":
            s_indices = ((max_labels_1 == source_) & (~nan_rows)).nonzero()
        else:
            s_indices = ((max_labels_1 == source_) & (max_labels_2 == target_) & (~nan_rows)).nonzero()
        if t == iterations - 1:
            break
        with torch.no_grad():
            X_pgd += alpha * X_pgd.grad.sign()
            if BENCH_BOX is None:
                X_pgd = torch.clamp(X_pgd, 0, 1)
            else:
                # HAR model only: project each attack iterate into its [-1,1] input domain; clipping to [0,1] would leave the domain, making the found solution useless as a warm start.
                _lo, _hi = BENCH_BOX
                _lo_t = torch.as_tensor(_lo, dtype=X_pgd.dtype, device=X_pgd.device).view(1, *X_pgd.shape[1:])
                _hi_t = torch.as_tensor(_hi, dtype=X_pgd.dtype, device=X_pgd.device).view(1, *X_pgd.shape[1:])
                X_pgd = torch.max(torch.min(X_pgd, _hi_t), _lo_t)
            X_pgd.requires_grad = True
            eps_pgd = update_attack(X_pgd, eps_pgd, alpha, size_, type_, dims)

    k_to_use = min(K_max, len(s_indices))
    best_val = torch.Tensor([0])

    if k_to_use > 0:
        s_indices = s_indices.reshape(-1)
        values, indices = torch.topk(diff1[s_indices], k=k_to_use)
        best_val = values[0]
        indices = s_indices[indices]
        images_to = X_pgd[indices, :]
        eps_to = eps_pgd[indices, :]
        
        layers_outputs = []
        if "3x" in model_name:
            layers_outputs.append(torch.mean(torch.sign((F.relu(model.fc1(images_to.reshape(-1, 784))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(F.relu(model.fc2((F.relu(model.fc1(images_to.reshape(-1, 784))))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign((F.relu(model.fc1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1, 784))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(F.relu(model.fc2((F.relu(model.fc1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1, 784))))))), dim=0))
        elif "4x" in model_name:
            # Clean input through layers
            x = images_to.reshape(-1, 784)
            x1 = F.relu(model.fc1(x))
            x2 = F.relu(model.fc2(x1))
            x3 = F.relu(model.fc3(x2))
            layers_outputs.append(torch.mean(torch.sign(x1), dim=0))
            layers_outputs.append(torch.mean(torch.sign(x2), dim=0))
            layers_outputs.append(torch.mean(torch.sign(x3), dim=0))

            # Attacked input through layers
            attacked_x = create_attacked(images_to, eps_to, type_, size_, dims).reshape(-1, 784)
            ax1 = F.relu(model.fc1(attacked_x))
            ax2 = F.relu(model.fc2(ax1))
            ax3 = F.relu(model.fc3(ax2))
            layers_outputs.append(torch.mean(torch.sign(ax1), dim=0))
            layers_outputs.append(torch.mean(torch.sign(ax2), dim=0))
            layers_outputs.append(torch.mean(torch.sign(ax3), dim=0))
        elif "10x" in model_name:
            # Clean input through layers
            x = images_to.reshape(-1, 784)
            x1 = F.relu(model.fc1(x))
            x2 = F.relu(model.fc2(x1))
            x3 = F.relu(model.fc3(x2))
            x4 = F.relu(model.fc4(x3))
            x5 = F.relu(model.fc5(x4))
            x6 = F.relu(model.fc6(x5))
            x7 = F.relu(model.fc7(x6))
            x8 = F.relu(model.fc8(x7))
            x9 = F.relu(model.fc9(x8))
            for xi in [x1, x2, x3, x4, x5, x6, x7, x8, x9]:
                layers_outputs.append(torch.mean(torch.sign(xi), dim=0))

            # Attacked input through layers
            attacked_x = create_attacked(images_to, eps_to, type_, size_, dims).reshape(-1, 784)
            ax1 = F.relu(model.fc1(attacked_x))
            ax2 = F.relu(model.fc2(ax1))
            ax3 = F.relu(model.fc3(ax2))
            ax4 = F.relu(model.fc4(ax3))
            ax5 = F.relu(model.fc5(ax4))
            ax6 = F.relu(model.fc6(ax5))
            ax7 = F.relu(model.fc7(ax6))
            ax8 = F.relu(model.fc8(ax7))
            ax9 = F.relu(model.fc9(ax8))
            for axi in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
                layers_outputs.append(torch.mean(torch.sign(axi), dim=0))
        elif "5x" in model_name:
            # Clean input through layers
            x = images_to.reshape(-1, 784)
            x1 = F.relu(model.fc1(x))
            x2 = F.relu(model.fc2(x1))
            x3 = F.relu(model.fc3(x2))
            x4 = F.relu(model.fc4(x3))
            layers_outputs.append(torch.mean(torch.sign(x1), dim=0))
            layers_outputs.append(torch.mean(torch.sign(x2), dim=0))
            layers_outputs.append(torch.mean(torch.sign(x3), dim=0))
            layers_outputs.append(torch.mean(torch.sign(x4), dim=0))
            # Attacked input through layers
            attacked_x = create_attacked(images_to, eps_to, type_, size_, dims).reshape(-1, 784)
            ax1 = F.relu(model.fc1(attacked_x))
            ax2 = F.relu(model.fc2(ax1))
            ax3 = F.relu(model.fc3(ax2))
            ax4 = F.relu(model.fc4(ax3))
            layers_outputs.append(torch.mean(torch.sign(ax1), dim=0))
            layers_outputs.append(torch.mean(torch.sign(ax2), dim=0))
            layers_outputs.append(torch.mean(torch.sign(ax3), dim=0))
            layers_outputs.append(torch.mean(torch.sign(ax4), dim=0))
        elif "6x" in model_name:
            # Clean input through layers
            x = images_to.reshape(-1, 784)
            x1 = F.relu(model.fc1(x))
            x2 = F.relu(model.fc2(x1))
            x3 = F.relu(model.fc3(x2))
            x4 = F.relu(model.fc4(x3))
            x5 = F.relu(model.fc5(x4))
            layers_outputs.append(torch.mean(torch.sign(x1), dim=0))
            layers_outputs.append(torch.mean(torch.sign(x2), dim=0))
            layers_outputs.append(torch.mean(torch.sign(x3), dim=0))
            layers_outputs.append(torch.mean(torch.sign(x4), dim=0))
            layers_outputs.append(torch.mean(torch.sign(x5), dim=0))
            # Attacked input through layers
            attacked_x = create_attacked(images_to, eps_to, type_, size_, dims).reshape(-1, 784)
            ax1 = F.relu(model.fc1(attacked_x))
            ax2 = F.relu(model.fc2(ax1))
            ax3 = F.relu(model.fc3(ax2))
            ax4 = F.relu(model.fc4(ax3))
            ax5 = F.relu(model.fc5(ax4))
            layers_outputs.append(torch.mean(torch.sign(ax1), dim=0))
            layers_outputs.append(torch.mean(torch.sign(ax2), dim=0))
            layers_outputs.append(torch.mean(torch.sign(ax3), dim=0))
            layers_outputs.append(torch.mean(torch.sign(ax4), dim=0))
            layers_outputs.append(torch.mean(torch.sign(ax5), dim=0))
        elif "2x" in model_name:
            layers_outputs.append(torch.mean(torch.sign((F.relu(model.fc1(images_to.reshape(-1, 784))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign((F.relu(model.fc1((create_attacked(images_to, eps_to, type_, size_, dims)).reshape(-1, 784))))), dim=0))
        elif model_name == "har":
            # Pretrained tabular FC nets: every fc except the last is a ReLU
            # layer. Clean pass first, then the attacked pass, matching the
            # order the other branches use (num_relu_layers = len//2).
            n_in = model.k * model.w * model.h
            hidden = [getattr(model, f"fc{i}") for i in range(1, 8)
                      if hasattr(model, f"fc{i}")][:-1]
            for src in (images_to,
                        create_attacked(images_to, eps_to, type_, size_, dims)):
                x = src.reshape(-1, n_in)
                for fc in hidden:
                    x = F.relu(fc(x))
                    layers_outputs.append(torch.mean(torch.sign(x), dim=0))
        elif "cnn" in model_name:
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.conv1(images_to.reshape(-1, dims[0], dims[1], dims[2]))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.conv2((F.relu(model.conv1(images_to.reshape(-1, dims[0], dims[1], dims[2])))))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.fc1(model.flatten1(F.relu(model.conv2((F.relu(model.conv1(images_to.reshape(-1, dims[0], dims[1], dims[2]))))))))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign((model.flatten1(F.relu(model.conv1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1, dims[0], dims[1], dims[2])))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.conv2((F.relu(model.conv1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1,  dims[0], dims[1], dims[2])))))))), dim=0))
            layers_outputs.append(torch.mean(torch.sign(model.flatten1(F.relu(model.fc1(model.flatten1(F.relu(model.conv2((F.relu(model.conv1((create_attacked(images_to, eps_to, type_, size_,dims)).reshape(-1,  dims[0], dims[1], dims[2]))))))))))), dim=0))
        if not layers_outputs:
            # No branch above matched this architecture. Writing empty hint
            # files without a fail marker makes hyper_attack_hints parse "" as
            # Float64 and abort the Julia run, so fail loudly instead.
            raise SystemExit(
                f"hyper_attack: no activation-pattern extraction for model_arch "
                f"'{model_name}'; add a branch in attack() before using it with "
                f"--use_hyper_attack true.")
        bools = ""
        strings = ""
        num_relu_layers = len(layers_outputs) // 2
        for l_no,l_data in enumerate(layers_outputs):
            b, s = build_str(l_data, l_no+1, num_relu_layers)
            bools += b
            strings += s
        bools = bools[0:-1]
        strings = strings[0:-1]
        with open("/tmp/strings_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write(strings)
        with open("/tmp/booleans_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write(bools)

    else:
        with open("/tmp/strings_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write("")
        with open("/tmp/booleans_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write("")
        with open("/tmp/fail_" + str(source_) + "_" + str(target_) + "_" + str(token_signature) + ".txt", "w") as file:
            file.write("")
    return best_val


# Per-coordinate input box for the benchmark nets, set by load_dataset and read
# by the sampling/clamping helpers. None for the image datasets, which keep the
# historical [0,1] domain.
BENCH_BOX = None
# Sample count for the benchmark nets' hyper-input pool. Large because rare
# advisories are very rare inside the verified region.
BENCH_N_SAMPLES = 1000000


def load_dataset( dataset, model_path=None ):
    if dataset == "har":
        # No dataset exists for these pretrained nets, and none is needed: the
        # attack seeds itself from random points in the verified input region
        # (see create_hyper_input), exactly as random_images already does for
        # the image datasets. Return empty splits and record the box.
        global BENCH_BOX
        from acas_box import verification_box
        model_dir = os.path.dirname(model_path) if model_path and os.path.isfile(model_path) else model_path
        lo, hi = verification_box(model_dir)
        BENCH_BOX = (lo, hi)
        return [], [], (1, int(lo.size), 1)
    if dataset == "mnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.MNIST(root='./data/',train=False, transform=transform, download=True)
    elif dataset == "fmnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.FashionMNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.FashionMNIST(root='./data/', train=False, transform=transform, download=True)
    elif dataset == "svhn":
        h_dim, w_dim, k_dim = 32, 32, 3
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = dsets.SVHN(root='./data/', transform=transform, download=True)
        val_size = 12000
        train_size = len(dataset) - 12000
        trainset, testset = random_split(dataset, [train_size, val_size])
    elif dataset == "cifar10":
        h_dim, w_dim, k_dim = 32, 32, 3
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.CIFAR10(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.CIFAR10(root='./data/', train=False, transform=transform, download=True)

    return trainset, testset, (k_dim, h_dim, w_dim)

def load_model( model_arch, model_path, dims=(1, 28, 28)):
    # dims = (channels, height, width) from load_dataset; used to build
    # conv nets with the correct input geometry (e.g. CIFAR-10 3x32x32).
    k_dim, h_dim, w_dim = dims
    if model_arch == "2x10":
        model = FNN_2_10()
    elif model_arch == "3x10":
        model = FNN_3_10()
    elif model_arch == "4x10":
        model = FNN_4_10()
    elif model_arch == "3x50":
        model = FNN_3_50()
    elif model_arch == "3x100":
        model = FNN_3_100()
    elif model_arch == "5x10":
        model = FNN_5_10()
    elif model_arch == "6x10":
        model = FNN_6_10()
    elif model_arch == "10x10":
        model = FNN_10_10()
    elif model_arch == "6x100":
        model = FNN_6_100()
    elif model_arch == "9x200":
        model = FNN_9_200()
    elif model_arch == "cnn0":
        model = CNN0()
    elif model_arch == "cnn1":
        model = CNN1(k=k_dim, w=w_dim, h=h_dim)
    elif model_arch == "cnn2":
        model = CNN2(k=k_dim, w=w_dim, h=h_dim)
    elif model_arch == "cnn3":
        model = CNN3()
    elif model_arch == "cnn4":
        model = CNN4(k=k_dim, w=w_dim, h=h_dim)
    elif model_arch == "cnn5":
        model = CNN5(k=k_dim, w=w_dim, h=h_dim)
    elif model_arch == "har":
        model = FNN_HAR(k=k_dim, w=w_dim, h=h_dim)
    else:
        assert False, "New model arch has been detected, please expand models.py and this if condition."

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model


def create_hyper_input(source, trainset, testset, M, dims, perturbation_type=None):

    train_images = [image for image, _ in trainset]
    test_images = [image for image, _ in testset]
    if BENCH_BOX is None:
        random_images = torch.rand(len(trainset)+len(testset), dims[0], dims[1], dims[2])
    else:
        # The benchmark nets have no train/test split, so every sample is drawn
        # from the verified input region; torch.rand's [0,1) would fall outside
        # it entirely.
        import numpy as _np
        lo, hi = BENCH_BOX
        # Far more samples than the image path uses. The verified box is
        # dominated by a few classes, so a source class can otherwise end up
        # with only a handful of points -- or one, which used to crash below.
        random_images = torch.from_numpy(
            _np.random.default_rng(1).uniform(lo, hi, size=(BENCH_N_SAMPLES, lo.size)).astype(_np.float32)
        ).view(-1, dims[0], dims[1], dims[2])
    parts = [random_images]
    if train_images:
        parts.append(torch.stack(train_images))
    if test_images:
        parts.append(torch.stack(test_images))
    all_samples = torch.cat(parts, dim=0)
    model.eval()
    batch_size = 1024
    classification_batches = []
    with torch.no_grad():
        for start in range(0, all_samples.shape[0], batch_size):
            batch = all_samples[start:start+batch_size].to(device)
            classification_batches.append(model(batch).cpu())
    classification = torch.cat(classification_batches, dim=0).to(device)
    _, predicted_labels = torch.max(classification, dim=1)
    indices_of_s = (predicted_labels == source).nonzero().reshape(-1)
    if indices_of_s.numel() == 0:
        raise SystemExit(
            f"hyper_attack: no sampled input is classified as source class {source} "
            f"within the input region; cannot build a warm start. Widen the region "
            f"or raise BENCH_N_SAMPLES.")
    if indices_of_s.numel() < M:
        print(f"  WARNING: only {indices_of_s.numel()} of {len(all_samples)} samples "
              f"are class {source} (wanted {M}); the warm start will be weak.")
    source_samples_classification = classification[indices_of_s]
    values, _ = torch.sort(source_samples_classification, descending=True, dim=1)
    differences = values[:, 0] - values[:, 1]
    _, sorted_indices = differences.sort(descending=True)
    sorted_indices_of_s = indices_of_s[sorted_indices]
    if perturbation_type == "max":
        # delta_max's objective IS this margin, so the best seeds are the
        # highest-margin samples. Spacing them out (below) is right for a
        # targeted attack, where diverse starting points matter, but here it
        # throws away the incumbent we are trying to maximise.
        uniform_indices = sorted_indices_of_s[:M]
    else:
        step_size = max(1, len(sorted_indices_of_s) // M)
        uniform_indices = sorted_indices_of_s[::step_size][:M]
    hyper_input = all_samples[uniform_indices.cpu()].to(device)
    return hyper_input


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VeGHar Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="mnist", help='dataset')
    parser.add_argument('--source', type=float, default=0, help='source')
    parser.add_argument('--target', type=float, default=1, help='target')
    parser.add_argument('--token', type=str, default="04082021", help='token')
    parser.add_argument('--model', type=str, default="3x10", help='3x10, 3x50, cnn1, or cnn2')
    parser.add_argument('--model_path', type=str, default="./models/3x10/model.pth", help='model')
    parser.add_argument('--perturbation', type=str, default="linf", help='perturbation')
    parser.add_argument('--perturbation_size', type=str, default="1", help='perturbation size')
    parser.add_argument('--gpu', type=int, default=0, help='dataset')
    parser.add_argument('--cpu', action='store_true', help='Force CPU-only mode (no GPU)')
    parser.add_argument('--M', type=int, default=1000, help='Number of samples to attack')
    parser.add_argument('--itr', type=int, default=500, help='Number of iterations')
    parser.add_argument('--alpha', type=float, default=0.01, help='Number of iterations')

    args = parser.parse_args()

    source = int(args.source)
    target = int(args.target)
    token_signature = args.token
    model_arch = args.model
    model_path = args.model_path
    perturbation_type = args.perturbation
    perturbation_size_to_parse = args.perturbation_size.split(",")
    perturbation_size = [float(i) for i in perturbation_size_to_parse]
    if perturbation_type == "occ" or perturbation_type == "translation" or perturbation_type == "rotation":
        perturbation_size = [0]+perturbation_size

    dataset = args.dataset
    M = args.M
    iterations = args.itr
    alpha = args.alpha
    if perturbation_type == "rotation":
        M = 10# TBD May 2024
        iterations = 50
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    DEVICE = device
    print("source:", source, "target:", target, "model_arch:", model_arch, "perturbation type:", perturbation_type, \
          "perturbation size:", perturbation_size, "dataset:", dataset)

    trainset, testset, dims = load_dataset(dataset, model_path)
    model = load_model(model_arch, model_path, dims)
    X = create_hyper_input(source, trainset, testset, M, dims, perturbation_type)

    best_val = attack(model, X, source, target, device, token_signature, model_arch, dims, perturbation_type, perturbation_size, iterations)

    print("best_val", best_val.item())
    #print("/tmp/best_val_" + str(source) + "_" + str(target) + "_" + str(token_signature) + ".txt")
    with open("/tmp/best_val_" + str(source) + "_" + str(target) + "_" + str(token_signature) + ".txt", "w") as file:
        file.write(str(best_val.item()))