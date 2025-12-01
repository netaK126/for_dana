import argparse
import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import random_split
import os
import sys
print("Working directory:", os.getcwd())
from models_definitions import *
from tqdm import tqdm
import ast
import time
def create_hyper_input(x, trainset, testset, dims):

    inputs = [x]# NETA 0 currently all the same - can be defined differently later
    hyper_input = torch.stack(inputs).to(device)
    train_images = [image for image, _ in trainset]
    train_images = torch.stack(train_images).to(device)
    test_images = [image for image, _ in testset]
    test_images = torch.stack(test_images).to(device)
    random_images = torch.rand(len(trainset)+len(testset), dims[0], dims[1], dims[2]).to(device)
    all_samples = torch.cat((random_images, train_images, test_images,hyper_input), dim=0).to(device)
    M=500
    step_size = len(all_samples)*2 // M
    return all_samples[::step_size][:M]
    # classification = model(all_samples)
    # _, predicted_labels = torch.max(classification, dim=1)
    # indices_of_s = (predicted_labels == source).nonzero().squeeze()
    # source_samples_classification = classification[indices_of_s]
    # values, _ = torch.sort(source_samples_classification, descending=True, dim=1)
    # differences = values[:, 0] - values[:, 1]
    # _, sorted_indices = differences.sort(descending=True)
    # sorted_indices_of_s = indices_of_s[sorted_indices]
    # step_size = len(sorted_indices_of_s) // M
    # uniform_indices = sorted_indices_of_s[::step_size][:M]
    # hyper_input = all_samples[uniform_indices]
    return hyper_input

def load_model(model_arch, model_path):
    if model_arch == "10x10":
        model = FNN_10_10()
    elif model_arch == "3x10":
        model = FNN_3_10()
    else:
        assert ("New model arch has been detected, please expand models.py and this if condition.")

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    return model


def load_dataset( dataset ):
    if dataset == "mnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.MNIST(root='./data/',train=False, transform=transform, download=True)

    return trainset, testset, (k_dim, h_dim, w_dim)

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


def create_attacked(X, eps, perturbation_type,size_,dims):
    if perturbation_type == "brightness":
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
    elif perturbation_type == "linf":
        Xout = torch.clamp(X+eps, 0, 1)
    return Xout

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


def attack(models_list, X, source_, device,model_name, dims,\
            type_="brightness", size_=1.0, iterations=10000, alpha=0.05, lambda_0 = 1.05, K_max=500):
    
    x_for_lower_bound = []
    delta_lower_bounds = []
    for model in models_list:
        models_eval= model.eval()
        M = len(X)
        X_pgd = X.clone().detach()
        X_pgd.requires_grad = True
        eps_pgd = define_attack(type_, size_, M, dims, device)
        ss = source_
        options_for_x = []
        options_for_lower_bound = []
        for t in tqdm(range(iterations), desc="Attack"):
            output = models_eval(X_pgd)
            output2 = models_eval(create_attacked(X_pgd,eps_pgd, type_, size_,dims))

            nan_indices = torch.isnan(output2)
            nan_rows = torch.any(nan_indices, dim=1)
            output2[nan_rows] = 0
            output_tmp = output.clone()
            output_tmp[torch.arange(M), ss] = float('-inf')
            max_not_ss, max_labels_ss = output_tmp.max(dim=1)
            diff1 = output[torch.arange(M), ss] - output[torch.arange(M), max_labels_ss]
            max_scores, max_labels = output2.max(dim=1)
            diff2 = output2[torch.arange(M), ss] - output2[torch.arange(M), max_labels]

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
            s_indices = ((max_labels_1 == ss) & (max_labels_2 != ss) & (~nan_rows)).nonzero()
            
            if s_indices.shape!=torch.Size([0,1]):
                options_for_x.append(X_pgd)
                options_for_lower_bound.append(diff1)

            if t == iterations - 1:
                break
            with torch.no_grad():
                X_pgd += alpha * X_pgd.grad.sign()
                X_pgd = torch.clamp(X_pgd, 0, 1)
                X_pgd.requires_grad = True
                eps_pgd = update_attack(X_pgd, eps_pgd, alpha, size_, type_, dims)

        k_to_use = min(K_max, len(options_for_x))
        # k_to_use = min(K_max, len(s_indices))
        if k_to_use!=0:
            print("found!")
            # values, indices = torch.topk(diff1[s_indices], k=k_to_use)
            # best_val = values[0]
            # indices = s_indices[indices]
            # images_to = X_pgd[indices, :]
            # eps_to = eps_pgd[indices, :]

            best_val = max(options_for_lower_bound)
            max_index = options_for_lower_bound.index(best_val)
            images_to = options_for_x[max_index]

        else:
            images_to = [None]
            best_val = [torch.tensor(0)]
        
        x_for_lower_bound.append(images_to[0])
        delta_lower_bounds.append(best_val[0].item())
    return delta_lower_bounds


def attack_ensemble(models_list, X, source_, device, dims,
                    type_="brightness", size_=1.0, iterations=10000, alpha=0.5, lambda_0=1.01, K_max=500):
    
    # Put all models in evaluation mode
    for model in models_list:
        model.eval()

    M = len(X)
    X_pgd = X.clone().detach().to(device)
    X_pgd.requires_grad = True
    eps_pgd = define_attack(type_, size_, M, dims, device)
    ss = source_
    
    options_for_x = []
    options_for_lower_bound = []
    velocity = torch.zeros_like(X_pgd).to(device)
    for t in tqdm(range(iterations), desc="Ensemble Attack"):
        # Aggregate predictions from all models
        output_sum = torch.zeros_like(models_list[0](X_pgd))
        x_adv = create_attacked(X_pgd, eps_pgd, type_, size_, dims)
        output2_sum = torch.zeros_like(models_list[0](x_adv))

        for model in models_list:
            output_sum += model(X_pgd)
            output2_sum += model(x_adv)
        
        # Average the outputs for the ensemble
        output = output_sum / len(models_list)
        output2 = output2_sum / len(models_list)

        nan_indices = torch.isnan(output2)
        nan_rows = torch.any(nan_indices, dim=1)
        output2[nan_rows] = 0
        
        output_tmp = output.clone()
        output_tmp[torch.arange(M), ss] = float('-inf')
        max_not_ss, max_labels_ss = output_tmp.max(dim=1)
        diff1 = output[torch.arange(M), ss] - output[torch.arange(M), max_labels_ss]
        
        max_scores, max_labels = output2.max(dim=1)
        diff2 = output2[torch.arange(M), ss] - output2[torch.arange(M), max_labels]
        
        lambdas_ = torch.tensor((torch.absolute(diff1) / (torch.absolute(diff2) + 1e-9)).detach().cpu().numpy()).to(device)
        lambdas_.requires_grad = False
        
        # This is the core loss for the ensemble
        diff = diff1 + lambda_0 * lambdas_ * diff2

        ss_tensor = ss
        if isinstance(ss, int):
            ss_tensor = torch.tensor([ss] * M, dtype=torch.long, device=device)
        loss_clean = torch.nn.functional.cross_entropy(output, ss_tensor)  # want correct classification
        loss_attack = torch.nn.functional.cross_entropy(output2, ss_tensor)  # want misclassification
        # loss = loss_clean - lambda_0 * loss_attack
        lambdas_ = torch.abs(diff1) / (torch.abs(diff2) + 1e-9)

        # margin-based loss: you can use e.g. squared hinge or just the negative margin
        loss = torch.sum(diff1 + lambda_0 * lambdas_ * diff2)
        # print(f"iter {t}: loss_clean={loss_clean.item():.4f}, loss_attack={loss_attack.item():.4f}, total={loss.item():.4f}")
        # print(torch.autograd.grad(loss, eps_pgd, retain_graph=True))

        # Zero gradients for all models before backpropagation
        for model in models_list:
            model.zero_grad()
        
        # Backpropagate the combined loss to the input X_pgd
        loss.backward()
        
        # Find successful attacks
        max_vals, max_inds = torch.topk(output, k=2, dim=1)
        max_labels_1 = max_inds[:, 0]
        max_vals, max_inds = torch.topk(output2, k=2, dim=1)
        max_labels_2 = max_inds[:, 0]
        s_indices = ((max_labels_1 == ss) & (max_labels_2 != ss) & (~nan_rows)).nonzero()
        
        if s_indices.shape != torch.Size([0, 1]):
            print(f"found! iter {t}")
            # print(f"iter {t}: diff1={diff1[s_indices].mean().item():.4f}, diff2={diff2.item():.4f}")
            options_for_x.append(X_pgd.clone().detach())
            options_for_lower_bound.append(diff1[s_indices].mean().item()) # Average diff1 for successful indices
        
        if t == iterations - 1:
            break
        
        with torch.no_grad():
            X_pgd_prev = X_pgd.clone()
            X_pgd += alpha * velocity
            X_pgd += alpha * X_pgd.grad.sign()
            X_pgd = torch.clamp(X_pgd, 0, 1)
            X_pgd.requires_grad = True
            eps_pgd_prev = eps_pgd.clone()
            eps_pgd = define_attack(type_, size_, X.size(0), dims, device)
            eps_pgd = eps_pgd.detach().clone().requires_grad_(True)   # force leaf+grad
            # print("leaf?", eps_pgd.is_leaf, "req_grad?", eps_pgd.requires_grad)
            # print(f"eps_pgd_prev==eps_pgd ---> {eps_pgd_prev==eps_pgd}")

    # Return the best attack found across all iterations
    if options_for_x:
        best_val = min(options_for_lower_bound)
        max_index = options_for_lower_bound.index(best_val)
        images_to = options_for_x[max_index]
        return [images_to], [best_val]
    else:
        return [None], [0.0]
    



def attack_ensemble_momentum(models_list, X, source_, device, dims,
                            type_="brightness", size_=1.0, iterations=1000, alpha=0.1, lambda_0=1.01, K_max=500, momentum=0.9):
    
    # Put all models in evaluation mode
    for model in models_list:
        model.eval()

    M = len(X)
    X_pgd = X.clone().detach().to(device)
    # This line ensures the gradient is computed during the forward pass
    X_pgd.requires_grad = True
    
    eps_pgd = define_attack(type_, size_, M, dims, device)
    ss = source_
    
    options_for_x = []
    options_for_lower_bound = []
    
    # Initialize velocity for momentum
    velocity = torch.zeros_like(X_pgd).to(device)

    for t in tqdm(range(iterations), desc="Ensemble Attack with Momentum"):
        # The forward pass must be done with requires_grad=True
        # This creates the computation graph needed for loss.backward()
        output_sum = torch.zeros_like(models_list[0](X_pgd))
        output2_sum = torch.zeros_like(models_list[0](create_attacked(X_pgd, eps_pgd, type_, size_, dims)))

        for model in models_list:
            output_sum += model(X_pgd)
            output2_sum += model(create_attacked(X_pgd, eps_pgd, type_, size_, dims))
        
        output = output_sum / len(models_list)
        output2 = output2_sum / len(models_list)

        nan_indices = torch.isnan(output2)
        nan_rows = torch.any(nan_indices, dim=1)
        output2[nan_rows] = 0
        
        output_tmp = output.clone()
        output_tmp[torch.arange(M), ss] = float('-inf')
        max_not_ss, max_labels_ss = output_tmp.max(dim=1)
        diff1 = output[torch.arange(M), ss] - output[torch.arange(M), max_labels_ss]
        
        max_scores, max_labels = output2.max(dim=1)
        diff2 = output2[torch.arange(M), ss] - output2[torch.arange(M), max_labels]
        
        lambdas_ = torch.tensor((torch.absolute(diff1) / (torch.absolute(diff2) + 1e-9)).detach().cpu().numpy()).to(device)
        lambdas_.requires_grad = False
        
        diff = diff1 + lambda_0 * lambdas_ * diff2
        loss = torch.sum(diff)
        print("loss = ")
        print(loss)
        
        # Zero gradients of the input tensor from previous steps
        if X_pgd.grad is not None:
            X_pgd.grad.zero_()
        
        # Backpropagate the combined loss to the input X_pgd
        loss.backward()
        
        # Update velocity with momentum
        velocity = momentum * velocity + X_pgd.grad.sign()
        # Find successful attacks
        max_vals, max_inds = torch.topk(output, k=2, dim=1)
        max_labels_1 = max_inds[:, 0]
        max_vals, max_inds = torch.topk(output2, k=2, dim=1)
        max_labels_2 = max_inds[:, 0]
        s_indices = ((max_labels_1 == ss) & (max_labels_2 != ss) & (~nan_rows)).nonzero()

        if s_indices.shape != torch.Size([0, 1]):
            print("found!")
            options_for_x.append(X_pgd.clone().detach())
            options_for_lower_bound.append(diff1[s_indices].mean().item()) # Average diff1 for successful indices
        

        if t == iterations - 1:
            break
        
        with torch.no_grad():
            # Update the input using the velocity
            X_pgd += alpha * velocity
            X_pgd = torch.clamp(X_pgd, 0, 1)
            
            # This line is crucial to re-enable gradient tracking for the next iteration
            X_pgd.requires_grad = True
            
            eps_pgd = update_attack(X_pgd, eps_pgd, alpha, size_, type_, dims)

    # Return the best attack found across all iterations
    if options_for_x:
        best_val = min(options_for_lower_bound)
        max_index = options_for_lower_bound.index(best_val)
        images_to = options_for_x[max_index]
        return [images_to], [best_val]
    else:
        return [None], [0.0]


import torch
from tqdm import tqdm
import random

def attack_ensemble_adaptive(models_list, X, source_, device, dims,
                            type_="brightness", size_=1.0, iterations=5000, alpha=0.05, lambda_0=1.01, momentum=0.9, n_restarts=1):
    
    best_images = None
    best_lower_bound = float('inf')

    for restart in range(n_restarts):
        print(f"Starting restart {restart + 1}/{n_restarts}")
        
        X_pgd = X.clone().detach().to(device)
        random_noise = (torch.rand_like(X_pgd) * 2 - 1) * 0.001
        X_pgd += random_noise
        X_pgd = torch.clamp(X_pgd, 0, 1)
        X_pgd.requires_grad = True
        
        eps_pgd = define_attack(type_, size_, len(X), dims, device)
        ss = source_
        
        options_for_x = []
        options_for_lower_bound = []
        velocity = torch.zeros_like(X_pgd).to(device)

        for t in tqdm(range(iterations), desc="Ensemble Attack with Adaptive Step"):
            output_sum = torch.zeros_like(models_list[0](X_pgd))
            output2_sum = torch.zeros_like(models_list[0](create_attacked(X_pgd, eps_pgd, type_, size_, dims)))
            for model in models_list:
                output_sum += model(X_pgd)
                output2_sum += model(create_attacked(X_pgd, eps_pgd, type_, size_, dims))
            
            output = output_sum / len(models_list)
            output2 = output2_sum / len(models_list)

            nan_indices = torch.isnan(output2)
            nan_rows = torch.any(nan_indices, dim=1)
            output2[nan_rows] = 0
            
            output_tmp = output.clone()
            output_tmp[torch.arange(len(X)), ss] = float('-inf')
            max_not_ss, max_labels_ss = output_tmp.max(dim=1)
            diff1 = output[torch.arange(len(X)), ss] - output[torch.arange(len(X)), max_labels_ss]
            
            max_scores, max_labels = output2.max(dim=1)
            diff2 = output2[torch.arange(len(X)), ss] - output2[torch.arange(len(X)), max_labels]
            
            lambdas_ = torch.tensor((torch.absolute(diff1) / (torch.absolute(diff2) + 1e-9)).detach().cpu().numpy()).to(device)
            lambdas_.requires_grad = False
            
            diff = diff1 + lambda_0 * lambdas_ * diff2
            loss = torch.sum(diff)
            # print("loss = ")
            # print(loss)
            
            if X_pgd.grad is not None:
                X_pgd.grad.zero_()
            
            loss.backward()

            # Change: Use the raw gradient for the update
            velocity = momentum * velocity + X_pgd.grad

            # Update the input using the velocity. Now the update is scaled by the magnitude of the gradient
            X_pgd = X_pgd + alpha * velocity 
            
            # Find successful attacks
            max_vals, max_inds = torch.topk(output, k=2, dim=1)
            max_labels_1 = max_inds[:, 0]
            max_vals, max_inds = torch.topk(output2, k=2, dim=1)
            max_labels_2 = max_inds[:, 0]
            s_indices = ((max_labels_1 == ss) & (max_labels_2 != ss) & (~nan_rows)).nonzero()
            # print(diff2[0].mean().item())
            # if diff2[0].mean().item()<0:
            #     print("diff1 = ")
            #     print(diff1[0].mean().item())
            #     print("diff2 = ")
            #     print(diff2[0].mean().item())
            #     print("-------------------------------------------")
            if s_indices.shape != torch.Size([0, 1]):
                print("FOUND!")
                options_for_x.append(X_pgd.clone().detach())
                options_for_lower_bound.append(diff1[s_indices].mean().item())
            
            if t == iterations - 1:
                break
            
            with torch.no_grad():
                X_pgd = torch.clamp(X_pgd, 0, 1)
                X_pgd.requires_grad = True
                eps_pgd = update_attack(X_pgd, eps_pgd, alpha, size_, type_, dims)

        if options_for_lower_bound:
            current_best_val = min(options_for_lower_bound)
            if current_best_val < best_lower_bound:
                best_lower_bound = current_best_val
                max_index = options_for_lower_bound.index(current_best_val)
                best_images = options_for_x[max_index]

    if best_images is not None:
        return [best_images], [best_lower_bound]
    else:
        return [None], [0.0]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VeGHar Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="mnist", help='dataset')
    parser.add_argument('--source', type=float, default=0, help='source')
    parser.add_argument('--model', type=str, default="3x10", help='10x10, 3x10, 3x50, cnn1, or cnn2')
    parser.add_argument('--model_path_list', type=str, 
                        default="['/root/Downloads/lucid/models_mnist_ensamble/mnist2.pth', '/root/Downloads/lucid/models_mnist_ensamble/mnist1.pth']",
                          help='models list')
    parser.add_argument('--perturbation', type=str, default="linf", help='perturbation')
    parser.add_argument('--perturbation_size', type=str, default="0.02", help='perturbation size')
    parser.add_argument('--gpu', type=int, default=0, help='dataset')
    parser.add_argument('--M', type=int, default=1000, help='Number of samples to attack')
    parser.add_argument('--itr', type=int, default=500, help='Number of iterations')
    parser.add_argument('--alpha', type=float, default=0.01, help='Number of iterations')
    parser.add_argument('--input_x', type=str, default="/root/Downloads/vaghar_as_should_be_originally_no_c_target/tensor_input.npy", help='the input from finding the upper bound')

    args = parser.parse_args()

    source = int(args.source)
    model_arch = args.model
    perturbation_type = args.perturbation
    perturbation_size_to_parse = args.perturbation_size.split(",")
    perturbation_size = [float(i) for i in perturbation_size_to_parse]
    model_path_list = ast.literal_eval(args.model_path_list)
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
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset, testset, dims = load_dataset(dataset)
    models_list = [load_model(model_arch, model_path) for model_path in model_path_list]
    # X = create_hyper_input(ast.literal_eval(args.input_x), models_list)
    x_vals = np.load(args.input_x)
    x_tensor = torch.from_numpy(x_vals).float()
    # X = create_hyper_input(x_tensor, models_list) # check if script ok - to be deleted later
    # X = create_hyper_input(trainset[5][0], models_list)
    X = create_hyper_input(x_tensor, trainset, testset, dims)
    start_time = time.time()
    _, lower_bounds_list = attack_ensemble(models_list, X, source, device, dims, perturbation_type, perturbation_size)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    print("time = ")
    print(elapsed_time)
    print("lower bounds = ")
    print(lower_bounds_list)