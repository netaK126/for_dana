import argparse
import time
import os
import torch
import numpy as np
import pickle
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import random_split
use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"
import warnings
warnings.filterwarnings("ignore")
procs = []

def flatten_params(dict):
    if dict is None:
        return 0
    f = ()
    for _, t in dict.items():
        f = f + (t.view(-1),)
    return torch.cat(f)

def vec2dict(w, dict):
    d = {}
    i = 0
    for x, t in dict.items():
        l = len(t.view(-1))
        d[x] = w[i:i + l].view(t.size())
        i += l
    return d

def load_hyper_dataset(models_path, dataset, device='cpu'):
    print('device:', device)
    train_set = torch.load('./datasets/' + dataset + '/train.pth')
    N = len(train_set[1])
    path = models_path + "model_modified2"#dataset
    model = torch.load(path + '.pth', map_location=torch.device(device))
    if os.path.exists(path+'W_all.pth'):
        W = torch.load(path+'W_all.pth')
    else:
        W = (flatten_params(model).unsqueeze(0),)
        for i in range(N):
            print('\r{:5.2f}%'.format(100 * i / N), end='')
            model = torch.load(path + str(i) + '.pth', map_location=torch.device(device))
            W = W + (flatten_params(model).unsqueeze(0),)
        print('\rdone!')
        W = torch.cat(W, dim=0)
        torch.save(W, path+'W_all.pth')
    w_rel = W + 0.0
    w_indexes = []
    results = []
    return W, w_rel, model, w_indexes, results

def load_dataset_images(models_path, dataset, model_hyper_path, device='cpu'):
    print('device:', device)
    mnist_num = 1
    if dataset == "mnist":
        h_dim, w_dim, k_dim = 28, 28, 1
        transform = transforms.Compose([transforms.ToTensor()])
        trainset = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
        testset = dsets.MNIST(root='./data/',train=False, transform=transform, download=True)
    N = len(trainset)
    path = models_path #+ dataset 
    model = torch.load(path+'model_itr18.pth', map_location=torch.device(device))
    if os.path.exists(path+'W_all.pth'):
        W = torch.load(path+'W_all.pth')
    else:
        W = (flatten_params(model).unsqueeze(0),)
        for i in [mnist_num]:#range(N):
            print('\r{:5.2f}%'.format(100 * i / N), end='')
            model = torch.load(path + 'model_itr18.pth', map_location=torch.device(device))
            W = W + (flatten_params(model).unsqueeze(0),)
        print('\rdone!')
        W = torch.cat(W, dim=0)
        torch.save(W, model_hyper_path +dataset+'W_all.pth')
    w_rel = W + 0.0
    w_indexes = []
    results = []
    return W, w_rel, model, w_indexes, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Privacy Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default="mnist", help='Dataset: twitter, crypto, adult, or credit.')
    parser.add_argument('--models_path', type=str, default="/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/", help='Path of the models.')
    parser.add_argument('--model_hyper_path', type=str, default="/root/Downloads/lucid_delta_diff_with_perturbation/models_4x10_mnist/", help='Path of the models.')
    parser.add_argument('--models_indexes', type=str, default="all",
                        help='Indexes of models to include in the hypernetowrk. all to include all of them and 1,2,3 to include only first three networks')

    args = parser.parse_args()
    dataset = args.dataset
    models_path = args.models_path
    models_indexes = args.models_indexes
    model_hyper_path = args.model_hyper_path
    if "mnist" in dataset:
        W, w_rel, model, w_indexes, results = load_dataset_images(models_path, dataset, model_hyper_path)
    else:
        W, w_rel, model, w_indexes, results = load_hyper_dataset(models_path, dataset)
    print(W.shape)

    if models_indexes == "all":
        cl = np.ones(len(w_rel.numpy()))
    else:
        models_indexes = models_indexes.split(",")
        models_indexes = [int(i) for i in models_indexes]
        cl = np.zeros(len(w_rel.numpy()))
        cl[models_indexes] = 1

    network_ind_rel = np.arange(W.shape[0])
    wmin, _ = w_rel[cl == 1, :].min(dim=0)
    wmax, _ = w_rel[cl == 1, :].max(dim=0)
    dmin = vec2dict(wmin, model)
    dmax = vec2dict(wmax, model)

    a = []
    for (i, j) in dmin.items():
        a.append(np.transpose(j.cpu().detach().numpy()))
    for i in a:
        pickle.dump(a, open(model_hyper_path + '/hypernetwork_min_box.p', "wb"))
    a = []
    for (i, j) in dmax.items():
        a.append(np.transpose(j.cpu().detach().numpy()))
    for i in a:
        pickle.dump(a, open(model_hyper_path + '/hypernetwork_max_box.p', "wb"))






















