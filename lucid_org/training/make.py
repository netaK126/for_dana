import numpy as np
import torch
import time
import pickle
import os
from models import deep, binary_loss
from train import train_model, calculate_accuracy
from data import DataLoader
import argparse

# includes a function to train and save new network.

def make_model(i=None, j=0, seed=666, path='../models/adult', data_path='../datasets/adult/',
               hidden_dim=5, hidden_num=2, batch_size=1024, lr=1e-1, reg=1e-5, wd=0, num_epochs=50, dev='cpu', dp=0, clip=0):

    device = torch.device(dev)
    print('device:', device)
    train_set = torch.load(data_path + 'train.pth')
    test_set = torch.load(data_path + 'test.pth')

    if i is not None:
        removed = (train_set[0][i], train_set[1][i])
    else:
        removed = (test_set[0][j], test_set[1][j])
        
    added = (test_set[0][j], test_set[1][j])
    np.random.seed(seed)
    torch.manual_seed(seed)
    input_dim = train_set[0].size(1)
    criterion =  torch.nn.CrossEntropyLoss()
    model = deep(input_dim, hidden_dim, hidden_num).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    _, _, _, W = train_model(model, opt, criterion, train_loader, None, reg, num_epochs, device, removed, added, lr * dp, clip)

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    acc, acc_above = calculate_accuracy(model, test_loader, device, 0,0)
    print('Test Accuracy: {:.10f}%'.format(acc))
    print('Test Accuracy Above: {:.10f}%'.format(acc_above))
    acc, acc_above = calculate_accuracy(model, train_loader, device)
    print('Train Accuracy: {:.10f}%'.format(acc))

    if i is not None:
        path = path + str(i)
    path_pth = path + '.pth'
    torch.save(model.state_dict(), path_pth)
    print(path_pth)
    
    a = []
    for i in model.parameters():
        a.append(np.transpose(i.cpu().detach().numpy()))
    pickle.dump(a, open(path + ".p", "wb"))
    
    return W, path

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Privacy Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start', type=int, default=0, help='Start index of networks to train')
    parser.add_argument('--end', type=int, default=-1, help='End index of networks to train')
    parser.add_argument('--neurons', type=int, default=50, help='Number of neurons in each layer')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dataset', type=str, default="twitter", help='Dataset')
    parser.add_argument('--device', type=str, default="cpu", help='Device')

    args = parser.parse_args()
    start = args.start
    end = args.end
    neurons = args.neurons
    layers = args.layers
    dataset = args.dataset
    device = args.device

    data_path = '../datasets/' + dataset + '/'
    train_set = torch.load(data_path + 'train.pth')
    N = len(train_set[1])
    path = '../models/' + dataset

    _,path_ = make_model(path=path, data_path=data_path, hidden_dim=neurons, hidden_num=layers, dev=device)
    if end == -1:
      end = N  
      
    for n in range(start,end):
        if os.path.exists(path + str(n) + '.pth'):
            print("File exists.")
            continue
        _,path_ = make_model(n, path=path, data_path=data_path, hidden_dim=neurons, hidden_num=layers, dev=device)





