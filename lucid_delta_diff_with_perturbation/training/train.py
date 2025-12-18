import numpy as np
import torch
import time
import pickle
from data import DataLoader

# includes function for training and evaluating networks.

def flatten_params(dict):
    """
    turns a state dict into a parameter vector.
    """
    if dict is None:
        return 0
    f = ()
    for _, t in dict.items():
        f = f + (t.view(-1),)
    return torch.cat(f)

def vec2dict(w, dict):
    """
    turns a parameter vector into a state dict
    inputs:
    w: parameter vector.
    dict: sample state dict.
    output:
    d: state dict.
    """
    d = {}
    i = 0
    for x, t in dict.items():
        l = len(t.view(-1))
        d[x] = w[i:i + l].view(t.size())
        i += l
    return d

def calculate_accuracy(model, dataloader, device=torch.device('cpu'), th0 = 0, th1 = 0):
    """
    calculate accuracy of model on an evaluation set.
    inputs:
    model: the neural network.
    dataloader: loader for the evaluation set.
    output:
    model_accuracy: accuracy of the model on the evaluation set.
    """
    model.eval()
    total_correct = 0
    total_images = 0
    total_correct_above_th = 0
    with torch.no_grad():
        while True:
            data, labels, last = dataloader.get_batch()
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == -1] = 0
            outputs = model(data)
            #predicted = outputs.sign().int()
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            a = ((outputs[:, 0] - outputs[:, 1]) > th0)
            c = ((outputs[:, 1] - outputs[:, 0]) > th1)
            b = (predicted == labels)
            #print(outputs[:, 0] - outputs[:, 1])
            total_correct_above_th += torch.logical_and(torch.logical_or(a,c), b).sum().item()

            if last:
                break
    #print("total_correct",total_correct,"total_images",total_images)
    model_accuracy = total_correct / total_images * 100
    above_thr_acc = total_correct_above_th/total_images * 100
    #print("Acc above threshold",total_correct_above_th/total_images * 100)
    return model_accuracy, above_thr_acc

def calculate_accuracy_with_log(model, dataloader, device=torch.device('cpu'), th0 = 0, th1 = 0):
    """
    calculate accuracy of model on an evaluation set.
    inputs:
    model: the neural network.
    dataloader: loader for the evaluation set.
    output:
    model_accuracy: accuracy of the model on the evaluation set.
    """
    model.eval()
    total_correct = 0
    total_images = 0
    total_correct_above_th = 0
    results0 = []
    results1 = []
    with torch.no_grad():
        while True:
            data, labels, last = dataloader.get_batch()
            data = data.to(device)
            labels = labels.to(device)
            labels[labels == -1] = 0
            outputs = model(data)
            #predicted = outputs.sign().int()
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            a = ((outputs[:, 0] - outputs[:, 1]) > th0)
            c = ((outputs[:, 1] - outputs[:, 0]) > th1)
            b = (predicted == labels)
            #print(outputs[:, 0] - outputs[:, 1])
            total_correct_above_th += torch.logical_and(torch.logical_or(a,c), b).sum().item()
            diff0 = outputs[:, 0] - outputs[:, 1]
            diff1 = outputs[:, 1] - outputs[:, 0]

            results0.append(diff0[torch.logical_and(labels==0, b)].numpy())
            results1.append(diff1[torch.logical_and(labels==1, b)].numpy())

            if last:
                break
    pickle.dump([results0,results1], open("save.p", "wb"))
    #print("total_correct",total_correct,"total_images",total_images)
    model_accuracy = total_correct / total_images * 100
    above_thr_acc = total_correct_above_th/total_images * 100
    #print("Acc above threshold",total_correct_above_th/total_images * 100)
    return model_accuracy, above_thr_acc

def train_model(model, opt, criterion, train_loader, val_loader=None, reg=0, num_epochs=10, device=torch.device('cpu'), removed=None, added=None, dp_flag = False, dp=0, clip=0, aug=None, prints=False):
    """
    train a neural network.
    inputs:
    model: the untrained network.
    opt: the optimizer object.
    criterion: the loss function.
    train_loader: data loader for the training set.
    val_loader: data loader for the validation set.
    reg: L1 regularization coefficient.
    num_epochs: number of training epochs.
    removed: the sample to remove from the training set.
    added: the sample to add to the training set.
    dp: scale of noise to add to the gradients.
    clip: clipping scale for the gradients.
    aug: data augmentation model.
    prints: boolean value. If true, prints statistics of each epoch.
    outputs:
    run_loss: list of the losses after each epoch.
    run_train_acc: list of accuracy on the training set after each epoch.
    run_val_acc: list of accuracy on the validation set after each epoch.
    W: matrix of the network parameters after each epoch.
    """
    run_loss = []
    run_train_acc = []
    run_val_acc = []
    val_acc = 0
    epoch = 0
    t = time.time()
    train_loss = []
    torch.set_num_threads(1)
    W = flatten_params(model.state_dict()).repeat(num_epochs + 1, 1)
    model.train()
    while epoch < num_epochs:
        data, labels, last = train_loader.get_batch()
        if removed is not None:
            d = removed[0]
            flags = ((data - d.unsqueeze(0)).view(data.size(0), -1).abs().sum(dim=1) == 0)
            data[flags] = added[0]
            labels[flags] = added[1]
        labels[labels == -1] = 0
        #labels = torch.nn.functional.one_hot(labels.to(torch.int64), 2).float()
        data = data.to(device)
        labels = labels.to(device)
        if added is not None and removed is None:
            data[0] = added[0]
            labels[0] = added[1]
        if aug is not None:
            data = aug.forward(data)
        output = model.forward(data)
        loss = criterion(output, labels.long())
        for z in model.parameters():
            loss += reg * z.abs().view(-1).sum()
        opt.zero_grad()
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=clip, norm_type=2.0)
        opt.step()
        train_loss.append(loss.data.item())
        #if dp > 0:
        #    d = {}
        #    for i, w in model.state_dict().items():
        #        d[i] = w + (clip * dp / len(labels)) * torch.randn(w.size()).to(device)
        #    model.load_state_dict(d)
        if last:
            loss = np.mean(train_loss)
            run_loss += [loss]
            train_acc = calculate_accuracy(model, train_loader, device)
            run_train_acc += [train_acc]
            if val_loader is not None:
                val_acc = calculate_accuracy(model, val_loader, device)
                run_val_acc += [val_acc]
            t = time.time() - t
            #print(epoch, loss, train_acc, val_acc, t)
            if dp_flag == True:
                print('Epoch: {:2d} | Loss: {:.4f} | Training Accuracy: {:.2f}% | Validation Accuracy: {:.2f}% | Epoch Time: {:.2f} secs'.format(epoch, loss, train_acc, val_acc, t))
                epsilon, best_alpha = opt.privacy_engine.get_privacy_spent(1e-5)
                print(f"(ε = {epsilon:.2f}, δ = {delta})")

            epoch += 1
            W[epoch] = flatten_params(model.state_dict())
            t = time.time()
            train_loss = []
            model.train()
    return run_loss, run_train_acc, run_val_acc, W

if __name__ == '__main__':
    # calculates the accuracy of our trained networks.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    names = ['adult', 'bank', 'credit']
    arch = ['', 'wide_', 'deep_']
    batch_size = 256
    for name in names:
        train_set = torch.load('../datasets/' + name + '/train.pth')
        test_set = torch.load('../datasets/' + name + '/test.pth')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        for ar in arch:
            model, _ = torch.load('../models/' + ar + name + '.pth', map_location=device)
            acc = calculate_accuracy(model, dataloader=test_loader, device=device)
            print(ar + name, acc)
