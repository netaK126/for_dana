import numpy as np
import torch
import time
import argparse
import subprocess

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Privacy Example', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--workers', type=int, default=64, help='Start index of networks to train')
    parser.add_argument('--neurons', type=int, default=50, help='Number of neurons in each layer')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--dataset', type=str, default="twitter", help='Dataset')
    parser.add_argument('--devices', type=str, default="cpu", help='Device: for example cpu or cuda:0 or cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7')

    args = parser.parse_args()
    workers = args.workers
    neurons = args.neurons
    layers = args.layers
    dataset = args.dataset
    devices_list = args.devices.split(",")

    data_path = '../datasets/' + dataset + '/'
    train_set = torch.load(data_path + 'train.pth')
    N = len(train_set[1])

    networks_per_device = int(np.ceil(N/len(devices_list)))
    networks_per_device_per_worker = int(np.ceil(networks_per_device/workers))
    processes = []
    start_time = time.time()
    for i in range(len(devices_list)):
        print("networks to analyze in ", devices_list[i], "are from ", i*networks_per_device, "to", (i+1) * networks_per_device)
        for j in range(workers):
            print("networks to analyze by worker ", j, "in device", devices_list[i], "are from ",
                  i * networks_per_device + j * networks_per_device_per_worker, "to",
                  i * networks_per_device + (j+1) * networks_per_device_per_worker)

            cmd = [
                "python3", "./make.py",
                "--start", str(i * networks_per_device + j * networks_per_device_per_worker),
                "--end", str(i * networks_per_device + (j + 1) * networks_per_device_per_worker),
                "--neurons", str(neurons),
                "--layers", str(layers),
                "--dataset", str(dataset),
                "--device", str(devices_list[i])
            ]

            print(cmd)

            process = subprocess.Popen(cmd)
            processes.append(process)


    for process in processes:
        process.wait()
    print("Total time is ", time.time()-start_time)
    print("All processes have finished!")

