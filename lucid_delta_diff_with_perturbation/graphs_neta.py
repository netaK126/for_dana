import matplotlib.pyplot as plt
import pandas as pd
import os

def get_results_from_file(file_path):
    with open(file_path) as f:
        first_line = f.readline().strip('\n')
        first_line_items = first_line.split(", ")
        l = round(float(first_line_items[1].split("= ")[-1]),5)
        r = round(float(first_line_items[2].split("= ")[-1]),5)
        t = round(float(first_line_items[3].split("= ")[-1]),5)
        return l,r,t
    return -1,-1,-1

def generate_bounds_graphs(file_path_list, title):
    bounds_dict = {}
    """
    {<separation_idx>:[<lower_bound>,<upper_bound>,<time>]}
    """
    for file_path in file_path_list:
        l,r,t = get_results_from_file(file_path)
        separation_layer = (file_path.split("_SplitingIndex")[-1]).split("_")[0]
        bounds_dict[separation_layer]=[l,r,t]
    bar_names = list(bounds_dict.keys())
    l_values = [v[0] for v in bounds_dict.values()]
    r_values = [v[1]-v[0] for v in bounds_dict.values()]
    print(bounds_dict)
    # Create the bar plot
    plt.figure(figsize=(12, 10))
    plt.bar(bar_names, l_values, color='skyblue', label='Lower Bound (l)')
    plt.bar(bar_names, r_values, bottom=l_values, color='lightcoral', label='Range (r - l)')
    plt.xlabel('Separation Layer')
    plt.ylabel('Bounds')
    plt.title('Bounds of delta_diff')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(title+".png")
    plt.show(block=True)
    
def generate_times_graphs(file_path_list, title):
    bounds_dict = {}
    """
    {<separation_idx>:<time>}
    """
    for file_path in file_path_list:
        print(file_path)
        l,r,t = get_results_from_file(file_path)
        separation_layer = (file_path.split("_SplitingIndex")[-1]).split("_")[0]
        bounds_dict[separation_layer]=t
    bar_names = list(bounds_dict.keys())
    t_values = [v for v in bounds_dict.values()]
    print(bounds_dict)
    # Create the bar plot
    plt.figure(figsize=(12, 10))
    plt.bar(bar_names, t_values, color='skyblue', label='times (l)')
    plt.xlabel('Separation Layer')
    plt.ylabel('times')
    plt.title('runningtime of delta_diff')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(title+".png")
    plt.show(block=True)

if __name__=="__main__":

    RESULTS_PATH = "/root/Downloads/lucid/results_neta_mnist_i"
    file_path_list_withLucid = sorted([os.path.join(RESULTS_PATH,f) for f in os.listdir(RESULTS_PATH) if "SplitingIndex" in f and "VSmnist5" in f and "ing1.txt" in f])
    # file_path_list_noLucid = sorted([os.path.join(RESULTS_PATH,f) for f in os.listdir(RESULTS_PATH) if "separationIndex" in f and "noLucid" in f and "Optimizing2" in f])
    generate_bounds_graphs(file_path_list_withLucid,"mnist1VSmnist5 - with Lucid c_tag=1 - bounds")
    generate_times_graphs(file_path_list_withLucid,"mnist1VSmnist5 - with Lucid c_tag=1 - times")
    exit()
    # generate_bounds_graphs(file_path_list_noLucid,"no Lucid c_tag=2 - bounds")
    # generate_times_graphs(file_path_list_noLucid,"no Lucid c_tag=2 - times")
    file_path_list_withLucid = sorted([os.path.join(RESULTS_PATH,f) for f in os.listdir(RESULTS_PATH) if "separationIndex" in f and "withLucid" in f and "Optimizing1" in f])
    file_path_list_noLucid = sorted([os.path.join(RESULTS_PATH,f) for f in os.listdir(RESULTS_PATH) if "separationIndex" in f and "noLucid" in f and "Optimizing1" in f])
    generate_bounds_graphs(file_path_list_withLucid,"with Lucid c_tag=1 - bounds")
    generate_times_graphs(file_path_list_withLucid,"with Lucid c_tag=1 - times")
    generate_bounds_graphs(file_path_list_noLucid,"no Lucid c_tag=1 - bounds")
    generate_times_graphs(file_path_list_noLucid,"no Lucid c_tag=1 - times")



