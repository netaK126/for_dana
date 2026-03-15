import os
import re
import subprocess

def run_julia_scripts(directory_path, composed_interval=False):
    # Regex pattern to capture SOMENUM and N
    # _linf_(.*?)_ctag matches the perturbation size
    # cTag(\d+)\.txt$ matches the cTag number at the end
    pattern = re.compile(r"_linf_(.*?)_ctag.*cTag(\d+)\.txt$")

    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} not found.")
        return

    for filename in os.listdir(directory_path):
        match = pattern.search(filename)
        
        if match:
            # Extract captured groups
            perturbation_size = match.group(1)
            c_tag_n = match.group(2)
            
            # Construct the command
            command = [
                "julia", 
                "run.jl", 
                "--ctag", c_tag_n,
                "--perturbation_size", perturbation_size,
                "--model_name", "3x10",
                "--model_path", "/root/Downloads/vaghar_org/models_as_in_vaghar/3x10_mnist_sgd/18/model.p",
                "--model_path2", "/root/Downloads/vaghar_org/models_as_in_vaghar/3x10_mnist_sgd/19/model.p",
                "--output_dir", "/root/Downloads/vaghar_org/results_for_transfer_for_vaghgar_pgd_n1IsItr18/",
                "--mode", "transfer",
                "--vaghar_results", os.path.join(directory_path, filename),
                "--c_tag_mode", "false",
                "--use_hyper_attack", "true",
                "--use_perturbed_intervals", "true",
                "--activate_vaghgar_deps", "true",
                "--use_intervals", "true",
                "--n1_p_mode", "false",
                "--n2_fewer_binars_encoding", "true",
                "--composed_interval", str(composed_interval).lower()
            ]
            
            print(f"Executing: {' '.join(command)} for file: {filename}")
            
            try:
                # Run the command and wait for it to finish
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running julia for {filename}: {e}")
            except FileNotFoundError:
                print("Error: 'julia' command not found. Ensure Julia is in your PATH.")
                break

# Usage
path_vaghar_results = "/root/Downloads/vaghar_org/replicating_vaghgar_pgd_results_itr18" 
# run_julia_scripts(path_vaghar_results, composed_interval=False)  # Set compose_interval to False to activate I^C constraints
run_julia_scripts(path_vaghar_results, composed_interval=True)   # Set compose_interval to True to activate I^C constraints