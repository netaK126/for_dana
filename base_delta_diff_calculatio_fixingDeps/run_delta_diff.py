import subprocess
import sys
import os

def run_julia_script_with_args(julia_script_path: str, arguments: list):
    """
    Executes a Julia script with specified command-line arguments.

    Args:
        julia_script_path: The file path to the Julia script (.jl file).
        arguments: A list of strings, where each string is an argument 
                   to be passed to the Julia script.
    """
    # 1. Define the full command list
    # The command is: ["julia", "script_path", "arg1", "arg2", ...]
    command = ["julia", julia_script_path] + arguments
    
    print(f"--- Running Julia command: {' '.join(command)} ---")
    
    try:
        # 2. Execute the command
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True,
            check=True
        )
        
        # 3. Print the results
        print("\n** Julia Script Execution Successful **")
        print("Exit Code:", result.returncode)
        
        if result.stdout:
            print("\n--- Output (stdout) ---")
            print(result.stdout.strip())
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Julia script failed with exit code {e.returncode}")
        print("Stderr:", e.stderr)
    except FileNotFoundError:
        print("❌ Error: 'julia' command not found. Ensure Julia is installed and in your system's PATH.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    julia_file = "/root/Downloads/base_delta_diff_calculatio_fixingDeps/run.jl"

    my_arguments = ["--model_path_org",
                    "/root/Downloads/code_deprecated_active_just_for_models/models/4x10_2/18/model.p",
                    "--model_path_second",
                    "/root/Downloads/code_deprecated_active_just_for_models/models/4x10_2/18/model.p",
                    "--model_path_vaghar_results",
                    "/root/Downloads/vaghar_as_should_be_originally_no_c_target/results_trying_something/4x10_model.p_linf_0.05_NoCtarget_RegularVaghar_Itr18.txt",
                    "--string_for_name", "sameNetwork"]
    run_julia_script_with_args(julia_file, my_arguments)

    # my_arguments = ["--model_path_org",
    #                 "/root/Downloads/code_deprecated_active_just_for_models/models/4x10_2/18/model.p",
    #                 "--model_path_second",
    #                 "/root/Downloads/code_deprecated_active_just_for_models/models/4x10_2/17/model.p",
    #                 "--model_path_vaghar_results",
    #                 "/root/Downloads/vaghar_as_should_be_originally_no_c_target/results_trying_something/4x10_model.p_linf_0.05_NoCtarget_RegularVaghar_Itr18.txt",
    #                 "--string_for_name", "itr17AndItr18"]
    # run_julia_script_with_args(julia_file, my_arguments)

    # my_arguments = ["--model_path_org",
    #                 "/root/Downloads/code_deprecated_active_just_for_models/models/4x10_2/18/model.p",
    #                 "--model_path_second",
    #                 "/root/Downloads/code_deprecated_active_just_for_models/models/4x10_2/18/model_p.p",
    #                 "--model_path_vaghar_results",
    #                 "/root/Downloads/vaghar_as_should_be_originally_no_c_target/results_trying_something/4x10_model.p_linf_0.05_NoCtarget_RegularVaghar_Itr18.txt",
    #                 "--string_for_name", "smallEpsPerturbation"]
    # run_julia_script_with_args(julia_file, my_arguments)

    