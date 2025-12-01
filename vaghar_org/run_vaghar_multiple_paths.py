import subprocess
import os
import sys
import time

# --- Configuration ---
# Set the directory you want to iterate over.
# '.' means the current directory where this script is running.
TARGET_DIR = r"/root/Downloads/code_deprecated_active_just_for_models/models/4x10_GeminiDistilation/"
JULIA_SCRIPT_PATH = r"/root/Downloads/vaghar_org/run.jl"

def run_julia_script_in_loop():
    """
    Scans the TARGET_DIR for files, and iterates through their full paths, 
    calling the julia_script.jl with the full path as an argument in each iteration.
    """
    
    # 1. Generate the list of full file paths to process
    files_to_process = []
    
    # Check if the Julia script exists
    if not os.path.exists(JULIA_SCRIPT_PATH):
        print(f"Error: Julia script not found at path: {JULIA_SCRIPT_PATH}", file=sys.stderr)
        print("Please ensure both 'driver.py' and 'julia_script.jl' are in the same directory.")
        return

    print(f"--- Starting Python Driver Loop (Scanning directory: {TARGET_DIR}) ---")
    
    try:
        # Iterate over contents of the target directory
        for item in os.listdir(TARGET_DIR):
            if ".pth" in item:
                continue
            # Construct the full path
            full_path = os.path.abspath(os.path.join(TARGET_DIR, item))
            
            # Filter: only include actual files (skip subdirectories, etc.)
            if os.path.isfile(full_path):
                files_to_process.append(full_path)

    except FileNotFoundError:
        print(f"Error: Target directory not found: {TARGET_DIR}", file=sys.stderr)
        return
    except Exception as e:
        print(f"An error occurred while listing files: {e}", file=sys.stderr)
        return

    if not files_to_process:
        print(f"Warning: No files found in directory: {TARGET_DIR}. Exiting.", file=sys.stderr)
        return

    # 2. Loop through the file paths and call Julia
    for i, file_path in enumerate(files_to_process):
        
        arg_str = file_path # The full path is our argument string
        
        print(f"\n[Iteration {i+1}/{len(files_to_process)}]: Calling Julia with file path: {arg_str}")
        
        # Construct the command array
        command = [
            "julia",             # The Julia interpreter executable
            JULIA_SCRIPT_PATH,   # The script file to run
            "--model_path",
            arg_str              # The file path argument for this iteration
        ]
        
        # Execute the command
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Print the output received from the Julia script's standard output
            print("Julia Output:")
            print(result.stdout.strip())
            
        except subprocess.CalledProcessError as e:
            # Handle non-zero exit codes from the Julia script
            print(f"Error executing Julia script (Exit Code {e.returncode}):", file=sys.stderr)
            print(f"STDERR: {e.stderr}", file=sys.stderr)
            
        except FileNotFoundError:
            # Handle case where the 'julia' command itself is not found
            print("Error: 'julia' command not found. Please ensure Julia is installed and accessible in your system PATH.", file=sys.stderr)
            break
            
        # Optional: Add a short pause between runs
        time.sleep(0.5) 

    print("\n--- Python Driver Loop Finished ---")

if __name__ == "__main__":
    run_julia_script_in_loop()