import subprocess

def run_julia_experiment(scenario_name, flags):
    """
    Calls run.jl with named flags.
    flags: A dictionary of { "--flag_name": value }
    """
    # Build the command list: ['julia', 'run.jl', '--flag1', 'value1', ...]
    cmd = ["julia", "run.jl"]
    for flag, value in flags.items():
        cmd.append(flag)
        cmd.append(str(value).lower())
    
    print(f"\n>>> Launching Scenario: {scenario_name}")
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in {scenario_name}: {e}")

# Defining your 4 scenarios
scenarios = {
    # "Intervals Only": {
    #     "--use_perturbed_intervals": True,
    #     "--activate_vaghgar_deps": False,
    #     "--use_hyper_attack": False
    # },
    # "Full Perturbation": {
    #     "--use_perturbed_intervals": True,
    #     "--activate_vaghgar_deps": True,
    #     "--use_hyper_attack": True
    # },
    "Deps/Attack Only": {
        "--use_perturbed_intervals": False,
        "--activate_vaghgar_deps": True,
        "--use_hyper_attack": True
    },
    # "Baseline": {
    #     "--use_perturbed_intervals": False,
    #     "--activate_vaghgar_deps": False,
    #     "--use_hyper_attack": False
    # },
}

if __name__ == "__main__":
    for name, flags in scenarios.items():
        run_julia_experiment(name, flags)