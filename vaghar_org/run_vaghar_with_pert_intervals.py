# import subprocess

# def run_julia_experiment(scenario_name, flags):
#     """
#     Calls run.jl with named flags.
#     flags: A dictionary of { "--flag_name": value }
#     """
#     # Build the command list: ['julia', 'run.jl', '--flag1', 'value1', ...]
#     cmd = ["julia", "run.jl"]
#     for flag, value in flags.items():
#         cmd.append(flag)
#         cmd.append(str(value).lower())
    
#     print(f"\n>>> Launching Scenario: {scenario_name}")
#     print(f"Executing: {' '.join(cmd)}")
    
#     try:
#         subprocess.run(cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Error in {scenario_name}: {e}")

# # Defining your 4 scenarios
# scenarios = {
#     "Intervals Only": {
#         "--use_perturbed_intervals": True,
#         "--activate_vaghgar_deps": False,
#         "--use_hyper_attack": False
#     },
#     "Full Perturbation": {
#         "--use_perturbed_intervals": True,
#         "--activate_vaghgar_deps": True,
#         "--use_hyper_attack": True
#     },
#     "Deps/Attack Only": {
#         "--use_perturbed_intervals": False,
#         "--activate_vaghgar_deps": True,
#         "--use_hyper_attack": True
#     },
#     "Baseline": {
#         "--use_perturbed_intervals": False,
#         "--activate_vaghgar_deps": False,
#         "--use_hyper_attack": False
#     },
# }

# if __name__ == "__main__":
#     for name, flags in scenarios.items():
#         run_julia_experiment(name, flags)


import subprocess
from itertools import product

def run_julia_experiment(scenario_name, flags):
    """
    מריץ את Julia עם הדגלים שהוגדרו.
    """
    cmd = ["julia", "run.jl"]
    for flag, value in flags.items():
        cmd.append(flag)
        cmd.append(str(value).lower())
    
    print(f"\n" + "="*50)
    print(f">>> Scenario: {scenario_name}")
    print(f"Executing: {' '.join(cmd)}")
    print("="*50)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"!!! Error in {scenario_name}: {e}")

def run_all_combinations():
    """
    מריץ אוטומטית את כל 32 השילובים האפשריים של הפרמטרים.
    """
    # רשימת הפרמטרים שביקשת
    parameter_names = [
        "--n2_fewer_binars_encoding",
        "--activate_vaghgar_deps",
        "--use_perturbed_intervals",
        "--use_hyper_attack",
        "--use_intervals"
    ]
    
    # יצירת כל השילובים האפשריים של True ו-False עבור 5 פרמטרים
    # (True, True, True, True, True), (True, True, True, True, False) ... וכו'
    combinations = list(product([True, False], repeat=len(parameter_names)))
    
    print(f"Starting total of {len(combinations)} experiments...")

    for i, combo in enumerate(combinations):
        # בניית מילון הדגלים להרצה הנוכחית
        flags = dict(zip(parameter_names, combo))
        
        # יצירת שם קריא להרצה לפי הדגלים שפעילים
        active_flags = [p.replace("--", "") for p in parameter_names if flags[p]]
        scenario_name = f"Run {i+1}: " + (", ".join(active_flags) if active_flags else "Baseline")
        
        run_julia_experiment(scenario_name, flags)

def run_specific_scenarios(selected_scenarios):
    """
    מאפשר לך להריץ רק רשימה ספציפית של שילובים שמעניינים אותך.
    """
    for i, combo in enumerate(selected_scenarios):
        # combo הוא מילון עם הפרמטרים
        scenario_name = f"Specific Run {i+1}"
        run_julia_experiment(scenario_name, combo)

if __name__ == "__main__":
    # --- אפשרות 1: להריץ את כל 32 השילובים האוטומטיים ---
    run_all_combinations()

    # --- אפשרות 2: להריץ רק שילובים ספציפיים (אם תרצה בעתיד) ---
    """
    my_selected_runs = [
        {
            "--n2_fewer_binars_encoding": True,
            "--activate_vaghgar_deps": True,
            "--use_perturbed_intervals": False,
            "--use_hyper_attack": True,
            "--use_intervals": True
        },
        # תוסיף עוד מילונים כאן...
    ]
    run_specific_scenarios(my_selected_runs)
    """