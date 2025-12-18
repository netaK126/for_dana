import torch
import os
import random
import argparse
from models import *
import pickle

def perturb_and_save_model(model_path, eps):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return

    base_name = os.path.splitext(model_path)[0]
    new_model_path_p = base_name.replace("_p", "") + "_p.p"
    new_model_path_pth = base_name.replace("_p", "") + "_p.pth"

    print(f"--- Starting perturbation process for: {model_path} ---")

    try:
        state_dict = torch.load(model_path)
        print("✅ Model weights loaded successfully.")

        # --- 3. Apply the Perturbation ---
        perturbed_state_dict = {}
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor) and param.dim() > 0:
                perturbation = (torch.rand_like(param) * 2 - 1) * eps
                perturbed_param = param + perturbation
                perturbed_state_dict[name] = perturbed_param
                if random.random() < 0.01:
                     max_pert = torch.max(torch.abs(perturbation))
                     print(f"   [Debug] Perturbed layer: {name}. Max perturbation: {max_pert.item():.6f}")

            else:
                perturbed_state_dict[name] = param
        
        print(f"✅ Perturbation (max magnitude $\epsilon$={eps}) applied to all parameters.")
        torch.save(perturbed_state_dict, new_model_path_p)
        print(f"✅ New perturbed model saved successfully to: {new_model_path_p}")
        torch.save(perturbed_state_dict, new_model_path_pth)
        print(f"✅ New perturbed model saved successfully to: {new_model_path_pth}")

    except Exception as e:
        print(f"❌ An error occurred during the process: {e}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default="4x10", help='3x10, 3x50,3x100,3x50 cnn1, cnn2 or cnn3')
    parser.add_argument('--eps', type=float, default=0.01, help='epsilon for the weights')
    parser.add_argument('--model_path', type=str, default="/root/Downloads/code_deprecated_active_just_for_models/models/4x10_2/18/model.pth", help='model path')
    args = parser.parse_args()

    model_path = args.model_path
    model_type = args.model
    eps = args.eps
    
    perturb_and_save_model(model_path, eps)