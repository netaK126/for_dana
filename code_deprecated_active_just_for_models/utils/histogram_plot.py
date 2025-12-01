import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from models import *
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
import matplotlib
matplotlib.use("Agg")


def compute_confidences_fixed_c(model, dataloader, c_tag, device="cuda"):
    """
    Compute confidence(x, c_tag, N) for all x in dataloader.
    """
    model.to(device)
    model.eval()

    all_conf = []

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, _labels = batch
            else:
                inputs = batch

            inputs = inputs.to(device)
            logits = model(inputs)
            B = logits.size(0)

            score_c = logits[:, c_tag]

            masked = logits.clone()
            masked[:, c_tag] = -1e9
            max_other, _ = masked.max(dim=1)

            conf = score_c - max_other
            all_conf.append(conf.cpu())

    return torch.cat(all_conf).numpy()
def plot_confidence_split_by_sign(confidences, c_tag, delta_max, title=None):
    """
    Plot confidence vs index.
    If delta_max is much larger than the maximum confidence, use a broken x-axis.
    Otherwise, use a normal single axis and just draw the red line.
    """

    conf_pos = confidences[confidences >= 0]

    if conf_pos.size == 0:
        print(f"[WARN] No non-negative confidences for c_tag={c_tag}, skipping plot.")
        return

    x_vals = conf_pos
    y_vals = np.arange(len(conf_pos))

    delta_max = max(delta_max, 0)
    max_conf = np.max(x_vals)
    min_conf = np.min(x_vals)

    # --- Decide whether to use broken axis ---
    # Heuristic: only break if delta_max is much larger than max_conf
    # You can tune these numbers (factor and offset) as you like.
    ratio = delta_max / (max_conf + 1e-9)
    use_broken = (ratio > 1.5) or (delta_max - max_conf > 2.0)

    if not use_broken:
        # ==============================
        # Simple, regular (non-broken) axis
        # ==============================
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        ax.plot(x_vals, y_vals, '.', markersize=8)

        # Make sure both confidences and delta_max are visible
        xmin = min_conf - 0.1
        xmax = max(max_conf, delta_max) + 0.5
        ax.set_xlim(xmin, xmax)

        # Red vertical line
        ax.axvline(delta_max, color='red', linestyle='--', linewidth=1.4)

        # Rotated label near the line
        ax.text(
            x=delta_max + 0.05,
            y=len(y_vals) * 0.5,
            s=f"delta_max = {delta_max}",
            color='red',
            fontsize=17,
            fontweight='bold',
            rotation=90,
            va='center',
            ha='left'
        )

        ax.set_ylabel(f"Test Set Index (classified as {c_tag})", fontsize=17)
        ax.set_xlabel("Confidence", fontsize=17)

        if title:
            ax.set_title(title, fontsize=17)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_name = f"confidence_split_by_sign_cTag={c_tag}.png"
        plt.savefig(out_name)
        print(f"Saved regular-axis plot to {out_name}")
        return

    # ==============================
    # Broken x-axis version (your original behavior)
    # ==============================

    # Right side range should start slightly before delta_max
    right_min = delta_max - 0.5

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, sharey=True, figsize=(10, 10),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    # ---- LEFT AXIS (normal confidence values) ----
    ax_left.plot(x_vals, y_vals, '.', markersize=8)
    ax_left.set_xlim(min_conf - 0.1, max_conf + 0.1)

    # ---- RIGHT AXIS (just around delta_max) ----
    ax_right.plot(x_vals, y_vals, '.', markersize=8)
    ax_right.set_xlim(right_min, delta_max + 0.5)

    # Draw the vertical line on the right axis
    ax_right.axvline(delta_max, color='red', linestyle='--', linewidth=1.4)

    # Rotated text near the red line
    ax_right.text(
        x=delta_max + 0.05,
        y=len(y_vals) * 0.5,
        s=f"delta_max = {delta_max}",
        color='red',
        fontsize=17,
        fontweight='bold',
        rotation=90,
        va='center',
        ha='left'
    )

    # Hide spines between the two plots
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)

    # Add diagonal lines indicating the break
    d = .015  # size of diagonal slashes
    kwargs = dict(transform=ax_left.transAxes, color='k', clip_on=False)
    ax_left.plot((1-d, 1+d), (-d, d), **kwargs)
    ax_left.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

    kwargs = dict(transform=ax_right.transAxes, color='k', clip_on=False)
    ax_right.plot((-d, d), (-d, d), **kwargs)
    ax_right.plot((-d, d), (1-d, 1+d), **kwargs)

    # Labels and title
    ax_left.set_ylabel(f"Test Set Index (classified as {c_tag})", fontsize=17)
    fig.text(0.5, 0.04, "Confidence", ha='center', fontsize=17)

    if title:
        plt.suptitle(title, fontsize=17)

    ax_left.grid(True, alpha=0.3)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name = f"confidence_split_by_sign_cTag={c_tag}.png"
    plt.savefig(out_name)
    print(f"Saved broken-axis plot to {out_name}")

# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor()])
    mnist_test = dsets.MNIST(root='./data/', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=128, shuffle=False)

    model = FNN_4_10()
    model.load_state_dict(torch.load(r"/root/Downloads/code_deprecated_active_just_for_models/models/4x10_distilation/EC_and_conf_alphaVal_alphaVal0.3_model.pth"))
    model.eval()
    for c_tag, delta_max in zip([1,2,3,4,5],[27.40,16.49,11.037,4.48,7.22]):
        # for 0.5 [1.61,40.47,18.15,6.26,6.45]
        conf = compute_confidences_fixed_c(model, test_loader, c_tag, device="cuda")
        
        plot_confidence_split_by_sign(conf,c_tag,delta_max, title=f"Confidence split by sign (c = {str(c_tag)})")
