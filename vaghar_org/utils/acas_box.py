"""Input-box sidecars for the pretrained benchmark nets (acas/har).

These nets carry a per-coordinate input box instead of the image pipeline's
[0,1] domain, written beside the model by nnet_to_pickle.py:

  model_box.txt     the region we VERIFY (for ACAS 1_1, the phi1/phi2 region)
  model_domain.txt  the full normalized range the net was trained for

Both are two comma-separated lines, lo then hi. Julia reads model_box.txt via
get_input_box in utils/datasets.jl; this module is the Python-side reader, used
by the hyper-attack (which must sample inside the verified region for its hints
to be feasible) and by the N2 build step.
"""
import os

import numpy as np


def read_box_file(path):
    """Read a two-line 'lo' / 'hi' comma-separated sidecar into float arrays."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    lo = np.array([float(v) for v in lines[0].split(",")], dtype=np.float64)
    hi = np.array([float(v) for v in lines[1].split(",")], dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError(f"{path}: lo/hi length mismatch ({lo.size} vs {hi.size})")
    return lo, hi


def verification_box(model_dir):
    """(lo, hi) of the region being verified -- model_box.txt."""
    path = os.path.join(model_dir, "model_box.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no model_box.txt under {model_dir}")
    return read_box_file(path)


def full_domain(model_dir):
    """(lo, hi) of the net's full normalized input range -- model_domain.txt.

    Falls back to the verification box when the sidecar is absent, which is only
    equivalent if no property region was requested at conversion time.
    """
    path = os.path.join(model_dir, "model_domain.txt")
    if os.path.exists(path):
        return read_box_file(path)
    return verification_box(model_dir)
