"""Input samples for the ACAS/HAR benchmark nets.

The ACAS Xu nets ship pretrained (Julian et al., policy compression) and their
original training data -- a dynamic-programming lookup table of advisory
Q-values -- is not public. Confirmed absent from sisl/NNet, from the VNN-COMP
acasxu benchmark (which ships only .onnx + .vnnlib + a manifest CSV), and
asserted unavailable by several repair papers. HorizontalCAS does publish
training data but its README states it "is not related in any way to ACAS"
(3 inputs, notional system), so it is not a substitute.

The established workaround is to sample the input space uniformly and label with
the network itself as an oracle. Leino et al., "Self-Correcting Neural Networks
For Safe Classification", describing the ART dataset: "a synthetic test set for
each network, consisting of 5,000 points uniformly sampled from the specified
state space and labeled using the respective network as an oracle." CARE does
the same at 10K. That is what this module builds.

TARGETS ARE SCORES, NOT CLASS LABELS. The ACAS nets were trained by regression
on the five advisory Q-values per state, not by cross-entropy on a single best
advisory, so distilling N1's raw output vector keeps N2's fine-tuning on the same
objective the net was originally trained against. make_acas_datasets therefore
returns float targets of shape (N,5); train with MSE. Passing labels=True gives
argmax class labels instead, for a cross-entropy run that matches the image
pipeline's loss at the cost of changing the objective.

Samples are drawn from the net's normalized DOMAIN (<model>_domain.txt, the
.nnet header range the net was normalized for), NOT from <model>_box.txt. The box
is the region we verify -- for ACAS 1_1 that is the narrow phi1/phi2 property
region -- and training only inside it would leave N2 unconstrained everywhere
else.
"""
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset


def read_box_file(path):
    """Read a two-line 'lo' / 'hi' comma-separated sidecar into float arrays."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    lo = np.array([float(v) for v in lines[0].split(",")], dtype=np.float64)
    hi = np.array([float(v) for v in lines[1].split(",")], dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError(f"{path}: lo/hi length mismatch ({lo.size} vs {hi.size})")
    return lo, hi


def acas_domain(model_dir):
    """(lo, hi) of the net's normalized input domain.

    Prefers <model_dir>/model_domain.txt. Falls back to model_box.txt for models
    converted before the domain sidecar existed; that fallback is only correct
    when no property region was requested at conversion time, so it warns.
    """
    dom = os.path.join(model_dir, "model_domain.txt")
    if os.path.exists(dom):
        return read_box_file(dom)
    box = os.path.join(model_dir, "model_box.txt")
    if os.path.exists(box):
        print(f"  WARNING: {dom} missing, sampling from model_box.txt instead. "
              f"If that box is a property region the samples will not cover the "
              f"net's full domain. Re-run nnet_to_pickle.py to emit the domain.")
        return read_box_file(box)
    raise FileNotFoundError(f"no model_domain.txt or model_box.txt under {model_dir}")


def make_acas_datasets(model, model_dir, n_train=60000, n_test=10000, seed=42,
                       device=None, batch_size=4096, labels=False):
    """(trainset, testset) of domain-sampled inputs with N1 as oracle.

    Targets are N1's raw 5-vector of scores (float, for MSE) unless labels=True,
    which yields argmax class indices (for cross-entropy). Either way N2 = N1 +
    extra SGD stays close to N1, the same relation the image N2s have. Note that
    N1's "accuracy" on this set is 100% by construction; the meaningful drift
    number is N2's agreement with N1 after fine-tuning.
    """
    lo, hi = acas_domain(model_dir)
    n_in = int(lo.size)
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=(n_train + n_test, n_in)).astype(np.float32)
    # FNN_ACAS is built k=1, w=5, h=1, so shape as (N,C,H,W) like the image
    # pipeline; models.py flattens internally.
    x_t = torch.from_numpy(x).view(-1, 1, n_in, 1)

    was_on = next(model.parameters()).device
    dev = device if device is not None else was_on
    model.to(dev).eval()
    outs = []
    with torch.no_grad():
        for start in range(0, x_t.shape[0], batch_size):
            outs.append(model(x_t[start:start + batch_size].to(dev)).cpu())
    scores = torch.cat(outs, dim=0)
    model.to(was_on)

    counts = torch.bincount(scores.argmax(dim=1), minlength=5).tolist()
    print(f"  ACAS samples: {n_train} train / {n_test} test, "
          f"advisory counts under N1 = {counts}")
    if min(counts) == 0:
        print("  NOTE: some advisories never occur in the sampled domain.")

    y_t = scores.argmax(dim=1) if labels else scores
    return (TensorDataset(x_t[:n_train], y_t[:n_train]),
            TensorDataset(x_t[n_train:], y_t[n_train:]))
