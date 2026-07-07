#!/usr/bin/env python3
"""Convert a Stanford .nnet file into the pickled weight format used by this
repo's Julia loader (`get_nn` in utils/models.jl), plus an optional PyTorch
state-dict for the hyper-attack.

The Julia loader expects a pickle of a flat Python list

    [W1, b1, W2, b2, ..., WL, bL]

where each Wk is a numpy array of shape (input_dim, output_dim) -- the TRANSPOSE
of a PyTorch/NNet weight row-matrix -- and each bk has shape (output_dim,). The
forward pass in Julia is `transpose(W) * x .+ b` (see utils/models.jl).

.nnet files store, per layer, a row-major weight matrix of shape
(output_dim, input_dim) followed by the bias, and carry an input normalization
(mins, maxes, means, ranges). The network body expects NORMALIZED inputs:
norm_x_i = (x_i - mean_i) / range_i. We keep the raw layer weights (so the saved
net consumes normalized inputs) and separately emit the per-coordinate
verification box in normalized coordinates:

    lo_i = (min_i - mean_i) / range_i ,  hi_i = (max_i - mean_i) / range_i

When the .nnet carries no usable box (e.g. HAR: means=0, ranges=1, mins=-inf,
maxes=+inf) pass --box LO,HI to apply a uniform box (HAR = [-1,1]).

--negate-output multiplies the final layer's (W,b) by -1 so the network's argMAX
equals the original argMIN (ACAS Xu selects the minimal-score advisory, but this
tool's MIP is argmax-based).

--emit-pth also writes model.pth (a torch state_dict with keys fc1.weight,
fc1.bias, ...) beside model.p, for utils/hyper_attack.py.

Usage:
    python nnet_to_pickle.py IN.nnet OUT.p [--box LO,HI] [--negate-output] [--emit-pth]
"""
import argparse
import json
import os
import pickle
import numpy as np


def read_nnet(path):
    """Parse a .nnet file. Returns (weights, biases, mins, maxes, means, ranges).

    weights[k] has shape (out_k, in_k) exactly as stored in the file.
    """
    with open(path) as f:
        lines = [ln for ln in f if not ln.strip().startswith("//")]

    def row(i):
        return [t for t in lines[i].strip().split(",") if t != ""]

    num_layers, input_size, output_size, _max = (int(x) for x in row(0))
    layer_sizes = [int(x) for x in row(1)]
    # row(2) is the symmetric flag (unused here).
    mins = np.array([float(x) for x in row(3)], dtype=np.float64)
    maxes = np.array([float(x) for x in row(4)], dtype=np.float64)
    means = np.array([float(x) for x in row(5)], dtype=np.float64)
    ranges = np.array([float(x) for x in row(6)], dtype=np.float64)

    weights, biases = [], []
    idx = 7
    for k in range(num_layers):
        out_k, in_k = layer_sizes[k + 1], layer_sizes[k]
        W = np.empty((out_k, in_k), dtype=np.float64)
        for r in range(out_k):
            vals = [float(x) for x in row(idx)]
            assert len(vals) == in_k, f"layer {k} row {r}: got {len(vals)} != {in_k}"
            W[r] = vals
            idx += 1
        b = np.empty(out_k, dtype=np.float64)
        for r in range(out_k):
            b[r] = float(row(idx)[0])
            idx += 1
        weights.append(W)
        biases.append(b)

    return weights, biases, mins[:input_size], maxes[:input_size], \
        means[:input_size], ranges[:input_size]


def mlp_eval(weights, biases, z):
    """ReLU-MLP forward pass on an already-normalized input vector z."""
    for k, (W, b) in enumerate(zip(weights, biases)):
        z = W @ z + b
        if k < len(weights) - 1:
            z = np.maximum(z, 0.0)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nnet")
    ap.add_argument("out")
    ap.add_argument("--box", default=None,
                    help="uniform LO,HI box overriding the .nnet header")
    ap.add_argument("--negate-output", action="store_true",
                    help="negate the final layer (argmin->argmax, for ACAS)")
    ap.add_argument("--emit-pth", action="store_true",
                    help="also write model.pth (torch state_dict) beside model.p")
    args = ap.parse_args()

    weights, biases, mins, maxes, means, ranges = read_nnet(args.nnet)
    input_size = weights[0].shape[1]

    if args.negate_output:
        weights[-1] = -weights[-1]
        biases[-1] = -biases[-1]

    # Flat [W1(in,out), b1, W2, b2, ...] with weights transposed to (in,out).
    flat = []
    for W, b in zip(weights, biases):
        flat.append(np.ascontiguousarray(W.T))
        flat.append(np.ascontiguousarray(b))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(flat, f, protocol=2)

    # Verification box, in the NORMALIZED coordinates the saved net consumes.
    if args.box is not None:
        lo_s, hi_s = (float(v) for v in args.box.split(","))
        lo = np.full(input_size, lo_s)
        hi = np.full(input_size, hi_s)
    else:
        lo = (mins - means) / ranges
        hi = (maxes - means) / ranges
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            raise SystemExit("nnet header has no finite input box; pass --box")
    box_stub = os.path.splitext(args.out)[0] + "_box"
    with open(box_stub + ".json", "w") as f:
        json.dump({"lo": lo.tolist(), "hi": hi.tolist()}, f)
    # Julia-friendly sidecar (no JSON dependency): two comma-separated lines,
    # lo then hi, read by get_input_box in utils/datasets.jl.
    with open(box_stub + ".txt", "w") as f:
        f.write(",".join(repr(float(v)) for v in lo) + "\n")
        f.write(",".join(repr(float(v)) for v in hi) + "\n")
    box_path = box_stub + ".json"

    # ---- validation: saved (in,out) weights on normalized input must match the
    # (possibly negated) reference MLP on the same normalized samples ----
    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(200):
        z = rng.uniform(np.where(np.isfinite(lo), lo, -1.0),
                        np.where(np.isfinite(hi), hi, 1.0))
        saved = z.copy()
        for k in range(0, len(flat), 2):
            saved = flat[k].T @ saved + flat[k + 1]
            if k < len(flat) - 2:
                saved = np.maximum(saved, 0.0)
        ref = mlp_eval(weights, biases, z.copy())
        max_err = max(max_err, float(np.max(np.abs(saved - ref))))

    if args.emit_pth:
        import torch
        sd = {}
        for i, (W, b) in enumerate(zip(weights, biases), start=1):
            sd[f"fc{i}.weight"] = torch.tensor(W, dtype=torch.float32)   # (out,in)
            sd[f"fc{i}.bias"] = torch.tensor(b, dtype=torch.float32)
        pth_path = os.path.splitext(args.out)[0] + ".pth"
        torch.save(sd, pth_path)

    print(f"{os.path.basename(args.nnet)}: layers={[w.shape for w in weights]}"
          f"{'  [output negated]' if args.negate_output else ''}")
    print(f"  input_size={input_size}  hidden_neurons="
          f"{sum(w.shape[0] for w in weights[:-1])}  outputs={weights[-1].shape[0]}")
    print(f"  wrote {args.out} ({len(flat)} arrays) + {os.path.basename(box_path)}"
          f"{' + model.pth' if args.emit_pth else ''}")
    print(f"  box lo[:3]={np.round(lo[:3],4).tolist()} hi[:3]={np.round(hi[:3],4).tolist()}")
    print(f"  validation max|saved-reference| = {max_err:.2e}")


if __name__ == "__main__":
    main()
