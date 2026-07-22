#!/usr/bin/env python3
"""ERAN CIFAR-10 ONNX -> vaghar_org .p + .pth, keyed by ONNX file-stem (ERAN_MODELS in models.py).

Normalization: from the ONNX Sub/Div nodes if present, else the per-dataset table. Conv nets get it
in the frozen 1x1 `norm` conv (diag 1/std, bias -mean/std); FC nets fold it into fc1. The net then
consumes raw [0,1] (matches get_nn's Julia side). .p uses the save_model convention
([np.ascontiguousarray(np.transpose(p)) for p in model.parameters()]) so the sweep's
train_extra_epochs re-saves stay consistent; .pth = state_dict().

Usage:
    python utils/onnx_to_pickle_eran.py <stem> <in.onnx> <out_dir> [--dataset cifar10 --k 3 --w 32 --h 32]
"""
import os
import sys
import pickle
import argparse
import numpy as np
import onnx
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import ERAN_MODELS, build_eran_model  # noqa: E402

_NORM = {  # dataset -> (mean, std)
    "cifar10": (np.array([0.4914, 0.4822, 0.4465]), np.array([0.2023, 0.1994, 0.2010])),
    "mnist":   (np.array([0.1307]),                 np.array([0.30810001])),
}


def _load(path):
    g = onnx.load(path).graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    convW = [(inits[n.input[1]], inits[n.input[2]]) for n in g.node if n.op_type == "Conv"]
    gemmW = [(inits[n.input[1]], inits[n.input[2]]) for n in g.node if n.op_type == "Gemm"]
    consts = [numpy_helper.to_array(a.t).flatten() for n in g.node if n.op_type == "Constant"
              for a in n.attribute if a.name == "value"]
    mean_std = (consts[0].astype(np.float64), consts[1].astype(np.float64)) if len(consts) >= 2 else None
    return convW, gemmW, mean_std


def convert(stem, onnx_path, out_dir, dataset="cifar10", k=3, w=32, h=32):
    convW, gemmW, mean_std = _load(onnx_path)
    mean, std = mean_std if mean_std is not None else _NORM[dataset]
    model = build_eran_model(stem, k, w, h).eval()
    with torch.no_grad():
        if hasattr(model, "convs"):                       # ERANConv: frozen 1x1 norm + body
            W = np.zeros((k, k, 1, 1), np.float32)
            for c in range(k):
                W[c, c, 0, 0] = 1.0 / std[c]
            model.norm.weight.copy_(torch.tensor(W))
            model.norm.bias.copy_(torch.tensor((-mean / std).astype(np.float32)))
            for c, (Wt, b) in zip(model.convs, convW):
                c.weight.copy_(torch.tensor(Wt)); c.bias.copy_(torch.tensor(b))
            for f, (Wt, b) in zip(model.fcs, gemmW):
                f.weight.copy_(torch.tensor(Wt)); f.bias.copy_(torch.tensor(b))
        else:                                             # ERANFullyConnected: fold norm into fc1
            m3 = np.repeat(mean, w * h); s3 = np.repeat(std, w * h)   # NCHW flatten order
            for j, (Wt, b) in enumerate(gemmW):
                Wt = Wt.astype(np.float64); b = b.astype(np.float64)
                if j == 0:
                    b = b - Wt @ (m3 / s3); Wt = Wt / s3[None, :]
                model.fcs[j].weight.copy_(torch.tensor(Wt.astype(np.float32)))
                model.fcs[j].bias.copy_(torch.tensor(b.astype(np.float32)))
    os.makedirs(out_dir, exist_ok=True)
    params = [np.ascontiguousarray(np.transpose(p.detach().cpu().numpy())) for p in model.parameters()]
    with open(os.path.join(out_dir, "model.p"), "wb") as f:
        pickle.dump(params, f)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
    # self-check: PyTorch(raw [0,1]) vs onnxruntime (raw if Sub/Div baked, else normalized)
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path)
        iname = sess.get_inputs()[0].name
        rng = np.random.default_rng(0)
        err = 0.0
        for _ in range(10):
            x = rng.uniform(0, 1, (1, k, w, h)).astype(np.float32)
            xo = x if mean_std is not None else \
                ((x - mean.reshape(1, k, 1, 1)) / std.reshape(1, k, 1, 1)).astype(np.float32)
            ref = sess.run(None, {iname: xo})[0]
            out = model(torch.tensor(x)).detach().numpy()
            err = max(err, float(np.abs(out - ref).max()))
        print(f"  self-check PyTorch(raw) vs onnx: {err:.2e}")
    except Exception as e:  # noqa: BLE001
        print(f"  (self-check skipped: {e})")
    print(f"{stem}: wrote model.p ({len(params)} arrays) + model.pth  ({'baked' if mean_std else 'table'} norm)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("onnx_path")
    ap.add_argument("out_dir")
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--w", type=int, default=32)
    ap.add_argument("--h", type=int, default=32)
    a = ap.parse_args()
    convert(a.stem, a.onnx_path, a.out_dir, a.dataset, a.k, a.w, a.h)
