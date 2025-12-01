import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# assume you already have this:
# from your_models import FNN_4_10
from models import *

def confidence_margin(logits: torch.Tensor, c_tag: int) -> torch.Tensor:
    """
    logits: [batch_size, num_classes]
    returns: [batch_size] where each entry is C(x, c_tag, N)
    C(x, c, N) = N(x)[c] - max_{j != c} N(x)[j]
    """
    # score for class c_tag
    c_scores = logits[:, c_tag]  # [B]

    # max over j != c_tag
    tmp = logits.clone()
    tmp[:, c_tag] = float("-inf")
    max_others, _ = tmp.max(dim=1)  # [B]

    return c_scores - max_others



def max_confidence_gap(teacher, quantized, c_tag: int, batch_size=256, device="cpu",
                       train=False, data_root="./data"):
    """
    teacher: original 4x10 network N
    quantized: quantized network Nq
    c_tag: integer in [0, num_classes-1]
    returns: scalar float, the max over all x in MNIST of |C(N) - C(Nq)|
    """
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root=data_root, train=train, transform=transform, download=True)
    loader = DataLoader(mnist, batch_size=batch_size, shuffle=False)

    teacher.to(device)
    quantized.to(device)
    teacher.eval()
    quantized.eval()

    max_gap = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            logits_T = teacher(x)      # [B, 10]
            logits_Q = quantized(x)    # [B, 10]

            C_T = confidence_margin(logits_T, c_tag)  # [B]
            C_Q = confidence_margin(logits_Q, c_tag)  # [B]

            diff = torch.abs(C_T - C_Q)  # [B]
            batch_max = diff.max()

            if batch_max > max_gap:
                max_gap = batch_max
        
        for j in range(10000):
            # print("random")
            x = torch.rand_like(x)
            x = x.to(device)

            logits_T = teacher(x)      # [B, 10]
            logits_Q = quantized(x)    # [B, 10]

            C_T = confidence_margin(logits_T, c_tag)  # [B]
            C_Q = confidence_margin(logits_Q, c_tag)  # [B]

            diff = torch.abs(C_T - C_Q)  # [B]
            batch_max = diff.max()

            if batch_max > max_gap:
                max_gap = batch_max


    return max_gap.item()

teacher = FNN_4_10()
teacher.load_state_dict(torch.load("/root/Downloads/code_deprecated_active_just_for_models/models/4x10/19/model.pth", map_location="cpu"))
teacher.eval()

# simple dynamic quantization
Nq = torch.quantization.quantize_dynamic(
    teacher, {nn.Linear}, dtype=torch.qint8
)
Nq.eval()

device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
for c in range(10):
    max_gap = max_confidence_gap(teacher, Nq, c_tag=c, device=device)
    print(f"Global max over x,c = {str(c)}: {max_gap}")
