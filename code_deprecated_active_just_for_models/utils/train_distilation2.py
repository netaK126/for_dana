import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from models import *
import pickle
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import sys
# ======================== פונקציות עזר ========================


def compute_margin_for_tag(logits: torch.Tensor, c_tag: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, C]
    c_tag:  [B]  (label index per sample, int64)
    returns: margin = logits[:, c_tag] - max_{j != c_tag} logits[:, j]
    """
    bsz = logits.size(0)
    chosen = logits[torch.arange(bsz, device=logits.device), c_tag]
    others = logits.clone()
    others[torch.arange(bsz, device=logits.device), c_tag] = float('-inf')
    max_others = others.max(dim=1).values
    return chosen - max_others

import torch
import torch.nn.functional as F

def distillation_loss_with_conf(
    student_logits,
    teacher_logits,
    labels,
    T: float = 4.0,
    alpha: float = 0.5,
    lambda_conf: float = 3.5,
    c_tag: int = 1,           # the fixed class whose confidence you care about
):
    """CE + KD + confidence |ΔC| (for fixed c_tag)"""

    # ---------- 1. Hard loss ----------
    ce_loss = F.cross_entropy(student_logits, labels)

    # ---------- 2. Proper KD loss ----------
    # Teacher: probabilities at temperature T (no grads needed)
    with torch.no_grad():
        p_teacher_T = F.softmax(teacher_logits / T, dim=1)

    # Student: log-probabilities at temperature T
    log_p_student_T = F.log_softmax(student_logits / T, dim=1)

    kd_loss = F.kl_div(log_p_student_T, p_teacher_T, reduction='batchmean') * (T * T)

    # ---------- 3. Confidence margin for fixed c_tag ----------
    def confidence_margin(logits: torch.Tensor, label_indices: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C]
        label_indices: [B], c_tag per sample
        returns: [B] margins C(x, N, c_tag)
        """
        B, C = logits.shape

        # logit for c_tag
        target_logits = logits.gather(1, label_indices.view(-1, 1)).squeeze(1)  # [B]

        # max over j != c_tag
        one_hot = F.one_hot(label_indices, num_classes=C).bool()         # [B, C]
        logits_others = logits.masked_fill(one_hot, float('-inf'))       # [B, C]
        max_others, _ = logits_others.max(dim=1)                          # [B]

        return target_logits - max_others                                 # [B]

    def confidence_margin_fixed_tag(logits: torch.Tensor, c_tag: int) -> torch.Tensor:
        B, C = logits.shape
        fixed_labels = torch.full((B,), c_tag, dtype=torch.long, device=logits.device)
        return confidence_margin(logits, fixed_labels)

    conf_teacher = confidence_margin_fixed_tag(teacher_logits, c_tag)  # [B]
    conf_student = confidence_margin_fixed_tag(student_logits, c_tag)  # [B]

    # L1 on confidence difference
    margin_loss = torch.mean(torch.abs(conf_student - conf_teacher))

    # ---------- 4. Combine ----------
    #(1 - alpha) * ce_loss + alpha * margin_loss
    return ce_loss, kd_loss, margin_loss


def save_model(model, itr, model_path):
    # if not os.path.isdir(model_path):
    #     print(f"Error: Directory '{model_path}' does not exist.")
    #     return

    # for filename in os.listdir(model_path):
    #     file_path = os.path.join(model_path, filename)
    #     if os.path.isfile(file_path):
    #         try:
    #             os.remove(file_path)
    #             print(f"Deleted: {file_path}")
    #         except OSError as e:
    #             print(f"Error deleting {file_path}: {e}")

    a = [np.transpose(p.cpu().detach().numpy()) for p in model.parameters()]
    pickle.dump(a, open(model_path + "_DanaTraining_model.p", "wb"))
    torch.save(model.state_dict(), model_path + "_model.pth")


# ======================== אימון ========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default="/root/Downloads/code_deprecated_active_just_for_models/models/4x10_distilation/")
    args = parser.parse_args()

    batch_size = args.batch_size
    num_epochs = args.epochs
    output_dir = args.output_dir
    loss_func_list = ["EC_and_KD_alphaVal"]
    alpha_vals_list = [0.3, 0.5, 0.7]
    original_stdout = sys.stdout 



    for loss_func in loss_func_list:
        for alpha_val in  alpha_vals_list:
            # Open the file in write mode ('w')
            file_path = output_dir+loss_func+"_alphaVal"+str(alpha_val)
            print(file_path)
            with open(file_path+".txt", 'w') as f:
                # Change stdout to point to the file object 'f'
                sys.stdout = f 
                teacher = FNN_4_10()
                teacher.load_state_dict(torch.load(r"/root/Downloads/code_deprecated_active_just_for_models/models/4x10/19/model.pth"))
                teacher.eval()

                student = FNN_4_10()
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                print(device, flush=True)
                student.to(device)
                teacher.to(device)

                transform = transforms.Compose([transforms.ToTensor()])
                mnist_train = dsets.MNIST(root='./data/', train=True, transform=transform, download=True)
                mnist_test = dsets.MNIST(root='./data/', train=False, transform=transform, download=True)
                train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
                test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

                # Use the pytorch-optimizer package name instead

                optimizer = torch.optim.Adam( student.parameters(), lr=1e-3, betas=(0.9, 0.99), weight_decay=1e-4)
                
                conf_diffs_real, conf_diffs_rand, max_diffs_real, max_diffs_rand = [], [], [], []

                # ======================== אימון ========================
                for p in teacher.parameters():
                    p.requires_grad_(False)
                new_data_inputs = []
                new_data_labels = []
                for epoch in range(num_epochs):
                    student.train()
                    running_loss = 0.0
                    hard_loss_epoch = 0.0
                    kd_loss_epoch = 0.0
                    margin_loss_epoch = 0.0
                    # for old_rand_input, old_rand_label in zip(new_data_inputs,new_data_labels):
                    #     student_logits_rand = student(old_rand_input)
                    #     with torch.no_grad():
                    #         teacher_logits_rand = teacher(old_rand_input)
                    #     old_rand_loss, hard_loss, kl_loss, margin_loss = distillation_loss_with_conf(
                    #         student_logits_rand, teacher_logits_rand, old_rand_label,
                    #         T=4.0, alpha=alpha_val, lambda_conf=3.0)

                    #     optimizer.zero_grad()
                    #     old_rand_loss.backward()
                    #     optimizer.step()

                    #     running_loss += old_rand_loss.item() * old_rand_input.size(0)
                    #     hard_loss_epoch += hard_loss.item()  * old_rand_input.size(0)
                    #     kd_loss_epoch+= kd_loss.item() * old_rand_input.size(0)
                    #     margin_loss_epoch += margin_loss.item()  * old_rand_input.size(0)
                            
                    for x, labels in train_loader:
                        x = x.to(device)
                        labels = labels.to(device)

                        # teacher is fixed
                        with torch.no_grad():
                            teacher_logits = teacher(x)

                        student_logits = student(x)

                        ce_loss, kd_loss, margin_loss = distillation_loss_with_conf(
                            student_logits, teacher_logits, labels,
                            T=2, alpha=alpha_val, lambda_conf=3.0
                        )
                        ce_loss = F.cross_entropy(student_logits, labels)
                        T=2
                        with torch.no_grad():
                            p_teacher_T = F.softmax(teacher_logits / T, dim=1)
                        # Student: log-probabilities at temperature T
                        log_p_student_T = F.log_softmax(student_logits / T, dim=1)
                        kd_loss = F.kl_div(log_p_student_T, p_teacher_T, reduction='batchmean') * (T * T)

                        loss = -1
                        if loss_func == "KD_and_conf_alphaVal":
                            if margin_loss<0:
                                continue
                            loss = (1 - alpha_val) * kd_loss + alpha_val * margin_loss
                        elif loss_func == "EC_and_conf_alphaVal":
                            if margin_loss<0:
                                continue
                            loss = (1 - alpha_val) * ce_loss + alpha_val * margin_loss
                        elif loss_func == "EC_and_KD_alphaVal":
                            loss = (1 - alpha_val) * ce_loss + alpha_val * kd_loss
                        elif loss_func == "EC":
                            loss = ce_loss
                        elif loss_func == "CONF":
                            if margin_loss<0:
                                continue
                            loss = margin_loss
                        elif loss_func == "KD":
                            loss = kd_loss   
                        if loss ==-1:
                            print("ERROR IN LOSS", flush=True)
                            exit(1)

                        loss_rand = 0
                        
                            
                        # for j in range(2):
                        #     # print("random")
                        #     x_rand = torch.rand_like(x)  # Uniform [0,1] noise

                        #     with torch.no_grad():
                        #         teacher_logits_rand = teacher(x_rand)

                        #     # No true labels — use teacher's pseudo-labels (argmax)
                        #     pseudo_labels = torch.argmax(teacher_logits_rand, dim=1)

                        #     student_logits_rand = student(x_rand)
                        #     loss_rand = loss_rand + distillation_loss_with_conf(
                        #         student_logits_rand, teacher_logits_rand, pseudo_labels,
                        #         T=4.0, alpha=0.5, lambda_conf=3.0)[0]
                        #     new_data_inputs.append(x_rand)
                        #     new_data_labels.append(pseudo_labels)

                        # ---------------------------
                        # 3. Total combined loss
                        # ---------------------------
                        # You can weigh the random input loss (e.g., 0.3) if you want
                        
                            
                        loss = loss + loss_rand

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * x.size(0)
                        hard_loss_epoch += ce_loss.item()  * x.size(0)
                        kd_loss_epoch+= kd_loss.item() * x.size(0)
                        margin_loss_epoch += margin_loss.item()  * x.size(0)

                    epoch_loss = running_loss / (len(train_loader.dataset)+len(new_data_inputs))
                    hard_loss_epoch = hard_loss_epoch / (len(train_loader.dataset)+len(new_data_inputs))
                    kd_loss_epoch = kd_loss_epoch / (len(train_loader.dataset)+len(new_data_inputs))
                    margin_loss_epoch = margin_loss_epoch / (len(train_loader.dataset)+len(new_data_inputs))
                    # print(f"Epoch {epoch+1}/{num_epochs} | loss={epoch_loss:.4f}")
                    print(f"Epoch {epoch+1}/{num_epochs} | loss={epoch_loss:.4f} | ce_loss={hard_loss_epoch:.4f} | kd_loss={kd_loss_epoch:.4f} | margin_loss={margin_loss_epoch:.4f}", flush=True)


                # ==================== שמירת מודל וגרפים ====================
                save_model(student, epoch, file_path)
                student.eval()

                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        predicted = student(x).argmax(1)
                        correct += (predicted == y).sum().item()
                        total += y.size(0)
                print("Student Accuracy:", correct / total, flush=True)

                correct = 0
                total = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        predicted = teacher(x).argmax(1)
                        correct += (predicted == y).sum().item()
                        total += y.size(0)
                print("teacher Accuracy:", correct / total, flush=True)

                correct = 0
                total = 0
                conf_diff = 0
                correct_teacher = 0
                correct_student = 0
                total_diff_labels = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)

                        teacher_logits = teacher(x)   # [B, C]
                        student_logits = student(x)   # [B, C]

                        predicted_teacher = teacher_logits.argmax(dim=1)   # [B]
                        predicted_student = student_logits.argmax(dim=1)   # [B]

                        correct_teacher += (predicted_teacher == y).sum().item()
                        correct_student += (predicted_student == y).sum().item()

                        diff_mask = (predicted_student != predicted_teacher)  # [B] bool
                        total_diff_labels += diff_mask.sum().item()

                        if diff_mask.any():
                            # Keep only mismatched samples
                            t_logits = teacher_logits[diff_mask]          # [M, C]
                            s_logits = student_logits[diff_mask]          # [M, C]
                            t_pred   = predicted_teacher[diff_mask]       # [M] indices of teacher's top class

                            # ---- teacher_conf: margin wrt teacher's own top class ----
                            # logit(teacher_top) - max_{j≠teacher_top} logit(j)
                            t_selected = t_logits.gather(1, t_pred.unsqueeze(1))   # [M, 1]

                            t_others = t_logits.clone()
                            t_others.scatter_(1, t_pred.unsqueeze(1), float('-inf'))
                            t_other_max, _ = t_others.max(dim=1, keepdim=True)     # [M, 1]

                            teacher_conf = (t_selected - t_other_max).squeeze(1)   # [M]

                            # ---- student_conf: use *teacher's* top class index ----
                            # logit_student(teacher_top) - max_{j≠teacher_top} logit_student(j)
                            s_selected = s_logits.gather(1, t_pred.unsqueeze(1))   # [M, 1]

                            s_others = s_logits.clone()
                            s_others.scatter_(1, t_pred.unsqueeze(1), float('-inf'))
                            s_other_max, _ = s_others.max(dim=1, keepdim=True)     # [M, 1]

                            student_conf = (s_selected - s_other_max).squeeze(1)   # [M]

                            # ---- track max |teacher_conf - student_conf| over all mismatched samples ----
                            batch_conf_diff = (teacher_conf - student_conf).abs().max().item()
                            conf_diff = max(conf_diff, batch_conf_diff)

                    print("max diff conf we found:", conf_diff)
                    print("differences in labels:", total, flush=True)
                    print("test set length:", len(test_loader))


                        

        sys.stdout = original_stdout
        print("Done with one model. training the next one...", flush=True)

    # plt.figure(figsize=(8,6))
    # plt.plot(conf_diffs_real, label='Mean |ΔC| (real data)', linewidth=2)
    # plt.plot(conf_diffs_rand, '--', label='Mean |ΔC| (random inputs)', linewidth=2)
    # plt.plot(max_diffs_real, label='Max |ΔC| (real data)', linewidth=1.5)
    # plt.plot(max_diffs_rand, '--', label='Max |ΔC| (random inputs)', linewidth=1.5)
    # plt.xlabel('Epoch')
    # plt.ylabel('|C_student - C_teacher|')
    # plt.title('Confidence Similarity — Mean & Max')
    # plt.legend()
    # plt.grid(True, linestyle=':')
    # plt.tight_layout()
    # plt.savefig("confidence_diff_stable.png")
    # plt.show()
