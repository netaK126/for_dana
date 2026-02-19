# VHAGaR Transfer: Certifying Adversarial Robustness Transfer Between Neural Networks via Mixed-Integer Programming

---

## Abstract

We present *VHAGaR Transfer*, an extension of the VHAGaR (Verifier of Hazardous Attacks for Global Robustness) framework that addresses the following question: given two neural networks N1 and N2 with identical architecture, and a certified robustness margin δ₁ for N1, can we determine whether N2 also inherits this robustness—or exhibit a *transfer failure* where N2 succeeds on clean inputs but fails under adversarial perturbation? We formulate this as a Mixed-Integer Program (MIP) that simultaneously encodes all four network evaluations—N1(x), N2(x), N1(x+δ), N2(x+δ)—into a single optimization problem whose objective is the *confidence gap difference* δ_diff = C(N2, x, c) − C(N1, x, c). We introduce four problem variants parameterized by two binary flags, `c_tag_mode` and `n1_p_mode`, yielding different gap-reversal constraints that capture untargeted versus targeted transfer failures, and absolute versus relative degradation. A PGD-based hyper-attack with adaptive gradient scaling provides tight feasible lower bounds as warm-start hints to the MIP solver. The framework is implemented in Julia/JuMP with Gurobi and the MIPVerify encoding engine.

---

## 1. Introduction

Neural network robustness verification has produced a rich set of formal methods for certifying that a network N cannot be fooled by adversarial perturbations: given a clean input x, a perturbation budget ε, and a source class c_s, a verifier either proves that no perturbation of L∞ norm at most ε changes the network's prediction, or produces an explicit counterexample. VHAGaR [1] addresses this problem globally by formulating it as a dual-network MIP that simultaneously optimizes over the clean input and the perturbation, encoding both forward passes as integer constraints.

A natural and practically significant extension arises in the context of neural network comparison. Suppose N1 has been formally verified with certified robustness margin δ₁: for every input x that N1 classifies as c_s with C(N1, x, c_s) ≥ δ₁, no L∞ perturbation of size ε can cause misclassification. Now suppose N2 is a second network—perhaps a later training checkpoint, a model with higher clean accuracy, or a model trained with different hyperparameters—that appears superior to N1 on clean inputs: C(N2, x, c_s) > C(N1, x, c_s). Does N2 inherit N1's robustness? Or can we find an input where N2's apparent advantage disguises a greater vulnerability to adversarial perturbation?

We call this the *transfer proof problem*. A positive answer (no transfer failure found) provides a certificate that N2 is at least as robust as N1 on the class of inputs N1 was verified on. A negative answer (transfer failure found) provides an explicit witness: an input x and perturbation δ where N2 confidently outperforms N1 on clean x, yet fails under perturbation in a way N1 would not.

### 1.1 Contributions

1. We formulate the transfer proof problem as a four-network MIP with the confidence gap difference as the objective.
2. We identify and formalize four semantically distinct problem variants via two binary flags (`c_tag_mode`, `n1_p_mode`), covering untargeted and targeted robustness transfer in both relative and absolute forms.
3. We design an adaptive PGD hyper-attack that produces tight feasible lower bounds for MIP warm-starting.
4. We implement the full pipeline in Julia/JuMP with Gurobi, with optional constraint additions for inter-network interval bounds and intra-network perturbation dependencies.

---

## 2. Notation and Background

### 2.1 Neural Networks and Confidence Margins

Let N : ℝ^d → ℝ^C denote a fully-connected feedforward neural network with K hidden layers of ReLU activations and C output classes. The forward pass is:

```
h₀ = x
z_k = W_k · h_{k-1} + b_k,    k = 1, …, K+1
h_k = ReLU(z_k),               k = 1, …, K
N(x) = z_{K+1}
```

For a network N, input x, and class c ∈ {1, …, C}, the *confidence margin* is:

```
C(N, x, c)  =  N(x)_c  −  max_{j ≠ c}  N(x)_j
```

The confidence margin is positive if and only if N predicts class c for input x. A value of C(N, x, c) ≥ δ for some δ > 0 means c is the predicted class with a margin of at least δ over all other classes.

### 2.2 MIP Encoding of ReLU Networks

Each ReLU neuron with pre-activation z, lower bound l, and upper bound u is encoded via the big-M formulation. Let h = ReLU(z) = max(0, z):

- **Always inactive** (u ≤ 0): fix h = 0.
- **Always active** (l ≥ 0): fix h = z.
- **Split neuron** (l < 0 < u): introduce binary variable a ∈ {0, 1} and add:

```
h ≥ 0
h ≥ z
h ≤ u · a
h ≤ z − l · (1 − a)
```

The binary variable a = 1 encodes the neuron being active (h = z > 0) and a = 0 encodes it being inactive (h = 0). Bounds (l, u) are computed by interval arithmetic propagation, LP relaxation, or MIP tightening applied layer by layer before encoding.

### 2.3 Standard VHAGaR: Dual-Network MIP

In standard mode, VHAGaR encodes two copies of N—one applied to a clean input x and one applied to a perturbed input x′ = x + e—into a single MIP sharing the perturbation variable e ∈ [−ε, ε]^d. The formulation for source class c_s and target class c_t is:

```
maximize  C(N, x, c_s)
subject to
    x ∈ [0,1]^d,   x′ = x + e,   ‖e‖_∞ ≤ ε,   x′ ∈ [0,1]^d
    C(N, x′, c_t) > 0    [x′ is classified as c_t]
    MIP encoding of N(x) and N(x′)
```

If the optimal value is negative, no (x, e) pair satisfies the constraints and N is globally robust for class pair (c_s, c_t) at perturbation ε.

---

## 3. The Transfer Proof Problem

### 3.1 Problem Setup

Let N1 and N2 be two neural networks with the same architecture. Let δ₁ > 0 be a robustness certificate for N1, produced by a prior VHAGaR run: it represents the tightest upper bound on C(N1, x, c_s) over all adversarial examples—that is, for every input x with C(N1, x, c_s) ≥ δ₁, there exists no perturbation e with ‖e‖_∞ ≤ ε that causes N1 to misclassify x + e.

The *transfer proof problem* asks: is there an input x and perturbation e such that:

1. **(Certification condition)** N1 is certified on x: C(N1, x, c_s) ≥ δ₁.
2. **(Clean advantage condition)** N2 outperforms N1 on x: C(N2, x, c_s) ≥ C(N1, x, c_s).
3. **(Transfer failure condition)** N2 fails a robustness criterion on x + e (specified by `c_tag_mode` and `n1_p_mode`; see Section 4).

If such (x, e) exists, it is an explicit *transfer failure*: N2 appeared superior on clean input but failed adversarially where N1 was certified safe. If no such (x, e) exists within ‖e‖_∞ ≤ ε, we have a formal transfer certificate.

### 3.2 Objective: Confidence Gap Difference

The natural optimization objective for the transfer proof problem is the *confidence gap difference*:

```
δ_diff(x)  =  C(N2, x, c_s)  −  C(N1, x, c_s)
```

This quantity measures the excess confidence of N2 over N1 on clean input x for the source class. Maximizing δ_diff identifies the *worst-case transfer failure*: the example where N2's apparent clean advantage is most pronounced, yet N2 still fails under perturbation.

---

## 4. The Transfer Proof MIP

### 4.1 Decision Variables

The MIP has the following variable groups:

| Variable | Domain | Description |
|----------|--------|-------------|
| `v_in` | [0, 1]^d | Clean input x |
| `v_e` | [−ε, ε]^d | Perturbation e |
| `v_x0` | [0, 1]^d | Perturbed input x′ = x + e |
| N1(x) activations | ℝ / {0,1} | Internal variables for network version `n1_org`, layers 1..K |
| N2(x) activations | ℝ / {0,1} | Internal variables for network version `n2_org`, layers K+1..2K |
| N1(x′) activations | ℝ / {0,1} | Internal variables for network version `n1_pert`, layers 2K+1..3K |
| N2(x′) activations | ℝ / {0,1} | Internal variables for network version `n2_pert`, layers 3K+1..4K |
| `v_out_n1` | ℝ^C | Output logits N1(x) |
| `v_out_n2` | ℝ^C | Output logits N2(x) |
| `v_out_n1_p` | ℝ^C | Output logits N1(x′) |
| `v_out_n2_p` | ℝ^C | Output logits N2(x′) |
| `conf_n1_x` | ℝ | C(N1, x, c_s) |
| `conf_n2_x` | ℝ | C(N2, x, c_s) |
| `conf_n1_xp` | ℝ | C(N1, x′, c_pert) |
| `conf_n2_xp` | ℝ | C(N2, x′, c_pert) |
| `delta_diff` | ℝ | C(N2, x, c_s) − C(N1, x, c_s) |

### 4.2 Constraints

Let ε₀ = 10⁻³ be a numerical tolerance separating strict and non-strict inequalities.

---

**(C1) Perturbation constraint:**

```
v_x0  =  v_in + v_e
```

The perturbed input is the clean input plus the perturbation. Combined with the domain bounds on `v_e`, this enforces ‖e‖_∞ ≤ ε. Combined with the bounds on `v_x0`, it enforces x + e ∈ [0, 1]^d.

---

**(C2) Four-network MIP encodings:**

Each of the four forward passes is encoded as a sequence of MIP constraints following the big-M ReLU formulation of Section 2.2. The four encodings are performed with a shared global layer counter, so each network copy occupies a disjoint range of layer indices:

```
[N1(x)]  → layers  1 .. K      (network_version = "n1_org")
[N2(x)]  → layers  K+1 .. 2K   (network_version = "n2_org")
[N1(x′)] → layers  2K+1 .. 3K  (network_version = "n1_pert")
[N2(x′)] → layers  3K+1 .. 4K  (network_version = "n2_pert")
```

N1(x) and N2(x) take `v_in` as their shared input. N1(x′) and N2(x′) take `v_x0` as their shared input. The perturbation variable `v_e` is thus the single coupling variable linking the clean and perturbed halves of the encoding.

---

**(C3) Confidence margin encoding:**

Each confidence margin C(N, x, c) = N(x)_c − max_{j≠c} N(x)_j is encoded via big-M. For a vector of output logits `v_out` and class c:

```
conf  =  v_out[c] − max_kk
max_kk  ≥  v_out[j]                             for all j ≠ c
max_kk  ≤  v_out[j] + M_conf · (1 − a_j)        for all j ≠ c
∑_{j≠c}  a_j  =  1,    a_j ∈ {0, 1}
```

where M_conf = 10⁶ is a big-M constant for the maximum encoding. This yields `conf = v_out[c] − max_{j≠c} v_out[j]` as an exact linear-integer formulation.

This encoding is applied four times to produce `conf_n1_x`, `conf_n2_x`, `conf_n1_xp`, `conf_n2_xp`.

---

**(C4) N1 certification constraint:**

```
conf_n1_x  ≥  δ₁ + ε₀
```

This restricts x to the certified region of N1: the clean input must be one that N1 was formally verified to handle correctly, with a margin of at least δ₁. Here δ₁ is read from the `best_bound` column of a prior VHAGaR results file (the MIP's proven upper bound on the optimal value).

---

**(C5) Clean confidence gap constraints:**

```
delta_diff  =  conf_n2_x − conf_n1_x
delta_diff  ≥  0
```

The first constraint defines δ_diff as the difference in confidence margins on the clean input. The second constraint enforces that N2 is at least as confident as N1 on x, ensuring we search only in regions where N2 holds a clean advantage.

---

**(C6) Gap-reversal constraint (parameterized by `c_tag_mode` and `n1_p_mode`):**

This is the core constraint that defines the transfer failure condition. Its precise form depends on the two binary flags. Let c_pert denote the class used to measure perturbed-input confidence:

```
c_pert  =  c_s    if  c_tag_mode = true   (source class; untargeted)
c_pert  =  c_t    if  c_tag_mode = false  (target class; targeted)
```

The four variants of (C6) are described in full in Section 5.

---

**Objective:**

```
maximize  delta_diff
```

The solver finds the input x and perturbation e within the feasible region of (C1)–(C6) that maximizes the confidence advantage of N2 over N1 on the clean input, while satisfying the transfer failure constraint.

---

### 4.3 Solver Configuration

The MIP is solved with Gurobi using the following settings:

| Attribute | Value | Purpose |
|-----------|-------|---------|
| `MIPFocus` | 3 | Focus on proving/improving the objective bound |
| `MIPGap` | 0.01 | Accept 1% suboptimality |
| `Threads` | 32 | Parallel branch-and-bound |
| `TimeLimit` | (user-specified) | Hard cutoff |
| `Cutoff` | PGD lower bound | Prune branches with objective ≤ known feasible |

The `Cutoff` value is provided by the hyper-attack (Section 6), allowing Gurobi to skip large portions of the branch-and-bound tree that cannot improve upon the already-known feasible solution.

---

## 5. Problem Variants: `c_tag_mode` and `n1_p_mode`

### 5.1 Overview

Two binary parameters govern the gap-reversal constraint (C6):

- **`c_tag_mode`** controls *which class* is used to evaluate N2's behavior on the perturbed input, determining whether the failure is *untargeted* (N2 loses confidence in the source class) or *targeted* (N2 gains confidence in a target class).
- **`n1_p_mode`** controls whether the gap-reversal constraint is *relative* (comparing N2(x′) to N1(x′)) or *absolute* (constraining N2(x′) alone).

Together they yield four problem variants. We denote the numerical tolerance ε₀ = 10⁻³.

---

### 5.2 Variant A: `c_tag_mode = true`, `n1_p_mode = true`

**Setting:** c_pert = c_s, constraint uses both networks' perturbed outputs.

**Gap-reversal constraint (C6):**
```
conf_n2_xp  −  conf_n1_xp  ≤  −ε₀
```

**Expanded form:**
```
C(N2, x′, c_s)  −  C(N1, x′, c_s)  ≤  −ε₀
```

**Interpretation:** Under adversarial perturbation x′ = x + e with ‖e‖_∞ ≤ ε, N2's confidence margin in the *source class* c_s is strictly lower than N1's confidence margin in c_s by at least ε₀. This is a *relative untargeted* transfer failure: N2 degrades more than N1 on the source class under perturbation.

Combined with (C5), this means:

```
C(N2, x, c_s) ≥ C(N1, x, c_s)       [N2 is more confident on clean x]
C(N2, x′, c_s) < C(N1, x′, c_s)      [N2 degrades more under perturbation]
```

The gap has *reversed*: N2 goes from being more confident than N1 to being less confident, under perturbation.

**Full problem (Variant A):**
```
maximize  δ_diff  =  C(N2, x, c_s) − C(N1, x, c_s)
subject to
    x ∈ [0,1]^d,   x′ = x + e,   ‖e‖_∞ ≤ ε,   x′ ∈ [0,1]^d
    C(N1, x, c_s) ≥ δ₁ + ε₀                   [C4: N1 certified]
    δ_diff = C(N2, x, c_s) − C(N1, x, c_s)     [C5a]
    δ_diff ≥ 0                                  [C5b: N2 advantage]
    C(N2, x′, c_s) − C(N1, x′, c_s) ≤ −ε₀     [C6: relative gap reversal]
    MIP encodings of N1(x), N2(x), N1(x′), N2(x′)
```

---

### 5.3 Variant B: `c_tag_mode = true`, `n1_p_mode = false`

**Setting:** c_pert = c_s, constraint uses only N2's perturbed output.

**Gap-reversal constraint (C6):**
```
conf_n2_xp  ≤  −ε₀
```

**Expanded form:**
```
C(N2, x′, c_s)  ≤  −ε₀
```

**Interpretation:** Under adversarial perturbation, N2 *misclassifies* the perturbed input x′ away from the source class c_s. N2 predicts some class other than c_s for x′. This is an *absolute untargeted* transfer failure: N2 actually crosses the decision boundary into misclassification, regardless of what N1 does on x′.

Note that Variant B is a *stronger* feasibility condition than Variant A. Variant A requires only that N2's confidence drops more than N1's; N1 might itself also misclassify x′, but N2 must be even worse. Variant B requires N2 to definitively misclassify x′, making it a more interpretable but also harder-to-find certificate.

**Full problem (Variant B):**
```
maximize  δ_diff  =  C(N2, x, c_s) − C(N1, x, c_s)
subject to
    x ∈ [0,1]^d,   x′ = x + e,   ‖e‖_∞ ≤ ε,   x′ ∈ [0,1]^d
    C(N1, x, c_s) ≥ δ₁ + ε₀                   [C4]
    δ_diff = C(N2, x, c_s) − C(N1, x, c_s)     [C5a]
    δ_diff ≥ 0                                  [C5b]
    C(N2, x′, c_s) ≤ −ε₀                        [C6: absolute misclassification]
    MIP encodings of N1(x), N2(x), N1(x′), N2(x′)
```

---

### 5.4 Variant C: `c_tag_mode = false`, `n1_p_mode = true`

**Setting:** c_pert = c_t ≠ c_s, constraint uses both networks' perturbed outputs.

**Gap-reversal constraint (C6):**
```
conf_n2_xp  −  conf_n1_xp  ≥  ε₀
```

**Expanded form:**
```
C(N2, x′, c_t)  −  C(N1, x′, c_t)  ≥  ε₀
```

**Interpretation:** Under adversarial perturbation, N2 gains *relatively more* confidence in the target class c_t than N1 does. N2 is more susceptible than N1 to a targeted adversarial attack toward c_t. This is a *relative targeted* transfer failure: the clean advantage of N2 is accompanied by higher vulnerability to targeted attacks.

In the iteration structure, `c_tag_mode = false` loops over all target classes c_t ≠ c_s, yielding a separate MIP for each pair (c_s, c_t).

**Full problem (Variant C):**
```
maximize  δ_diff  =  C(N2, x, c_s) − C(N1, x, c_s)
subject to
    x ∈ [0,1]^d,   x′ = x + e,   ‖e‖_∞ ≤ ε,   x′ ∈ [0,1]^d
    C(N1, x, c_s) ≥ δ₁ + ε₀                   [C4]
    δ_diff = C(N2, x, c_s) − C(N1, x, c_s)     [C5a]
    δ_diff ≥ 0                                  [C5b]
    C(N2, x′, c_t) − C(N1, x′, c_t) ≥ ε₀       [C6: relative targeted gap]
    MIP encodings of N1(x), N2(x), N1(x′), N2(x′)
```

---

### 5.5 Variant D: `c_tag_mode = false`, `n1_p_mode = false`

**Setting:** c_pert = c_t ≠ c_s, constraint uses only N2's perturbed output.

**Gap-reversal constraint (C6):**
```
conf_n2_xp  ≥  ε₀
```

**Expanded form:**
```
C(N2, x′, c_t)  ≥  ε₀
```

**Interpretation:** Under adversarial perturbation, N2 *confidently predicts the target class c_t* for the perturbed input x′. The perturbed input x′ is an explicit targeted adversarial example for N2. This is an *absolute targeted* transfer failure and the strongest form of transfer failure in this framework: x′ is a complete attack success against N2.

Combined with (C4), this means N1 was certified on x (and hence, by the prior VHAGaR certificate, cannot be misclassified by any perturbation of size ε at the input x), yet x′ = x + e is a successful targeted adversarial example for N2.

**Full problem (Variant D):**
```
maximize  δ_diff  =  C(N2, x, c_s) − C(N1, x, c_s)
subject to
    x ∈ [0,1]^d,   x′ = x + e,   ‖e‖_∞ ≤ ε,   x′ ∈ [0,1]^d
    C(N1, x, c_s) ≥ δ₁ + ε₀                   [C4]
    δ_diff = C(N2, x, c_s) − C(N1, x, c_s)     [C5a]
    δ_diff ≥ 0                                  [C5b]
    C(N2, x′, c_t) ≥ ε₀                         [C6: targeted misclassification]
    MIP encodings of N1(x), N2(x), N1(x′), N2(x′)
```

---

### 5.6 Summary and Semantic Relationships

The four variants form a 2×2 lattice:

```
                    n1_p_mode = true            n1_p_mode = false
                    (relative failure)          (absolute failure)
                  ┌────────────────────────┬────────────────────────┐
c_tag_mode = true │ Variant A              │ Variant B              │
(untargeted)      │ C(N2,x′,cs)            │ C(N2,x′,cs) ≤ −ε₀     │
                  │  − C(N1,x′,cs) ≤ −ε₀  │                        │
                  ├────────────────────────┼────────────────────────┤
c_tag_mode = false│ Variant C              │ Variant D              │
(targeted)        │ C(N2,x′,ct)            │ C(N2,x′,ct) ≥ ε₀      │
                  │  − C(N1,x′,ct) ≥ ε₀   │                        │
                  └────────────────────────┴────────────────────────┘
```

**Feasibility relationships:** Variant B is a strictly stronger constraint than Variant A (a solution to Variant B where N1 is also robust satisfies Variant A, but not conversely). Similarly, Variant D is a strictly stronger constraint than Variant C (when N1 does not misclassify c_t). Thus:

- Variants B and D have *smaller* feasible sets (harder to find solutions) but yield *stronger* transfer failure certificates.
- Variants A and C have *larger* feasible sets but require only relative degradation of N2 compared to N1.

---

## 6. Hyper-Attack: PGD Warm-Start

### 6.1 Role of the Hyper-Attack

The branch-and-bound MIP solver prunes nodes with objective bound below the incumbent (`Cutoff`). A tight lower bound therefore dramatically reduces solve time. The hyper-attack runs a PGD-based optimization in continuous relaxation to find a feasible point for constraints (C4)–(C6), producing a lower bound on the optimal δ_diff. If the attack succeeds, this lower bound is passed to Gurobi as `Cutoff` and as a MIP start (warm-start hints for binary ReLU variables).

### 6.2 Input Candidate Pool

A pool of M input candidates is sampled to initialize the PGD:

1. Collect images from the training set, test set, and random uniform noise.
2. Filter to those N1 classifies as c_s with C(N1, x, c_s) ≥ δ₁ (satisfying constraint C4).
3. Sort by C(N1, x, c_s) in descending order; subsample uniformly at stride ⌊|filtered| / M⌋ to cover the confidence spectrum.

### 6.3 PGD Attack Loop

The attack jointly optimizes over the input x ∈ [0, 1]^d and perturbation e ∈ [−ε, ε]^d (or the scalar/spatial perturbation for non-L∞ types) by PGD gradient ascent on a composite loss. At iteration t:

**Compute clean outputs:**
```
Δ_diff  =  C(N2, x, c_s)  −  C(N1, x, c_s)
```

**Compute perturbed input and gap:**
```
x′  =  clip(x + e, 0, 1)
```

For `n1_p_mode = false`:
```
gap_pert  =  C(N2, x′, c_pert)
```

For `n1_p_mode = true`:
```
gap_pert  =  C(N2, x′, c_pert)  −  C(N1, x′, c_pert)
```

where c_pert = c_s if `c_tag_mode = true`, c_pert = c_t otherwise.

**Adaptive scaling:**

To prevent one term from dominating the gradient when the two quantities differ in magnitude, an adaptive weight is computed per sample:

```
λ̃  =  |Δ_diff| / (|gap_pert| + η),    η = 10⁻⁹
```

This scales the gap penalty proportionally to the current δ_diff magnitude, maintaining a balanced optimization landscape throughout training.

**Loss computation:**

```
For c_tag_mode = true:   ℒ  =  Δ_diff  +  λ₀ · λ̃ · gap_pert
For c_tag_mode = false:  ℒ  =  Δ_diff  −  λ₀ · λ̃ · gap_pert
```

where λ₀ = 1.01 is a base scaling hyperparameter. The gradient of ℒ with respect to x and e is computed by backpropagation through both N1 and N2.

**Gradient ascent step:**
```
x  ←  clip( x  +  α · sign(∇_x ℒ),  0, 1 )
e  ←  clip( e  +  α · sign(∇_e ℒ),  −ε, ε )
```

where α = 0.01 is the step size and T = 500 is the default number of iterations.

### 6.4 Feasibility Filter

After T iterations, candidates are tested against all four feasibility conditions:

1. N1 classifies x as c_s: argmax N1(x) = c_s.
2. **Constraint (C4):** C(N1, x, c_s) ≥ δ₁.
3. **Constraint (C5b):** Δ_diff ≥ 0.
4. **Constraint (C6):**

| Mode | Feasibility condition |
|------|-----------------------|
| c_tag_mode = true  | gap_pert < 0 |
| c_tag_mode = false | gap_pert > 0 |

From the feasible candidates, the one with the largest Δ_diff is selected as the best warm-start.

### 6.5 Warm-Start Hint Generation

For the best feasible candidate (x*, e*, x*′ = clip(x* + e*, 0, 1)), ReLU activation patterns are extracted for all four network copies:

```
act_n1_x  = activations of N1(x*)     [layers 1..K]
act_n2_x  = activations of N2(x*)     [layers K+1..2K]
act_n1_xp = activations of N1(x*′)    [layers 2K+1..3K]  (if n1_p_mode)
act_n2_xp = activations of N2(x*′)    [layers 3K+1..4K]
```

Each activation value is thresholded to produce a binary hint:
- Post-ReLU value > 1 − τ (τ = 0.01): binary hint = 1 (neuron active)
- Post-ReLU value < τ: binary hint = 0 (neuron inactive)
- Otherwise: binary hint = −1 (unknown; solver decides)

The hint strings are written to `/tmp/` files and read by the Julia MIP solver, which uses them to initialize the binary variables `a_{k,i}` in the big-M ReLU encoding. The best Δ_diff value is passed as the `Cutoff` parameter to Gurobi.

---

## 7. Optional Constraint Enhancements

Three families of optional constraints can be added to tighten the MIP relaxation.

### 7.1 Inter-Network Interval Constraints (`use_intervals`)

When N1 and N2 are "close" networks (e.g., different training checkpoints), their activation differences at each layer are small. Layer-by-layer interval bounds on h_k^{N2}(x) − h_k^{N1}(x) can be derived from the weight difference ΔW_k = W_k^{N2} − W_k^{N1} and propagated through the network. Adding these as linear constraints in the MIP tightens the relaxation between the `n1_org` and `n2_org` encoding blocks.

### 7.2 Perturbation Interval Constraints (`use_perturbed_intervals`)

For each network copy Nᵢ, bounds on the activation difference Δh_k = h_k^{Nᵢ}(x + e) − h_k^{Nᵢ}(x) at each layer k can be propagated using the perturbation bound ‖e‖_∞ ≤ ε. These constraints link the clean and perturbed encoding blocks of each network:

```
perturbed_interval_constraints(N1, "n1_org", "n1_pert")
perturbed_interval_constraints(N2, "n2_org", "n2_pert")
```

### 7.3 VHAGaR Dependency Constraints (`activate_vaghgar_deps`)

VHAGaR's standard mode computes a φ-dependency matrix tracking, for each pair of corresponding neurons across the clean and perturbed network copies, whether their activations are equal, monotonically related, anti-monotonically related, or unknown. These dependency constraints—linear equality and inequality constraints on the split neuron binary variables—can be reused in the transfer MIP:

```
# N1: link clean layers 1..K to perturbed layers 2K+1..3K
perturbation_dependencies(N1; activation_start=1, layers_offset=2K)
# N2: link clean layers K+1..2K to perturbed layers 3K+1..4K
perturbation_dependencies(N2; activation_start=K+1, layers_offset=2K)
```

These constraints can significantly reduce the effective number of free binary variables, accelerating branch-and-bound.

---

## 8. Full Execution Flow

The complete transfer proof pipeline for a given (c_s, c_t) pair is:

```
1. Load N1, N2
2. Read δ₁ from prior VHAGaR results (best_bound for class pair)
3. If δ₁ ≤ 0: skip (N1 not certified for this pair)
4. [Optional] Run hyper-attack → suboptimal_solution, activation hints
5. Build four-network MIP:
   - Create shared variables v_in, v_e, v_x0
   - Encode N1(v_in)  → v_out_n1   [layers  1.. K, "n1_org"]
   - Encode N2(v_in)  → v_out_n2   [layers K+1..2K, "n2_org"]
   - Encode N1(v_x0)  → v_out_n1_p [layers 2K+1..3K, "n1_pert"]
   - Encode N2(v_x0)  → v_out_n2_p [layers 3K+1..4K, "n2_pert"]
6. [Optional] Apply warm-start hints to binary variables
7. [Optional] Add dependency constraints
8. [Optional] Add inter-network interval constraints
9. [Optional] Add perturbation interval constraints
10. Add objective and constraints (C4)–(C6) via mip_set_transfer_property
11. Set Gurobi attributes (MIPFocus=3, Cutoff, TimeLimit, MIPGap=0.01)
12. optimize!()
13. Log: incumbent_obj, best_bound, solve_time
14. Save results: source, target, incumbent_obj, best_bound, solve_time
```

The result is interpreted as:
- **incumbent_obj > 0 with best_bound > 0**: Transfer failure found. incumbent_obj is the δ_diff of the discovered counterexample.
- **best_bound ≤ 0**: No transfer failure exists with the given constraints (formal transfer certificate, within MIP gap tolerance).
- **Time limit reached**: Inconclusive; best_bound gives an upper bound on the maximum possible δ_diff.

---

## 9. Discussion

### 9.1 Variant Selection Guidelines

The four variants trade off between finding power and certification strength:

- **Variant B** (`c_tag_mode=true, n1_p_mode=false`) is the most practically interpretable: a solution is an explicit untargeted adversarial example for N2, with clean input certified for N1. It is also the easiest to find (largest feasible set among untargeted variants) and produces the clearest security implication.

- **Variant D** (`c_tag_mode=false, n1_p_mode=false`) is the targeted analog: a solution is a targeted adversarial example for N2. Useful when one is concerned about specific misclassification targets (e.g., safety-critical class pairs).

- **Variants A and C** (`n1_p_mode=true`) are more conservative: they require N2 to fail *relative to N1*, not absolutely. A solution to Variant A is not necessarily an adversarial example for N2 (N2 might still classify x′ correctly, just less confidently than N1). These variants are appropriate when one wants to certify that N2 never degrades faster than N1, a weaker but still meaningful robustness property.

### 9.2 Computational Complexity

The transfer proof MIP has four times as many network-encoding variables as a standard single-copy verification. Specifically:

- **Variables:** O(4 · K · n) binary + O(4 · K · n) continuous (internal activations) + O(2C · 4) binary (confidence margin encodings).
- **Constraints:** O(4 · K · n · 4) big-M ReLU constraints + O(4C) confidence margin constraints.

For a 4×10 MNIST network (K = 3, n = 10, C = 10, d = 784), this gives approximately 120 binary activation variables plus additional confidence-margin binary variables, manageable within Gurobi's branch-and-bound.

### 9.3 Tightness of Transfer Certificates

The transfer certificate (best_bound ≤ 0) is formally sound: within the 1% MIPGap tolerance, no input x and perturbation e satisfy all constraints. However, the tightness of the certificate depends on the quality of bound tightening during encoding. Loose bounds produce more split neurons and a larger MIP, potentially leading to timeouts rather than certificates. The optional constraint additions (Sections 7.1–7.3) directly address this by reducing the effective search space.

---

## 10. Conclusion

VHAGaR Transfer formalizes the question of adversarial robustness transfer between neural networks as a four-network MIP. The two binary flags `c_tag_mode` and `n1_p_mode` yield four semantically distinct problem variants:

| Variant | c_tag_mode | n1_p_mode | Transfer failure type |
|---------|------------|-----------|----------------------|
| A | true | true | Relative untargeted: N2 degrades more than N1 from source confidence |
| B | true | false | Absolute untargeted: N2 misclassifies perturbed input (untargeted) |
| C | false | true | Relative targeted: N2 gains more target confidence than N1 |
| D | false | false | Absolute targeted: N2 predicts target class on perturbed input |

The framework provides both an attack mode (finding explicit transfer failure witnesses) and a verification mode (certifying transfer with a proven bound). The PGD hyper-attack with adaptive gradient scaling accelerates MIP solving by providing tight lower bounds and activation-level warm-start hints. Optional constraint additions further tighten the relaxation using inter-network interval propagation and VHAGaR dependency propagation.

---

## References

[1] VHAGaR: Verifier of Hazardous Attacks for Global Robustness. Source code: `vaghar_org/`.

[2] Anderson, R., et al. "Strong mixed-integer programming formulations for trained neural networks." *Mathematical Programming* 183 (2020): 3–39.

[3] Tjeng, V., Xiao, K., Tedrake, R. "Evaluating Robustness of Neural Networks with Mixed Integer Programming." *ICLR 2019*.

[4] Madry, A., et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR 2018*.

[5] Katz, G., et al. "Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks." *CAV 2017*.

[6] Gurobi Optimization, LLC. *Gurobi Optimizer Reference Manual*, 2023.

---

## Appendix: Key Implementation Details

### A.1 Layer Indexing Convention

Global layer indices in `layers_info_dict[(layer, neuron)]`:

```
n1_org:   layer ∈ {1, …, K}
n2_org:   layer ∈ {K+1, …, 2K}
n1_pert:  layer ∈ {2K+1, …, 3K}
n2_pert:  layer ∈ {3K+1, …, 4K}
```

For a 4×10 network: K = 3, so n1_org uses layers 1–3, n2_org uses 4–6, n1_pert uses 7–9, n2_pert uses 10–12.

### A.2 JuMP Variable Naming

Binary ReLU variables in JuMP follow the naming convention:
```
{network_version}a_layerCount{rel_layer}_neuronCount{n}_{abs_layer}_{n}
```

For example, neuron 5 in ReLU layer 2 of N2(x) (abs_layer = K+2):
```
n2_orga_layerCount2_neuronCount5_{K+2}_5
```

### A.3 `define_conf!` Function

The confidence margin variable is constructed by `define_conf!(m, d, c, key, name)` which:
1. Creates variable `conf` and auxiliary `max_kk`.
2. Adds: `conf = d[key][c] − max_kk`.
3. For each j ≠ c: adds binary `a_j`, constraint `max_kk ≥ d[key][j]`, constraint `max_kk ≤ d[key][j] + 10⁶ · (1 − a_j)`.
4. Adds: `∑_{j≠c} a_j = 1`.
5. Returns `conf`.

This is invoked for all four confidence margins (`conf_n1_x`, `conf_n2_x`, `conf_n1_xp`, `conf_n2_xp`) regardless of `n1_p_mode`. When `n1_p_mode = false`, `conf_n1_xp` is computed in the MIP but does not appear in constraint (C6).

### A.4 Command-Line Interface

```bash
julia run.jl \
  --mode transfer \
  --dataset mnist \
  --model_name 4x10 \
  --model_path /path/to/n1_weights.p \
  --model_path2 /path/to/n2_weights.p \
  --vaghar_results /path/to/vaghar_n1_results.txt \
  --perturbation linf \
  --perturbation_size 0.05 \
  --ctag 1 \
  --ct "2,3,4,5,6,7,8,9" \
  --c_tag_mode true \
  --n1_p_mode true \
  --timout 4000 \
  --output_dir ./results/ \
  --name_to_save transfer_exp1 \
  --use_hyper_attack true
```
