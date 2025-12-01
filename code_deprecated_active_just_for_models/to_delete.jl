using JuMP
using Gurobi

"One fully-connected quantized layer as seen by the MILP model."
struct QuantizedLayer
    Wq::Matrix{Int}        # Quantized integer weights w_q[m][k, j]
    b::Vector{Float64}     # Floating-point biases b[m][k]
    s_act::Float64         # Activation scale s_m   (for this layer's output)
    s_w::Float64           # Weight scale s_{m,W}   (shared by all weights in this layer)
    lq::Vector{Float64}    # Lower bounds l_q[m][k] on ζ_m,k  (dequantized outputs)
    uq::Vector{Float64}    # Upper bounds u_q[m][k] on ζ_m,k
end

"""
    encode_quantized_network(model, layers, b_bits)

Encode a fully-connected quantized network N_q into a MILP model.

Inputs:
  - model  :: JuMP.Model      – the MILP model we add variables/constraints to
  - layers :: Vector{QuantizedLayer}
      layers[m] represents layer m (1-based Julia index, but math: layer 1..L)
  - b_bits :: Int
      number of bits for the activation quantizer (e.g. 8)

Returns:
  - x      :: Vector{VariableRef}          – input variables (x_1..x_d)
  - zeta   :: Vector{Vector{VariableRef}}  – dequantized activations ζ_ℓ,k
                                            zeta[1]   = layer 0 (input after quantizer)
                                            zeta[ℓ+1] = layer ℓ outputs, ℓ=1..L
  - zhat   :: Vector{Vector{VariableRef}}  – pre-activations ẑ_ℓ,k  for ℓ=1..L
"""
function encode_quantized_network(model::Model,
                                  layers::Vector{QuantizedLayer},
                                  b_bits::Int)

    # Number of layers (hidden + output)
    L = length(layers)

    # Input dimension: number of columns in the first layer's weight matrix
    # Wq[1] has size (n_out_1, n_in_1), and n_in_1 is the input dimension d.
    d = size(layers[1].Wq, 2)

    # --------------------------------------------------------------
    # 1. Define input variables x ∈ [0, 1]^d
    # --------------------------------------------------------------
    # x is the (normalized) input to the network.
    @variable(model, 0 <= x[1:d] <= 1)

    # --------------------------------------------------------------
    # 2. Allocate containers for activations and pre-activations
    # --------------------------------------------------------------
    # zeta[ℓ][k]  = dequantized activation at "layer index" ℓ-1 in math:
    #   ℓ = 1          → layer 0 (input after quantized representation)
    #   ℓ = 2..L+1     → layer 1..L outputs
    #
    # zhat[m][k] = pre-activation ẑ_m,k for layer m (m=1..L)
    zeta = Vector{Vector{VariableRef}}(undef, L + 1)
    zhat = Vector{Vector{VariableRef}}(undef, L)

    # --------------------------------------------------------------
    # 3. Layer 0: dequantized input ζ₀,k
    # --------------------------------------------------------------
    # The quantized network N_q does not use x directly.
    # Instead, the input is also quantized and then dequantized, giving ζ₀.
    #
    # The paper encodes: ζ₀,k ∈ x_k + [-s₀/2, s₀/2]
    # where s₀ is the activation scale of the first "input quantizer" (here we
    # reuse the activation scale of layer 1, layers[1].s_act, as s₀).
    #
    # This represents the rounding error when x is quantized then dequantized.
    #
    @variable(model, zeta0[1:d])
    zeta[1] = zeta0

    s0 = layers[1].s_act  # scale used for input quantization

    for k in 1:d
        # ζ₀,k - x_k ≤  s₀ / 2
        @constraint(model, zeta[1][k] - x[k] <=  s0 / 2)
        # ζ₀,k - x_k ≥ -s₀ / 2
        @constraint(model, zeta[1][k] - x[k] >= -s0 / 2)
    end

    # --------------------------------------------------------------
    # 4. Hidden + output layers (m = 1..L)
    # --------------------------------------------------------------
    for m in 1:L
        layer = layers[m]

        Wq   = layer.Wq   # integer weights
        bvec = layer.b    # float biases
        s_act = layer.s_act  # activation scale for this layer's output
        s_w   = layer.s_w    # weight scale for this layer

        lq = layer.lq    # lower bounds l_q[m,k] on ζ_m,k
        uq = layer.uq    # upper bounds u_q[m,k] on ζ_m,k

        n_out, n_in = size(Wq)  # n_out neurons, each with n_in inputs

        # ----------------------------------------------------------
        # 4.1 Variables for this layer:
        #     ẑ_m,k (pre-activation) and ζ_m,k (dequantized quantized-ReLU)
        # ----------------------------------------------------------
        @variable(model, zhat_m[1:n_out])   # ẑ_m,k
        @variable(model, zeta_m[1:n_out])   # ζ_m,k

        zhat[m]   = zhat_m
        zeta[m+1] = zeta_m    # remember: zeta index ℓ=m+1 ↔ layer m in math

        # ----------------------------------------------------------
        # 4.2 Affine pre-activation:
        #     ẑ_m,k = b_m,k + s_{m,W} * Σ_j ( w_q[m][k,j] * ζ_{m-1,j} )
        #
        # Intuition:
        #   - w_q[m][k,j] is an integer (e.g. -127..127 for int8)
        #   - s_{m,W} rescales it to floating-point
        #   - ζ_{m-1,j} is the dequantized output of previous layer
        # ----------------------------------------------------------
        for k in 1:n_out
            @constraint(model,
                zhat[m][k] ==
                    bvec[k] + s_w * sum(Wq[k, j] * zeta[m][j] for j in 1:n_in)
            )
        end

        # ----------------------------------------------------------
        # 4.3 Quantized ReLU with "clipped parallelogram" relaxation
        #
        # We want to encode approximately:
        #   r      = max(0, ẑ_m,k)                # ReLU
        #   q      = clamp( round(r / s_act), 0, 2^b_bits - 1 )   # int
        #   ζ_m,k  = s_act * q                   # dequantized output
        #
        # Exactly modeling floor+clamp is expensive in MILP, so the paper
        # uses a polyhedral relaxation with 2 booleans a_q, b_q per neuron.
        #
        # Define:
        #   u_c^m = s_act * (2^b_bits - 1)   (max possible dequantized value)
        #
        # and use precomputed bounds:
        #   l_q[m,k] <= ζ_m,k <= u_q[m,k]
        #
        # The constraints below implement Equation (3) in the paper:
        #   - "off" (ReLU inactive): ζ = 0
        #   - "on, inside grid": ζ follows a parallelogram around ẑ
        #   - "clipped at max": ζ = u_c^m
        # ----------------------------------------------------------
        uc = s_act * (2.0^b_bits - 1.0)  # u_c^m

        # Binary indicator variables:
        #   a_q = 1 means neuron is active (ReLU "on" region + clip region)
        #   b_q = 1 means we are in the "clipped to max" case
        @variable(model, a_q[1:n_out], Bin)
        @variable(model, b_q[1:n_out], Bin)

        for k in 1:n_out
            lk = lq[k]  # lower bound on ζ_m,k
            uk = uq[k]  # upper bound on ζ_m,k

            # ------------------------------
            # (1) a_q ≥ b_q
            # ------------------------------
            # If we are in the clipping case (b_q=1), neuron must be active (a_q=1).
            @constraint(model, a_q[k] >= b_q[k])

            # ------------------------------
            # (2) 0 ≤ ζ_m,k ≤ u_q * a_q
            # ------------------------------
            # - If a_q = 0 → ζ_m,k ≤ 0 (and ≥ 0 from below), so ζ_m,k = 0.
            # - If a_q = 1 → ζ_m,k ≤ u_q, so we use full upper bound.
            @constraint(model, zeta[m+1][k] >= 0)
            @constraint(model, zeta[m+1][k] <= uk * a_q[k])

            # ------------------------------
            # (3) ζ_m,k ≥ ẑ_m,k - s_act/2 - (u_q - u_c) * b_q
            # ------------------------------
            # This is the "left" slanted edge of the parallelogram:
            #   - If not clipping (b_q = 0):
            #       ζ ≥ ẑ - s/2
            #     which encodes rounding ±s/2 around ẑ.
            #   - If clipping (b_q = 1):
            #       ζ ≥ ẑ - s/2 - (u_q - u_c)
            #     pushing the line down so that at the top we can reach u_c.
            @constraint(model,
                zeta[m+1][k] >= zhat[m][k] - s_act / 2 - (uk - uc) * b_q[k]
            )

            # ------------------------------
            # (4) ζ_m,k ≤ ẑ_m,k + s_act/2 - l_q * (1 - a_q)
            # ------------------------------
            # This is the "right" slanted edge of the parallelogram:
            #   - If neuron is active (a_q = 1):
            #       ζ ≤ ẑ + s/2
            #   - If neuron is inactive (a_q = 0):
            #       ζ ≤ ẑ + s/2 - l_q
            #     which, combined with other constraints, drives ζ to zero.
            @constraint(model,
                zeta[m+1][k] <= zhat[m][k] + s_act / 2 - lk * (1 - a_q[k])
            )

            # ------------------------------
            # (5) ζ_m,k ≥ l_q
            # ------------------------------
            # Global lower bound on this neuron’s dequantized value.
            @constraint(model, zeta[m+1][k] >= lk)

            # ------------------------------
            # (6) ζ_m,k ≥ u_c * b_q
            # ------------------------------
            # If we are in clipping region (b_q = 1), force ζ ≥ u_c.
            # Together with ζ ≤ u_c below, this will give ζ = u_c.
            @constraint(model, zeta[m+1][k] >= uc * b_q[k])

            # ------------------------------
            # (7) ζ_m,k ≤ u_c
            # ------------------------------
            # Global maximum possible dequantized value (all bits = 1).
            @constraint(model, zeta[m+1][k] <= uc)
        end
    end

    return x, zeta, zhat
end

# Example of usage:
model = Model(Gurobi.Optimizer)

# You would fill `layers::Vector{QuantizedLayer}` using data exported from PyTorch.
# For example:
# layers = [
#     QuantizedLayer(Wq1, b1, s_act1, s_w1, lq1, uq1),
#     QuantizedLayer(Wq2, b2, s_act2, s_w2, lq2, uq2),
#     ...
# ]
#
# x, zeta, zhat = encode_quantized_network(model, layers, 8)
