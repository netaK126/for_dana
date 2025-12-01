import numpy as np

# Suppress scientific notation
np.set_printoptions(suppress=True)

# Generate randomly distributed parameters
params = np.random.uniform(low=-50, high=150, size=10000)

# Introduce an outlier
params[-1] = 1000

# Round each number to the second decimal place
params = np.round(params, 2)

# Print the parameters
print(params)

def clamp(params_q: np.array, lower_bound: int, upper_bound: int) -> np.array:
    params_q[params_q < lower_bound] = lower_bound
    params_q[params_q > upper_bound] = upper_bound
    return params_q

def asymmetric_quantization(params: np.array, bits: int) -> tuple[np.array, float, int]:
    alpha = np.max(params)
    beta = np.min(params)
    scale = (alpha - beta) / (2**bits-1)
    zero = -1*np.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits-1
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero

def asymmetric_quantization_percentile(params: np.array, bits: int, percentile: float = 99.99) -> tuple[np.array, float, int]:
    # find the percentile value
    alpha = np.percentile(params, percentile)
    beta = np.percentile(params, 100-percentile)
    scale = (alpha - beta) / (2**bits-1)
    zero = -1*np.round(beta / scale)
    lower_bound, upper_bound = 0, 2**bits-1
    quantized = clamp(np.round(params / scale + zero), lower_bound, upper_bound).astype(np.int32)
    return quantized, scale, zero


def asymmetric_dequantize(params_q: np.array, scale: float, zero: int) -> np.array:
    return (params_q - zero) * scale

def quantization_error(params: np.array, params_q: np.array):
    # calculate the MSE
    return np.mean((params - params_q)**2)

(asymmetric_q, asymmetric_scale, asymmetric_zero) = asymmetric_quantization(params, 8)
(asymmetric_q_percentile, asymmetric_scale_percentile, asymmetric_zero_percentile) = asymmetric_quantization_percentile(params, 8)

print(f'Original:')
print(np.round(params, 2))
print('')
print(f'Asymmetric (min-max) scale: {asymmetric_scale}, zero: {asymmetric_zero}')
print(asymmetric_q)
print(f'')
print(f'Asymmetric (percentile) scale: {asymmetric_scale_percentile}, zero: {asymmetric_zero_percentile}')
print(asymmetric_q_percentile)


# Dequantize the parameters back to 32 bits
params_deq_asymmetric = asymmetric_dequantize(asymmetric_q, asymmetric_scale, asymmetric_zero)
params_deq_asymmetric_percentile = asymmetric_dequantize(asymmetric_q_percentile, asymmetric_scale_percentile, asymmetric_zero_percentile)

print(f'Original:')
print(np.round(params, 2))
print('')
print(f'Dequantized (min-max):')
print(np.round(params_deq_asymmetric,2))
print('')
print(f'Dequantized (percentile):')
print(np.round(params_deq_asymmetric_percentile,2))

# Calculate the quantization error
print(f'{"Error (min-max) excluding outlier: ":>40}{np.round(quantization_error(params[:-1], params_deq_asymmetric[:-1]),2)}')
print(f'{"Error (percentile) excluding outlier: ":>40}{np.round(quantization_error(params[:-1], params_deq_asymmetric_percentile[:-1]),2)}')