import cupy as cp
from numba import cuda

@cuda.jit
def sgd_update_kernel(
    weights: cp.ndarray,
    biases: cp.ndarray,
    dweights: cp.ndarray,
    dbiases: cp.ndarray,
    learning_rate: float
):
    """CUDA kernel for SGD parameter updates."""
    i, j = cuda.grid(2)
    if i < weights.shape[0] and j < weights.shape[1]:
        weights[i, j] -= learning_rate * dweights[i, j]
    if i < biases.shape[0]:
        biases[i] -= learning_rate * dbiases[i]

@cuda.jit
def momentum_update_kernel(
    weights: cp.ndarray,
    biases: cp.ndarray,
    dweights: cp.ndarray,
    dbiases: cp.ndarray,
    momentum_weights: cp.ndarray,
    momentum_biases: cp.ndarray,
    learning_rate: float,
    momentum: float
):
    """CUDA kernel for momentum parameter updates."""
    i, j = cuda.grid(2)
    if i < weights.shape[0] and j < weights.shape[1]:
        momentum_weights[i, j] = momentum * momentum_weights[i, j] - learning_rate * dweights[i, j]
        weights[i, j] += momentum_weights[i, j]
    if i < biases.shape[0]:
        momentum_biases[i] = momentum * momentum_biases[i] - learning_rate * dbiases[i]
        biases[i] += momentum_biases[i]

@cuda.jit
def rmsprop_update_kernel(
    weights: cp.ndarray,
    biases: cp.ndarray,
    dweights: cp.ndarray,
    dbiases: cp.ndarray,
    cache_weights: cp.ndarray,
    cache_biases: cp.ndarray,
    learning_rate: float,
    decay_rate: float,
    epsilon: float
):
    """CUDA kernel for RMSprop parameter updates."""
    i, j = cuda.grid(2)
    if i < weights.shape[0] and j < weights.shape[1]:
        cache_weights[i, j] = decay_rate * cache_weights[i, j] + (1 - decay_rate) * dweights[i, j] ** 2
        weights[i, j] -= learning_rate * dweights[i, j] / (cp.sqrt(cache_weights[i, j]) + epsilon)
    if i < biases.shape[0]:
        cache_biases[i] = decay_rate * cache_biases[i] + (1 - decay_rate) * dbiases[i] ** 2
        biases[i] -= learning_rate * dbiases[i] / (cp.sqrt(cache_biases[i]) + epsilon)

@cuda.jit
def adam_update_kernel(
    weights: cp.ndarray,
    biases: cp.ndarray,
    dweights: cp.ndarray,
    dbiases: cp.ndarray,
    m_weights: cp.ndarray,
    m_biases: cp.ndarray,
    v_weights: cp.ndarray,
    v_biases: cp.ndarray,
    learning_rate: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    t: int
):
    """CUDA kernel for Adam parameter updates."""
    i, j = cuda.grid(2)
    if i < weights.shape[0] and j < weights.shape[1]:
        # Update biased first moment estimate
        m_weights[i, j] = beta1 * m_weights[i, j] + (1 - beta1) * dweights[i, j]
        # Update biased second raw moment estimate
        v_weights[i, j] = beta2 * v_weights[i, j] + (1 - beta2) * dweights[i, j] ** 2
        # Compute bias-corrected first moment estimate
        m_hat = m_weights[i, j] / (1 - beta1 ** t)
        # Compute bias-corrected second raw moment estimate
        v_hat = v_weights[i, j] / (1 - beta2 ** t)
        # Update parameters
        weights[i, j] -= learning_rate * m_hat / (cp.sqrt(v_hat) + epsilon)
    if i < biases.shape[0]:
        # Update biased first moment estimate
        m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * dbiases[i]
        # Update biased second raw moment estimate
        v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * dbiases[i] ** 2
        # Compute bias-corrected first moment estimate
        m_hat = m_biases[i] / (1 - beta1 ** t)
        # Compute bias-corrected second raw moment estimate
        v_hat = v_biases[i] / (1 - beta2 ** t)
        # Update parameters
        biases[i] -= learning_rate * m_hat / (cp.sqrt(v_hat) + epsilon) 