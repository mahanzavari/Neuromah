#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

__global__ void dropout_forward_kernel(
    const float* input,
    float* output,
    int* mask,
    int size,
    float rate,
    curandState* states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    curandState local_state = states[idx];
    float random = curand_uniform(&local_state);
    states[idx] = local_state;
    
    if (random > rate) {
        output[idx] = input[idx] / (1.0f - rate);
        mask[idx] = 1;
    } else {
        output[idx] = 0.0f;
        mask[idx] = 0;
    }
}

__global__ void dropout_backward_kernel(
    const float* dvalues,
    const int* mask,
    float* dinputs,
    int size,
    float rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    dinputs[idx] = dvalues[idx] * mask[idx] / (1.0f - rate);
}

__global__ void init_curand_states_kernel(
    curandState* states,
    unsigned long long seed,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    curand_init(seed, idx, 0, &states[idx]);
} 