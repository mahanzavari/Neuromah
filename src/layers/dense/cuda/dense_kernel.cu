#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define TILE_SIZE 32

__global__ void dense_forward_kernel(
    const float* input,
    const float* weights,
    const float* biases,
    float* output,
    int batch_size,
    int input_size,
    int output_size
) {

     __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (input_size + TILE_SIZE - 1) / TILE_SIZE; tile++) {

        int input_col = tile * TILE_SIZE + threadIdx.x;
        if (row < batch_size && input_col < input_size) {
            input_tile[threadIdx.y][threadIdx.x] = input[row * input_size + input_col];
        } else {
            input_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        
        int weight_row = tile * TILE_SIZE + threadIdx.y;
        if (weight_row < input_size && col < output_size) {
            weight_tile[threadIdx.y][threadIdx.x] = weights[weight_row * output_size + col];
        } else {
            weight_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += input_tile[threadIdx.y][k] * weight_tile[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    
    if (row < batch_size && col < output_size) {
        output[row * output_size + col] = sum + biases[col];
    }
}


__global__ void dense_backward_weights_kernel(
    const float* input,
    const float* dvalues,
    float* dweights,
    int batch_size,
    int input_size,
    int output_size
) {
    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float dvalues_tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (batch_size + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        
        int input_row = tile * TILE_SIZE + threadIdx.y;
        if (input_row < batch_size && col < input_size) {
            input_tile[threadIdx.y][threadIdx.x] = input[input_row * input_size + col];
        } else {
            input_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        
        int dvalues_col = tile * TILE_SIZE + threadIdx.x;
        if (row < output_size && dvalues_col < batch_size) {
            dvalues_tile[threadIdx.y][threadIdx.x] = dvalues[dvalues_col * output_size + row];
        } else {
            dvalues_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += input_tile[threadIdx.y][k] * dvalues_tile[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    
    if (row < output_size && col < input_size) {
        dweights[row * input_size + col] = sum / batch_size;
    }
}


__global__ void dense_backward_input_kernel(
    const float* dvalues,
    const float* weights,
    float* dinputs,
    int batch_size,
    int input_size,
    int output_size
) {
    __shared__ float dvalues_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    
    for (int tile = 0; tile < (output_size + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        
        int dvalues_col = tile * TILE_SIZE + threadIdx.x;
        if (row < batch_size && dvalues_col < output_size) {
            dvalues_tile[threadIdx.y][threadIdx.x] = dvalues[row * output_size + dvalues_col];
        } else {
            dvalues_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        
        int weight_row = tile * TILE_SIZE + threadIdx.y;
        if (weight_row < output_size && col < input_size) {
            weight_tile[threadIdx.y][threadIdx.x] = weights[weight_row * input_size + col];
        } else {
            weight_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += dvalues_tile[threadIdx.y][k] * weight_tile[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    
    if (row < batch_size && col < input_size) {
        dinputs[row * input_size + col] = sum;
    }
}


__global__ void dense_backward_biases_kernel(
    const float* dvalues,
    float* dbiases,
    int batch_size,
    int output_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += dvalues[i * output_size + idx];
        }
        dbiases[idx] = sum / batch_size;
    }
} 