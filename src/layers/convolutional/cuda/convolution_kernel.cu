#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16
#define BLOCK_SIZE 16

__global__ void conv2d_forward_kernel(
    const float* input,
    const float* weights,
    float* output,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int kernel_size,
    int stride,
    int padding
) {
    // Shared memory for input and weight tiles
    __shared__ float input_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float weight_tile[TILE_SIZE][TILE_SIZE];
    
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    int batch = blockIdx.z;
    int out_channel = blockIdx.y;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;
    int out_col = threadIdx.y;
    
    if (out_row >= output_height || out_col >= output_width) return;
    
    float sum = 0.0f;
    
    // Loop over input channels
    for (int in_channel = 0; in_channel < input_channels; in_channel++) {
        // Loop over kernel rows
        for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
            int in_row = out_row * stride + kernel_row - padding;
            
            // Loop over kernel columns
            for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                int in_col = out_col * stride + kernel_col - padding;
                
                if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width) {
                    int input_idx = ((batch * input_channels + in_channel) * input_height + in_row) * input_width + in_col;
                    int weight_idx = ((out_channel * input_channels + in_channel) * kernel_size + kernel_row) * kernel_size + kernel_col;
                    
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    int output_idx = ((batch * output_channels + out_channel) * output_height + out_row) * output_width + out_col;
    output[output_idx] = sum;
}

__global__ void conv2d_backward_kernel(
    const float* input,
    const float* dvalues,
    float* dweights,
    float* dinputs,
    int batch_size,
    int input_channels,
    int input_height,
    int input_width,
    int output_channels,
    int kernel_size,
    int stride,
    int padding
) {
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    int batch = blockIdx.z;
    int out_channel = blockIdx.y;
    int in_channel = blockIdx.x;
    int kernel_row = threadIdx.x;
    int kernel_col = threadIdx.y;
    
    float dw = 0.0f;
    
    // Compute gradient for weights
    for (int out_row = 0; out_row < output_height; out_row++) {
        for (int out_col = 0; out_col < output_width; out_col++) {
            int in_row = out_row * stride + kernel_row - padding;
            int in_col = out_col * stride + kernel_col - padding;
            
            if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width) {
                int input_idx = ((batch * input_channels + in_channel) * input_height + in_row) * input_width + in_col;
                int dvalues_idx = ((batch * output_channels + out_channel) * output_height + out_row) * output_width + out_col;
                
                dw += input[input_idx] * dvalues[dvalues_idx];
            }
        }
    }
    
    int weight_idx = ((out_channel * input_channels + in_channel) * kernel_size + kernel_row) * kernel_size + kernel_col;
    atomicAdd(&dweights[weight_idx], dw);
    
    // Compute gradient for inputs
    for (int out_row = 0; out_row < output_height; out_row++) {
        for (int out_col = 0; out_col < output_width; out_col++) {
            int in_row = out_row * stride + kernel_row - padding;
            int in_col = out_col * stride + kernel_col - padding;
            
            if (in_row >= 0 && in_row < input_height && in_col >= 0 && in_col < input_width) {
                int input_idx = ((batch * input_channels + in_channel) * input_height + in_row) * input_width + in_col;
                int dvalues_idx = ((batch * output_channels + out_channel) * output_height + out_row) * output_width + out_col;
                
                atomicAdd(&dinputs[input_idx], weights[weight_idx] * dvalues[dvalues_idx]);
            }
        }
    }
} 