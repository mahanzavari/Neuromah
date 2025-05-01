#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE_SIZE 16

__global__ void max_pool2d_forward_kernel(
    const float* input,
    float* output,
    int* indices,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int pool_size,
    int stride
) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    
    int batch = blockIdx.z;
    int channel = blockIdx.y;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;
    int out_col = threadIdx.y;
    
    if (out_row >= output_height || out_col >= output_width) return;
    
    float max_val = -INFINITY;
    int max_idx = -1;
    
    for (int pool_row = 0; pool_row < pool_size; pool_row++) {
        for (int pool_col = 0; pool_col < pool_size; pool_col++) {
            int in_row = out_row * stride + pool_row;
            int in_col = out_col * stride + pool_col;
            
            int input_idx = ((batch * channels + channel) * input_height + in_row) * input_width + in_col;
            float val = input[input_idx];
            
            if (val > max_val) {
                max_val = val;
                max_idx = input_idx;
            }
        }
    }
    
    int output_idx = ((batch * channels + channel) * output_height + out_row) * output_width + out_col;
    output[output_idx] = max_val;
    indices[output_idx] = max_idx;
}

__global__ void max_pool2d_backward_kernel(
    const float* dvalues,
    const int* indices,
    float* dinputs,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width
) {
    int batch = blockIdx.z;
    int channel = blockIdx.y;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;
    int out_col = threadIdx.y;
    
    if (out_row >= output_height || out_col >= output_width) return;
    
    int output_idx = ((batch * channels + channel) * output_height + out_row) * output_width + out_col;
    int input_idx = indices[output_idx];
    
    atomicAdd(&dinputs[input_idx], dvalues[output_idx]);
}

__global__ void avg_pool2d_forward_kernel(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int pool_size,
    int stride
) {
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    
    int batch = blockIdx.z;
    int channel = blockIdx.y;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;
    int out_col = threadIdx.y;
    
    if (out_row >= output_height || out_col >= output_width) return;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int pool_row = 0; pool_row < pool_size; pool_row++) {
        for (int pool_col = 0; pool_col < pool_size; pool_col++) {
            int in_row = out_row * stride + pool_row;
            int in_col = out_col * stride + pool_col;
            
            if (in_row < input_height && in_col < input_width) {
                int input_idx = ((batch * channels + channel) * input_height + in_row) * input_width + in_col;
                sum += input[input_idx];
                count++;
            }
        }
    }
    
    int output_idx = ((batch * channels + channel) * output_height + out_row) * output_width + out_col;
    output[output_idx] = sum / count;
}

__global__ void avg_pool2d_backward_kernel(
    const float* dvalues,
    float* dinputs,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int pool_size,
    int stride
) {
    int batch = blockIdx.z;
    int channel = blockIdx.y;
    int out_row = blockIdx.x * blockDim.x + threadIdx.x;
    int out_col = threadIdx.y;
    
    if (out_row >= output_height || out_col >= output_width) return;
    
    float dvalue = dvalues[((batch * channels + channel) * output_height + out_row) * output_width + out_col];
    float avg_dvalue = dvalue / (pool_size * pool_size);
    
    for (int pool_row = 0; pool_row < pool_size; pool_row++) {
        for (int pool_col = 0; pool_col < pool_size; pool_col++) {
            int in_row = out_row * stride + pool_row;
            int in_col = out_col * stride + pool_col;
            
            if (in_row < input_height && in_col < input_width) {
                int input_idx = ((batch * channels + channel) * input_height + in_row) * input_width + in_col;
                atomicAdd(&dinputs[input_idx], avg_dvalue);
            }
        }
    }
} 