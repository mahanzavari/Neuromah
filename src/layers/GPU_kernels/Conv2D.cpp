#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace py = pybind11;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



__global__ void im2col_kernel(const float* __restrict__ input,
                              int channels, int height, int width,
                              int kH, int kW,
                              int out_h, int out_w,
                              float* __restrict__ im2col)
{
    int total = channels * kH * kW * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int tmp = idx;
        int col_index = tmp % (out_h * out_w);
        tmp /= (out_h * out_w);
        int kernel_index = tmp % (kH * kW);
        int channel = tmp / (kH * kW);

        int out_row = col_index / out_w;
        int out_col = col_index % out_w;

        int p = kernel_index / kW;
        int q = kernel_index % kW;

        int in_row = out_row + p;
        int in_col = out_col + q;

        int input_index = channel * height * width + in_row * width + in_col;
        int im2col_index = (channel * kH * kW + kernel_index) * (out_h * out_w) + col_index;
        im2col[im2col_index] = input[input_index];
    }
}

py::array_t<float> conv2d_cpu(py::array_t<float> input, py::array_t<float> kernel) {
    auto buf_input = input.request(), buf_kernel = kernel.request();

    if (buf_input.ndim != 4 || buf_kernel.ndim != 4)
        throw std::runtime_error("Input and kernel must be 4D tensors");

    int batch        = buf_input.shape[0];
    int in_channels  = buf_input.shape[1];
    int in_h         = buf_input.shape[2];
    int in_w         = buf_input.shape[3];

    int out_channels = buf_kernel.shape[0];
    int kernel_ic    = buf_kernel.shape[1];
    int kH           = buf_kernel.shape[2];
    int kW           = buf_kernel.shape[3];

    if (in_channels != kernel_ic)
        throw std::runtime_error("Mismatch between input channels and kernel channels");

    int out_h = in_h - kH + 1;
    int out_w = in_w - kW + 1;
    int K = in_channels * kH * kW;    // common dimension (im2col rows)
    int N = out_h * out_w;            // im2col columns

    std::vector<float> output_host(batch * out_channels * out_h * out_w);

    float* h_input = static_cast<float*>(buf_input.ptr);
    float* h_kernel = static_cast<float*>(buf_kernel.ptr);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float* d_kernel;
    size_t kernel_size = out_channels * in_channels * kH * kW * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice));

    for (int b = 0; b < batch; b++) {
        float* h_img = h_input + b * in_channels * in_h * in_w;

        float* d_input;
        size_t input_size = in_channels * in_h * in_w * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_input, input_size));
        CUDA_CHECK(cudaMemcpy(d_input, h_img, input_size, cudaMemcpyHostToDevice));

        float* d_im2col;
        size_t im2col_size = K * N * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_im2col, im2col_size));

        int total_threads = K * N;
        int blockSize = 256;
        int numBlocks = (total_threads + blockSize - 1) / blockSize;
        im2col_kernel<<<numBlocks, blockSize>>>(d_input, in_channels, in_h, in_w,
                                                kH, kW, out_h, out_w,
                                                d_im2col);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        float* d_output;
        size_t output_size = out_channels * N * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_output, output_size));
        CUDA_CHECK(cudaMemset(d_output, 0, output_size));
        int M = out_channels;  // rows of the intended output.
        float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t stat = cublasSgemm(handle,
                                          CUBLAS_OP_T, CUBLAS_OP_T,
                                          N, M, K,
                                          &alpha,
                                          d_im2col, N,
                                          d_kernel, K,
                                          &beta,
                                          d_output, N);
        if (stat != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS GEMM failed");

        std::vector<float> output_image(out_channels * N);
        CUDA_CHECK(cudaMemcpy(output_image.data(), d_output, output_size, cudaMemcpyDeviceToHost));

        std::copy(output_image.begin(), output_image.end(),
                  output_host.begin() + b * out_channels * N);

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_im2col));
        CUDA_CHECK(cudaFree(d_output));
    }

    CUDA_CHECK(cudaFree(d_kernel));
    cublasDestroy(handle);

    py::array_t<float> result({batch, out_channels, out_h, out_w});
    auto buf_result = result.request();
    std::memcpy(buf_result.ptr, output_host.data(), output_host.size() * sizeof(float));
    return result;
}

py::tuple conv2d_backward_cpu(py::array_t<float> input,
                                py::array_t<float> kernel,
                                py::array_t<float> doutput) {
    throw std::runtime_error("conv2d_backward_cpu is not implemented for GPU acceleration");
}

PYBIND11_MODULE(_Conv2DBackend_cpp, m) {
    m.def("conv2d_cpu", &conv2d_cpu, "GPU-accelerated 2D convolution using im2col and cuBLAS");
    m.def("conv2d_backward_cpu", &conv2d_backward_cpu, "GPU-accelerated backward pass for 2D convolution (not implemented)");
}
