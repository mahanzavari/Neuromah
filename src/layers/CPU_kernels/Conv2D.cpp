#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Core>
#include <omp.h>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace py = pybind11;

//=== Helper functions ========================================================

/*
 * im2col: Instead of allocating a new matrix every time, we accept a pointer
 * to an already allocated buffer for the im2col matrix.
 *
 * This function writes into im2colBuf which is assumed to have room for
 * (channels * kH * kW) * ( (height - kH + 1)*(width - kW + 1) ) floats.
 */
void im2col(const float* input, int channels, int height, int width,
            int kH, int kW, float* im2colBuf)
{
    int out_h = height - kH + 1;
    int out_w = width - kW + 1;
    int patch_size = channels * kH * kW;
    int col = 0;
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            int row = 0;
            for (int c = 0; c < channels; c++) {
                for (int p = 0; p < kH; p++) {
                    for (int q = 0; q < kW; q++) {
                        int in_row = i + p;
                        int in_col = j + q;
                        im2colBuf[row + col * patch_size] =
                            input[c * height * width + in_row * width + in_col];
                        row++;
                    }
                }
            }
            col++;
        }
    }
}

/*
 * col2im: Similar to im2col, writes directly into the provided dinput buffer.
 */
void col2im(const float* im2colBuf, int channels, int height, int width,
            int kH, int kW, float* dinput)
{
    int out_h = height - kH + 1;
    int out_w = width - kW + 1;
    int patch_size = channels * kH * kW;
    std::memset(dinput, 0, sizeof(float) * channels * height * width);
    
    int col = 0;
    for (int i = 0; i < out_h; i++) {
        for (int j = 0; j < out_w; j++) {
            int row = 0;
            for (int c = 0; c < channels; c++) {
                for (int p = 0; p < kH; p++) {
                    for (int q = 0; q < kW; q++) {
                        int in_row = i + p;
                        int in_col = j + q;
                        dinput[c * height * width + in_row * width + in_col] +=
                            im2colBuf[row + col * patch_size];
                        row++;
                    }
                }
            }
            col++;
        }
    }
}

//=== Optimized Forward Pass ==================================================

/*
 * conv2d_cpu: This function now reuses a temporary im2col buffer allocated in a
 * thread-local static vector. This avoids allocating a new matrix for every
 * sample if the dimensions do not change.
 */
py::array_t<float> conv2d_cpu(py::array_t<float> input, py::array_t<float> kernel) {
    auto buf_input = input.request(), buf_kernel = kernel.request();

    if (buf_input.ndim != 4 || buf_kernel.ndim != 4)
        throw std::runtime_error("Input and kernel must be 4D tensors");

    // Input dimensions.
    int batch_size   = buf_input.shape[0];
    int in_channels  = buf_input.shape[1];
    int in_height    = buf_input.shape[2];
    int in_width     = buf_input.shape[3];

    // Kernel dimensions.
    int out_channels = buf_kernel.shape[0];
    int kernel_ic    = buf_kernel.shape[1];
    int kH           = buf_kernel.shape[2];
    int kW           = buf_kernel.shape[3];

    if (in_channels != kernel_ic)
        throw std::runtime_error("Mismatch between input channels and kernel channels");

    int out_height = in_height - kH + 1;
    int out_width  = in_width - kW + 1;
    int im2col_rows = in_channels * kH * kW;
    int im2col_cols = out_height * out_width;

    // Pointers to data.
    float* ptr_input  = static_cast<float*>(buf_input.ptr);
    float* ptr_kernel = static_cast<float*>(buf_kernel.ptr);

    // Reshape kernel into a matrix of shape [out_channels, im2col_rows].
    Eigen::Map<Eigen::MatrixXf> kernelMat(ptr_kernel, out_channels, im2col_rows);

    // Allocate the output tensor (we use one large vector for the whole batch).
    std::vector<float> output_data(batch_size * out_channels * out_height * out_width, 0.0f);

    // Thread-local temporary buffer for im2col.
    // (Reallocated only if the size changes)
    thread_local std::vector<float> im2col_buffer;
    size_t im2col_size = im2col_rows * im2col_cols;
    if (im2col_buffer.size() < im2col_size)
        im2col_buffer.resize(im2col_size);

    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        const float* input_img = ptr_input + b * in_channels * in_height * in_width;
        // Fill the thread-local im2col buffer.
        im2col(input_img, in_channels, in_height, in_width, kH, kW, im2col_buffer.data());
        // Map the im2col buffer to an Eigen matrix (no extra allocation).
        Eigen::Map<Eigen::MatrixXf> im2colMat(im2col_buffer.data(), im2col_rows, im2col_cols);
        // Compute GEMM: kernelMat * im2colMat.
        Eigen::MatrixXf outMat = kernelMat * im2colMat;  // [out_channels x im2col_cols]
        // Write the output into the global output_data vector.
        float* output_ptr = output_data.data() + b * out_channels * out_height * out_width;
        std::memcpy(output_ptr, outMat.data(), sizeof(float) * out_channels * im2col_cols);
    }

    // Create a NumPy array that takes ownership of the output_data.
    // (Alternatively, you can avoid copying by creating a capsule that wraps the vector's memory.)
    py::array_t<float> result({batch_size, out_channels, out_height, out_width});
    auto buf_result = result.request();
    std::memcpy(buf_result.ptr, output_data.data(), sizeof(float) * output_data.size());
    return result;
}

//=== Optimized Backward Pass ================================================

py::tuple conv2d_backward_cpu(py::array_t<float> input,
                                py::array_t<float> kernel,
                                py::array_t<float> doutput) {
    auto buf_input   = input.request();
    auto buf_kernel  = kernel.request();
    auto buf_doutput = doutput.request();

    if (buf_input.ndim != 4 || buf_kernel.ndim != 4 || buf_doutput.ndim != 4)
        throw std::runtime_error("All inputs must be 4D tensors");

    // Input dimensions.
    int batch_size   = buf_input.shape[0];
    int in_channels  = buf_input.shape[1];
    int in_height    = buf_input.shape[2];
    int in_width     = buf_input.shape[3];

    // Kernel dimensions.
    int out_channels = buf_kernel.shape[0];
    int kernel_ic    = buf_kernel.shape[1];
    int kH           = buf_kernel.shape[2];
    int kW           = buf_kernel.shape[3];

    if (in_channels != kernel_ic)
        throw std::runtime_error("Mismatch between input channels and kernel channels");

    int out_height = buf_doutput.shape[2];
    int out_width  = buf_doutput.shape[3];
    int im2col_rows = in_channels * kH * kW;
    int im2col_cols = out_height * out_width;

    float* ptr_input   = static_cast<float*>(buf_input.ptr);
    float* ptr_kernel  = static_cast<float*>(buf_kernel.ptr);
    float* ptr_doutput = static_cast<float*>(buf_doutput.ptr);

    // Map kernel as matrix.
    Eigen::Map<Eigen::MatrixXf> kernelMat(ptr_kernel, out_channels, im2col_rows);

    // Allocate buffers for gradients.
    std::vector<float> dinputs_data(batch_size * in_channels * in_height * in_width, 0.0f);
    std::vector<float> dweights_data(out_channels * im2col_rows, 0.0f);
    Eigen::Map<Eigen::MatrixXf> dweightsMat(dweights_data.data(), out_channels, im2col_rows);
    dweightsMat.setZero();

    // Thread-local temporary im2col buffer for backward pass.
    thread_local std::vector<float> im2col_buffer;
    size_t im2col_size = im2col_rows * im2col_cols;
    if (im2col_buffer.size() < im2col_size)
        im2col_buffer.resize(im2col_size);

    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        const float* input_img = ptr_input + b * in_channels * in_height * in_width;
        const float* doutput_img = ptr_doutput + b * out_channels * out_height * out_width;
        float* dinput_img = dinputs_data.data() + b * in_channels * in_height * in_width;
        
        // Build im2col for this sample.
        im2col(input_img, in_channels, in_height, in_width, kH, kW, im2col_buffer.data());
        Eigen::Map<Eigen::MatrixXf> im2colMat(im2col_buffer.data(), im2col_rows, im2col_cols);
        Eigen::Map<const Eigen::MatrixXf> doutputMat(doutput_img, out_channels, im2col_cols);

        // Compute local dweights: dW += doutputMat * im2colMat^T.
        Eigen::MatrixXf local_dweights = doutputMat * im2colMat.transpose();

        #pragma omp critical
        {
            dweightsMat += local_dweights;
        }

        // Compute gradient with respect to input:
        // dx_col = kernelMat^T * doutputMat.
        Eigen::MatrixXf dx_col = kernelMat.transpose() * doutputMat;
        // Reconstruct dinput via col2im.
        col2im(dx_col.data(), in_channels, in_height, in_width, kH, kW, dinput_img);
    }

    // Prepare output arrays.
    py::array_t<float> dinputs_np({batch_size, in_channels, in_height, in_width});
    py::array_t<float> dweights_np({out_channels, in_channels, kH, kW});
    
    auto buf_dinputs_np  = dinputs_np.request();
    auto buf_dweights_np = dweights_np.request();
    
    std::memcpy(buf_dinputs_np.ptr,
                dinputs_data.data(),
                sizeof(float) * dinputs_data.size());
    std::memcpy(buf_dweights_np.ptr,
                dweights_data.data(),
                sizeof(float) * dweights_data.size());
    
    return py::make_tuple(dinputs_np, dweights_np);
}

PYBIND11_MODULE(_Conv2DBackend_cpp, m) {
    m.def("conv2d_cpu", &conv2d_cpu, "Optimized 2D Convolution using im2col/GEMM with reduced memory overhead");
    m.def("conv2d_backward_cpu", &conv2d_backward_cpu, "Optimized backward pass for 2D Convolution using im2col/col2im and GEMM with reduced memory overhead");
}
