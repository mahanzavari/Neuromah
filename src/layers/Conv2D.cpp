#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

// Simple Conv2D function (no optimizations yet)
py::array_t<float> conv2d(py::array_t<float> input, py::array_t<float> kernel) {
    auto buf_input = input.request(), buf_kernel = kernel.request();
    
    if (buf_input.ndim != 4 || buf_kernel.ndim != 4) {
        throw std::runtime_error("Input and kernel must be 4D tensors");
    }

    auto ptr_input = static_cast<float *>(buf_input.ptr);
    auto ptr_kernel = static_cast<float *>(buf_kernel.ptr);

    // Get dimensions
    int batch_size = buf_input.shape[0];
    int in_channels = buf_input.shape[1];
    int in_height = buf_input.shape[2];
    int in_width = buf_input.shape[3];

    int out_channels = buf_kernel.shape[0];
    int kernel_height = buf_kernel.shape[2];
    int kernel_width = buf_kernel.shape[3];

    int out_height = in_height - kernel_height + 1;
    int out_width = in_width - kernel_width + 1;

    std::vector<float> output(batch_size * out_channels * out_height * out_width, 0.0f);

    // Simple convolution (no stride, padding, etc.)
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_channels; oc++) {
            for (int oh = 0; oh < out_height; oh++) {
                for (int ow = 0; ow < out_width; ow++) {
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_channels; ic++) {
                        for (int kh = 0; kh < kernel_height; kh++) {
                            for (int kw = 0; kw < kernel_width; kw++) {
                                int ih = oh + kh;
                                int iw = ow + kw;
                                sum += ptr_input[b * in_channels * in_height * in_width +
                                                 ic * in_height * in_width +
                                                 ih * in_width +
                                                 iw] *
                                       ptr_kernel[oc * in_channels * kernel_height * kernel_width +
                                                  ic * kernel_height * kernel_width +
                                                  kh * kernel_width +
                                                  kw];
                            }
                        }
                    }
                    output[b * out_channels * out_height * out_width +
                           oc * out_height * out_width +
                           oh * out_width +
                           ow] = sum;
                }
            }
        }
    }

    // Convert vector to numpy array
    return py::array_t<float>({batch_size, out_channels, out_height, out_width}, output.data());
}

// Pybind11 module definition
PYBIND11_MODULE(conv2d_cpp, m) {
    m.def("conv2d", &conv2d, "2D Convolution (C++ implementation)");
}
