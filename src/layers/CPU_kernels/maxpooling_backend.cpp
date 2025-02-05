#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>

namespace py = pybind11;

/*
 * maxpool2d_forward_cpu:
 *
 * Given an input tensor of shape [batch, channels, in_h, in_w],
 * pool window sizes pool_h, pool_w, strides stride_h, stride_w, and a
 * padding mode ("valid" or "same"), this function returns a tuple:
 *   (output, max_indices)
 *
 *  - output is a [batch, channels, out_h, out_w] tensor,
 *  - max_indices is an integer tensor of shape [batch, channels, out_h, out_w, 2]
 *    storing the (row, col) position (in the padded input) of the max value.
 */
py::tuple maxpool2d_forward_cpu(py::array_t<float> input,
                                int pool_h, int pool_w,
                                int stride_h, int stride_w,
                                std::string padding) {
    auto buf_input = input.request();
    if (buf_input.ndim != 4)
        throw std::runtime_error("Input must be a 4D tensor");

    int batch    = buf_input.shape[0];
    int channels = buf_input.shape[1];
    int in_h     = buf_input.shape[2];
    int in_w     = buf_input.shape[3];

    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    int out_h, out_w;
    if (padding == "same") {
        out_h = static_cast<int>(std::ceil((float)in_h / stride_h));
        out_w = static_cast<int>(std::ceil((float)in_w / stride_w));
        int pad_h = std::max((out_h - 1) * stride_h + pool_h - in_h, 0);
        int pad_w = std::max((out_w - 1) * stride_w + pool_w - in_w, 0);
        pad_top    = pad_h / 2;
        pad_bottom = pad_h - pad_top;
        pad_left   = pad_w / 2;
        pad_right  = pad_w - pad_left;
    } else { // valid
        out_h = (in_h - pool_h) / stride_h + 1;
        out_w = (in_w - pool_w) / stride_w + 1;
    }

    // If padding=="same", create a padded copy of the input.
    std::vector<float> padded;
    int padded_h = in_h, padded_w = in_w;
    if (padding == "same") {
        padded_h = in_h + pad_top + pad_bottom;
        padded_w = in_w + pad_left + pad_right;
        padded.resize(batch * channels * padded_h * padded_w, 0.0f);
        float* ptr_input = static_cast<float*>(buf_input.ptr);
        // For each image and channel, copy into the padded location.
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < in_h; i++) {
                    std::memcpy(&padded[b * channels * padded_h * padded_w +
                                         c * padded_h * padded_w +
                                         (i + pad_top) * padded_w + pad_left],
                                &ptr_input[b * channels * in_h * in_w +
                                           c * in_h * in_w +
                                           i * in_w],
                                sizeof(float) * in_w);
                }
            }
        }
    }
    // Pointer to use for pooling: padded if needed, else original input.
    float* data_ptr = (padding == "same") ? padded.data() : static_cast<float*>(buf_input.ptr);
    int data_h = (padding == "same") ? padded_h : in_h;
    int data_w = (padding == "same") ? padded_w : in_w;

    // Allocate output and max_indices.
    std::vector<float> output(batch * channels * out_h * out_w, 0.0f);
    // max_indices shape: (batch, channels, out_h, out_w, 2)
    std::vector<int> max_indices(batch * channels * out_h * out_w * 2, 0);

    // Loop over each image/channel. Parallelize over batch and channel.
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int h_start = h * stride_h;
                    int w_start = w * stride_w;
                    int h_end = std::min(h_start + pool_h, data_h);
                    int w_end = std::min(w_start + pool_w, data_w);
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_i = h_start, max_j = w_start;
                    for (int i = h_start; i < h_end; i++) {
                        for (int j = w_start; j < w_end; j++) {
                            int index = b * channels * data_h * data_w +
                                        c * data_h * data_w +
                                        i * data_w + j;
                            float val = data_ptr[index];
                            if (val > max_val) {
                                max_val = val;
                                max_i = i;
                                max_j = j;
                            }
                        }
                    }
                    int out_index = b * channels * out_h * out_w +
                                    c * out_h * out_w +
                                    h * out_w + w;
                    output[out_index] = max_val;
                    // Save the (row, col) indices in the padded space.
                    int base = out_index * 2;
                    max_indices[base]     = max_i;
                    max_indices[base + 1] = max_j;
                }
            }
        }
    }

    // Create NumPy arrays to return.
    py::array_t<float> out_array({batch, channels, out_h, out_w});
    py::array_t<int> idx_array({batch, channels, out_h, out_w, 2});
    auto buf_out = out_array.request();
    std::memcpy(buf_out.ptr, output.data(), sizeof(float) * output.size());
    auto buf_idx = idx_array.request();
    std::memcpy(buf_idx.ptr, max_indices.data(), sizeof(int) * max_indices.size());

    return py::make_tuple(out_array, idx_array);
}

/*
 * maxpool2d_backward_cpu:
 *
 * Given the upstream gradients (dvalues) of shape [batch, channels, out_h, out_w],
 * the max_indices (from forward pass) of shape [batch, channels, out_h, out_w, 2],
 * and the original input (needed to determine the original spatial dimensions),
 * this function computes the gradient with respect to the input.
 *
 * If padding=="same", the dinputs are computed for the padded input and then unpadded.
 */
py::array_t<float> maxpool2d_backward_cpu(py::array_t<float> dvalues,
                                          py::array_t<int> max_indices,
                                          py::array_t<float> input,
                                          int pool_h, int pool_w,
                                          int stride_h, int stride_w,
                                          std::string padding) {
    auto buf_input   = input.request();
    auto buf_dvalues = dvalues.request();
    auto buf_idx     = max_indices.request();

    if (buf_input.ndim != 4)
        throw std::runtime_error("Input must be a 4D tensor");
    if (buf_dvalues.ndim != 4)
        throw std::runtime_error("dvalues must be a 4D tensor");
    if (buf_idx.ndim != 5)
        throw std::runtime_error("max_indices must be a 5D tensor");

    int batch    = buf_input.shape[0];
    int channels = buf_input.shape[1];
    int in_h     = buf_input.shape[2];
    int in_w     = buf_input.shape[3];
    int out_h    = buf_dvalues.shape[2];
    int out_w    = buf_dvalues.shape[3];

    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;
    int padded_h = in_h, padded_w = in_w;
    if (padding == "same") {
        int temp_out_h = static_cast<int>(std::ceil((float)in_h / stride_h));
        int temp_out_w = static_cast<int>(std::ceil((float)in_w / stride_w));
        int pad_h = std::max((temp_out_h - 1) * stride_h + pool_h - in_h, 0);
        int pad_w = std::max((temp_out_w - 1) * stride_w + pool_w - in_w, 0);
        pad_top    = pad_h / 2;
        pad_bottom = pad_h - pad_top;
        pad_left   = pad_w / 2;
        pad_right  = pad_w - pad_left;
        padded_h = in_h + pad_top + pad_bottom;
        padded_w = in_w + pad_left + pad_right;
    }

    // Allocate dinputs for the padded input shape.
    std::vector<float> dinputs(batch * channels * padded_h * padded_w, 0.0f);
    float* dvals_ptr = static_cast<float*>(buf_dvalues.ptr);
    int* idx_ptr = static_cast<int*>(buf_idx.ptr);

    // For every output gradient, add it to the location stored in max_indices.
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < out_h; h++) {
                for (int w = 0; w < out_w; w++) {
                    int out_index = b * channels * out_h * out_w +
                                    c * out_h * out_w +
                                    h * out_w + w;
                    float grad = dvals_ptr[out_index];
                    int base = out_index * 2;
                    int max_i = idx_ptr[base];
                    int max_j = idx_ptr[base + 1];
                    int index = b * channels * padded_h * padded_w +
                                c * padded_h * padded_w +
                                max_i * padded_w + max_j;
                    #pragma omp atomic
                    dinputs[index] += grad;
                }
            }
        }
    }

    // If padding=="same", remove the padding.
    if (padding == "same") {
        std::vector<float> unpadded(batch * channels * in_h * in_w, 0.0f);
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < in_h; i++) {
                    std::memcpy(&unpadded[b * channels * in_h * in_w +
                                         c * in_h * in_w +
                                         i * in_w],
                                &dinputs[b * channels * padded_h * padded_w +
                                         c * padded_h * padded_w +
                                         (i + pad_top) * padded_w + pad_left],
                                sizeof(float) * in_w);
                }
            }
        }
        py::array_t<float> result({batch, channels, in_h, in_w});
        auto buf_result = result.request();
        std::memcpy(buf_result.ptr, unpadded.data(), sizeof(float) * unpadded.size());
        return result;
    } else {
        py::array_t<float> result({batch, channels, in_h, in_w});
        auto buf_result = result.request();
        std::memcpy(buf_result.ptr, dinputs.data(), sizeof(float) * dinputs.size());
        return result;
    }
}

PYBIND11_MODULE(_MaxPooling2DBackend_cpp, m) {
    m.def("maxpool2d_forward_cpu", &maxpool2d_forward_cpu,
          "Max pooling forward pass on CPU.\n"
          "Arguments: input, pool_h, pool_w, stride_h, stride_w, padding");
    m.def("maxpool2d_backward_cpu", &maxpool2d_backward_cpu,
          "Max pooling backward pass on CPU.\n"
          "Arguments: dvalues, max_indices, input, pool_h, pool_w, stride_h, stride_w, padding");
}
