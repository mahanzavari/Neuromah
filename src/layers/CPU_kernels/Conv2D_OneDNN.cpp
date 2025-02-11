#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>
#include <stdexcept>
#include <unordered_map>

namespace py = pybind11;
using namespace dnnl;
src\include\oneDNN\include\oneapi\dnnl\dnnl.hpp
/*
 * conv2d_cpu:
 *
 * This function performs a forward 2D convolution using oneDNN.
 *
 * Parameters:
 *   - input: A 4D NumPy array with shape [N, C, H, W].
 *   - kernel: A 4D NumPy array with shape [OC, C, kH, kW].
 *   - stride_h: (Optional) The vertical stride (default = 1).
 *   - stride_w: (Optional) The horizontal stride (default = 1).
 *
 * Note:
 *   Because your Python wrapper already pads the input when using padding=='same',
 *   this function assumes that the input is ready for convolution with zero padding.
 *
 * Returns:
 *   A 4D NumPy array with shape [N, OC, out_H, out_W] computed as:
 *
 *       out_H = floor((H - kH) / stride_h) + 1,
 *       out_W = floor((W - kW) / stride_w) + 1.
 */

// Function to perform 2D conv          // forcecast == force casting to float
py::array_t<float> conv2d__forward_cpu(py::array_t<float, py::array::c_style | py::array::forcecast> input,
                              py::array_t<float, py::array::c_style | py::array::forcecast> kernel,
                              int stride_h = 1, int stride_w = 1)  {
          // py::buffer_info
          auto input_buf = input.request();
          auto kernel_buf = kernel.request();

          if (input_buf.ndim != 4 || kernel_buf.ndim != 4)
               throw std::runtime_error("Input and weights must be 4-dimentional");
          // input
          int N      = static_cast<int>(input_buf.shape[0]);
          int C      = static_cast<int>(input_buf.shape[1]);
          int H      = static_cast<int>(input_buf.shape[2]);
          int W      = static_cast<int>(input_buf.shape[3]);

          int OC     = static_cast<int>(kernel_buf.shape[0]);
          int kC     = static_cast<int>(kernel_buf.shape[1]);
          int kH     = static_cast<int>(kernel_buf.shape[2]);
          int kW     = static_cast<int>(kernel_buf.shape[3]);
          
          if (kC != C)
               throw std::runtime_error("The number of input channels in weights must match
          that of the input");

          // conv params
          int out_H = (H - kH) / stride_h + 1;
          int out_W = (W - kW) / stride_w + 1;

          // OneDNN
          engine eng(engine::kind::cpu , 0);
          stream eng_stream(eng);

          // defining memory dims
          memory::dims src_dims    = {N , C , H, W};
          memory::dims weight_dims = {OC , C , kH , kW};
          memory::dims dst_dims    = {N, OC , out_H , out_W};
          memory::dims strides     = {stride_h , stride_w};
          // zero pad cause inputs are padded(by python wrapper)
          memory::dims padding     = {0 , 0};

          // memory descriptors
          // We use the "nchw" format for input and output, and "oihw" for the kernel. (learn)
          auto src_md    = memory::desc(src_dims , memory::data_type::f32 , memory::format_tag::nchw);
          auto weight_md = memory::desc(weight_dims , memory::data_type::f32 , memory::format_tag::nchw);
          auto dst_md  = memory::desc(dst_dims , memory::data_type::f32 , memory::format_tag::nchw);

          // memo objects
          auto src_mem    = memory(src_md , eng , input_buf.ptr);
          auto weight_mem = memory(weight_md , eng , kernel_buf.ptr);

          // temp buffer for Conv output
          std::vector<float> output_data(N * OC * out_H * out_W , 0.0f);
          auto dst_mem = memory(dst_md , eng , output_data.data());

          // conv primitive - forward conv descriptor
          auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
                                                     algorithm::convolution_direct,
                                                     src_md , weight_md , dst_md,
                                                     strides , padding , padding);
          auto conv_pd  = convolution_forward::primitive_desc(conv_desc , eng);
          auto conv = convolution_forward(conv_pd);

          // exe
          std::unordered_map<int , memory> conv_args;
          conv_args.insert({DNNL_ARG_SRC , src_mem});
          conv_args.insert({DNNL_ARG_WEIGHTS , weights_mem});
          conv_args.insert({DNNL_ARG_DST , dst_mem});

          conv.execute(eng_stream , conv_args);
          eng_stream.wait();

          // wrap output into numpy array - allocated new array and copied data into it
          py::array_t<float> result({N , OC , out_H , out_W});
          auto buf_result = result.request();
          std::memcpy(buf_result.ptr , output_data.data() , sizeof(float) * output_data.size());
          return result; 
     }

/*
 * Backward Convolution:
 *
 * Computes the gradients with respect to the input (dinputs) and the kernel (dweights)
 * given the upstream gradient (doutput). The backward pass is computed using oneDNN’s
 * backward convolution primitives.
 *
 * Parameters:
 *   - input: 4D NumPy array with shape [N, C, H, W] (the original input).
 *   - kernel: 4D NumPy array with shape [OC, C, kH, kW] (the convolution kernels).
 *   - doutput: 4D NumPy array with shape [N, OC, out_H, out_W] (upstream gradients).
 *   - stride_h: Vertical stride (default = 1).
 *   - stride_w: Horizontal stride (default = 1).
 *
 * Returns:
 *   A tuple (dinputs, dweights) where:
 *     - dinputs is a 4D NumPy array with the same shape as input.
 *     - dweights is a 4D NumPy array with the same shape as kernel.
 */
py::array_t<float> conv2d_backward_cpu(py::array_t<float , py::array::c_style | py::array::forcecast> input,
                                       py::array_t<float , py::array::c_style | py::array::forcecast> kernel,
                                       py::array_t<float , py::array::c_style | py::array::forcecast> doutputs,
                                       int stride_h = 1 , int stride_w = 1){
          auto input_buf     = input.request();
          auto kernel_buf    = kernel.request();
          auto doutputs_buf  = doutputs.request();

          if (input_buf.ndim != 4 || kernel_buf.ndim != 4 || doutputs_buf.ndim != 4)
               throw std::runtime_error("All inputs must be 4D tensors.");

          int N = static_cast<int>(input_buf.shape[0]);
          int C = static_cast<int>(input_buf.shape[1]);
          int H = static_cast<int>(input_buf.shape[2]);
          int W = static_cast<int>(input_buf.shape[3]);

          int OC = static_cast<int>(kernel_buf.shape[0]);
          int kC = static_cast<int>(kernel_buf.shape[1]);
          int kH = static_cast<int>(kernel_buf.shape[2]);
          int kW = static_cast<int>(kernel_buf.shape[3]);

          if (kC != C)
               throw std::runtime_error("Kernel inputs channels must match input channels.");

          // doutput dims == [N , OC , out_H , out_W]
          int out_H = static_cast<int>(doutputs.shape[2]);
          int out_W = static_cast<int>(doutputs.shape[3]);

          engine eng(engine::kind::cpu , 0);
          stream eng_stream(eng);

          // memory dims
          memory::dims src_dims   = {N , C , H , W};
          memory::dims weight_dims = {OC, C, kH, kW};
          memory::dims dst_dims    = {N, OC, out_H, out_W};
          memory::dims strides     = {stride_h, stride_w};
          memory::dims padding     = {0, 0};

          // memo decs
          auto src_md    = memory::desc(src_dims , memory::data_type::f32 , memory::format_tag::nchw);
          auto weight_md = memory::desc(weight_dims , memory::data_type::f32 , memory::format_tag::nchw);
          auto dst_md    = memory::desc(dst_dims , memory::data_type::f32 , memory::format_tag::nchw);
          // forward conv desc as a hint for backward primitives 
          auto conv_desc_forward = convolution_forward::desc(prop_kind::forward_training,
                                                             algorithm::convolution_direct,
                                                             src_md , weight_md , dst_md,
                                                             strides , padding , padding);
          auto conv_pd_fwd = convolution_foward::primitive_desc(conv_desc_forward , eng);
          // backward primitive desc
          auto conv_bwd_data_desc = convolution_backward_data::desc(src_md, weight_md, dst_md,
                                                                strides, padding, padding);
          auto conv_pd_bwd_data = convolution_backward_data::primitive_desc(conv_bwd_data_desc, eng, conv_pd_fwd);

          // backward weights primitive descriptor 
          auto conv_bwd_weights_desc = convolution_backward_weights::desc(src_md, weight_md, dst_md,
                                                                            strides, padding, padding);
          auto conv_pd_bwd_weights = convolution_backward_weights::primitive_desc(conv_bwd_weights_desc, eng, conv_pd_fwd);
          

          auto src_mem    = memory(src_md , eng , input_buf.data);
          auto weight_mem = memory(weight_md , eng , kernel_buf.data);
          auto doutputs_mem = memory(dst_md , eng , doutputs_buf.data);


          // Allocate buffers for gradients.
          // dinputs: gradient with respect to the input (same shape as input).
          std::vector<float> dinput_data(N * C * H * W, 0.0f);
          auto dinput_mem = memory(src_md, eng, dinput_data.data());
          
          // dweights: gradient with respect to the kernel (same shape as kernel).
          std::vector<float> dweights_data(OC * C * kH * kW, 0.0f);
          auto dweights_mem = memory(weight_md, eng, dweights_data.data());
          
          // --- Execute backward data primitive ---
          {
          std::unordered_map<int, memory> bwd_data_args;
          bwd_data_args.insert({DNNL_ARG_DIFF_DST, doutput_mem});
          bwd_data_args.insert({DNNL_ARG_WEIGHTS, weight_mem});
          bwd_data_args.insert({DNNL_ARG_DIFF_SRC, dinput_mem});
          auto conv_bwd_data = convolution_backward_data(conv_pd_bwd_data);
          conv_bwd_data.execute(eng_stream, bwd_data_args);
          }
          
          // --- Execute backward weights primitive ---
          {
          std::unordered_map<int, memory> bwd_weights_args;
          bwd_weights_args.insert({DNNL_ARG_SRC, src_mem});
          bwd_weights_args.insert({DNNL_ARG_DIFF_DST, doutput_mem});
          bwd_weights_args.insert({DNNL_ARG_WEIGHTS , dweights_mem});
          auto conv_bwd_weights = convolution_backward_weights(conv_pd_bwd_weights);
          conv_bwd_weights.execute(eng_stream, bwd_weights_args);
          }
          
          eng_stream.wait();
          
          // --- Wrap the gradient buffers into NumPy arrays ---
          py::array_t<float> dinputs({N, C, H, W});
          py::array_t<float> dweights({OC, C, kH, kW});
          
          auto buf_dinputs  = dinputs.request();
          auto buf_dweights = dweights.request();
          
          std::memcpy(buf_dinputs.ptr, dinput_data.data(), sizeof(float) * dinput_data.size());
          std::memcpy(buf_dweights.ptr, dweights_data.data(), sizeof(float) * dweights_data.size());
          
          return py::make_tuple(dinputs, dweights);     
     }

PYBIND11_MODULE(_Conv2DBackend_oneDNN_cpu, m) {
     m.doc() = "oneDNN-based convolution implementation with forward and backward passes";
     m.def("conv2d_forward_onednn_cpu", &conv2d__forward_cpu,
          py::arg("input"),
          py::arg("kernel"),
          py::arg("stride_h") = 1,
          py::arg("stride_w") = 1,
          "2D convolution forward pass using oneDNN. The input should be pre-padded if needed.");
     m.def("conv2d_backward_onednn_cpu", &conv2d_backward_cpu,
          py::arg("input"),
          py::arg("kernel"),
          py::arg("doutput"),
          py::arg("stride_h") = 1,
          py::arg("stride_w") = 1,
          "2D convolution backward pass using oneDNN. Returns a tuple (dinputs, dweights).");
}
