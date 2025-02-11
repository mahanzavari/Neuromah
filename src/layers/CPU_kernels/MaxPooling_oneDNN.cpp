#include <oneapi/dnnl/dnnl.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;
namespace dnnl = dnnl;

py::array_t<float> maxpool2d_forward_onednn(py::array_t<float> input,
                                            int pool_h, int pool_w,
                                            int stride_h, int stride_w,
                                            std::string padding) {
    auto buf_input = input.request();
    if (buf_input.ndim != 4)
        throw std::runtime_error("Input must be a 4D tensor");

    int batch = buf_input.shape[0];
    int channels = buf_input.shape[1];
    int in_h = buf_input.shape[2];
    int in_w = buf_input.shape[3];

    dnnl::engine eng(dnnl::engine::kind::cpu, 0);
    dnnl::stream strm(eng);

    dnnl::memory::dims src_dims = {batch, channels, in_h, in_w};
    dnnl::memory::dims kernel = {pool_h, pool_w};
    dnnl::memory::dims strides = {stride_h, stride_w};
    dnnl::memory::dims padding_l, padding_r;
    
    if (padding == "same") {
        int out_h = (in_h + stride_h - 1) / stride_h;
        int out_w = (in_w + stride_w - 1) / stride_w;
        padding_l = {(out_h - 1) * stride_h + pool_h - in_h, (out_w - 1) * stride_w + pool_w - in_w};
        padding_r = padding_l;
    } else {
        padding_l = {0, 0};
        padding_r = {0, 0};
    }

    dnnl::memory::dims dst_dims = {batch, channels,
                                   (in_h - pool_h + 2 * padding_l[0]) / stride_h + 1,
                                   (in_w - pool_w + 2 * padding_l[1]) / stride_w + 1};

    auto src_mem = dnnl::memory({{src_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw}, eng, buf_input.ptr);
    auto dst_mem = dnnl::memory({{dst_dims}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw}, eng);

    auto pool_desc = dnnl::pooling_forward::desc(dnnl::prop_kind::forward_training,
                                                 dnnl::algorithm::pooling_max,
                                                 src_mem.get_desc(), dst_mem.get_desc(),
                                                 strides, kernel, padding_l, padding_r);

    auto pool_prim_desc = dnnl::pooling_forward::primitive_desc(pool_desc, eng);
    auto pool = dnnl::pooling_forward(pool_prim_desc);
    
    std::unordered_map<int, dnnl::memory> args = {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_DST, dst_mem}};
    pool.execute(strm, args);
    strm.wait();
    
    py::array_t<float> out_array({batch, channels, dst_dims[2], dst_dims[3]});
    auto buf_out = out_array.request();
    std::memcpy(buf_out.ptr, dst_mem.get_data_handle(), sizeof(float) * batch * channels * dst_dims[2] * dst_dims[3]);
    return out_array;
}

PYBIND11_MODULE(_MaxPooling2DBackend_onednn, m) {
    m.def("maxpool2d_forward_onednn", &maxpool2d_forward_onednn,
          "Max pooling forward pass using oneDNN.");
}
