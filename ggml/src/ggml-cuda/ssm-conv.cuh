#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

void ggml_cuda_ling_conv(ggml_backend_cuda_context & ctx, const ggml_tensor * state, const ggml_tensor * projection,
                         const ggml_tensor * weights, ggml_tensor * out, ggml_tensor * next_state, bool norm, float eps);
