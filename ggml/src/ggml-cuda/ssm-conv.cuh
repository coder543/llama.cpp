#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);
void ggml_cuda_op_ssm_conv_step(ggml_backend_cuda_context & ctx, const ggml_tensor * window, const ggml_tensor * conv,
        const ggml_tensor * residual, ggml_tensor * out, ggml_tensor * state);
