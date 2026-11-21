#include "common.cuh"

#define CUDA_CLAMP_BLOCK_SIZE 256

void ggml_cuda_op_clamp(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_clip_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst, float eps, float limit);
