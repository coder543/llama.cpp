#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add_scale(ggml_backend_cuda_context & ctx, const ggml_tensor * add, ggml_tensor * dst);
