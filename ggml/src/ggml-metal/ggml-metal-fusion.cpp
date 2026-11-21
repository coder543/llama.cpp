#include "ggml-metal-fusion.h"

#include "ggml-metal-common.h"
#include "ggml-metal-device.h"

#include "ggml-backend-impl.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <set>
#include <string>
#include <vector>

// derive the non-empty op sequence from the raw `ops_all` sequence
static std::vector<ggml_op> ggml_metal_fusion_filter_ops(const std::vector<ggml_op> & ops_all) {
    std::vector<ggml_op> ops;

    for (ggml_op op : ops_all) {
        if (!ggml_op_is_empty(op)) {
            ops.push_back(op);
        }
    }

    return ops;
}

struct ggml_metal_fusion {
    ggml_metal_fusion_id id;

    std::vector<ggml_op> ops;     // non-empty op sequence, derived from ops_all
    std::vector<ggml_op> ops_all; // full raw op sequence (may include empty RESHAPE/VIEW nodes)
    std::vector<int>     outs;    // additional fused output nodes, relative to ops

    // if unsafe: the generic chain/shape + ggml_can_fuse_subgraph checks are skipped and the
    // check callback below is the sole validator (used for patterns that are not elision chains,
    // e.g. the gdn + cache-cpy write-through fusion)
    bool unsafe;

    // extra backend constraints on top of ggml_can_fuse_subgraph
    // nodes[j] is the j-th node of the pattern; node_idxs[idx + j] is its raw graph index
    bool (*check)(const struct ggml_metal_fusion   * fusion,
                  const struct ggml_tensor * const * nodes,
                  const struct ggml_cgraph         * gf,
                  const int                        * node_idxs,
                        int                          idx,
                        ggml_metal_fusion_mode       mode);

    ggml_metal_fusion(
            ggml_metal_fusion_id id,
            const std::vector<ggml_op> & ops_all,
            const std::vector<int> & outs,
            bool unsafe,
            bool (*check)(const struct ggml_metal_fusion   * fusion,
                          const struct ggml_tensor * const * nodes,
                          const struct ggml_cgraph         * gf,
                          const int                        * node_idxs,
                                int                          idx,
                                ggml_metal_fusion_mode       mode))
        : id(id),
          ops(ggml_metal_fusion_filter_ops(ops_all)),
          ops_all(ops_all),
          outs(outs),
          unsafe(unsafe),
          check(check) {
    }
};

ggml_metal_fusion_id ggml_metal_fusion_get_id(const ggml_metal_fusion * fusion) {
    return fusion->id;
}

// ---- helpers -------------------------------------------------------------

// true if two tensors live in the same Metal buffer
static bool ggml_metal_fusion_same_buffer(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b) {
        return false;
    }

    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    ggml_metal_buffer_t ca = (ggml_metal_buffer_t) ba->context;
    ggml_metal_buffer_t cb = (ggml_metal_buffer_t) bb->context;

    return ggml_metal_buffer_get_id(ca, a).metal == ggml_metal_buffer_get_id(cb, b).metal;
}

// true if the memory of two tensors overlaps in the same Metal buffer
static bool ggml_metal_fusion_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    ggml_backend_buffer_t ba = a->view_src ? a->view_src->buffer : a->buffer;
    ggml_backend_buffer_t bb = b->view_src ? b->view_src->buffer : b->buffer;

    const ggml_metal_buffer_id bid_a = ggml_metal_buffer_get_id((ggml_metal_buffer_t) ba->context, a);
    const ggml_metal_buffer_id bid_b = ggml_metal_buffer_get_id((ggml_metal_buffer_t) bb->context, b);

    if (bid_a.metal == nullptr || bid_a.metal != bid_b.metal) {
        return false;
    }

    return bid_a.offs <= bid_b.offs
        ? bid_b.offs - bid_a.offs < ggml_nbytes(a)
        : bid_a.offs - bid_b.offs < ggml_nbytes(b);
}

// ---- pattern checks ------------------------------------------------------

// NORM/RMS_NORM + MUL + ADD: the weight/bias of each fused step must match the norm input
// width, be contiguous rows, and the fused outputs must stay F32
static bool ggml_metal_fusion_check_norm(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    GGML_ASSERT(fusion->ops.size() >= 2);

    if (fusion->id == GGML_METAL_FUSION_NORM_SCALE) {
        GGML_ASSERT(fusion->ops.size() == 2);

        const ggml_tensor * scale = nodes[1];
        if (scale->op != GGML_OP_SCALE || scale->src[0] != nodes[0] || scale->src[1] ||
            scale->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        // MUL/ADD are commutative, the previous node can be either operand
        const ggml_tensor * src = nodes[j]->src[0] == nodes[j - 1] ? nodes[j]->src[1] : nodes[j]->src[0];

        // the weight/bias must have the same row width as the norm input
        if (src->ne[0] != nodes[0]->ne[0]) {
            return false;
        }

        if (!ggml_is_contiguous_rows(src)) {
            return false;
        }

        if (nodes[j]->type != GGML_TYPE_F32) {
            return false;
        }
    }

    return true;
}

// SSM_CONV + UNARY (silu)
static bool ggml_metal_fusion_check_ssm_conv_silu(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(mode);

    const ggml_tensor * conv = nodes[0];
    const ggml_tensor * un   = nodes[1];

    if (conv->op != GGML_OP_SSM_CONV || un->op != GGML_OP_UNARY || un->src[0] != conv || un->src[1]) {
        return false;
    }

    if (ggml_get_unary_op(un) != GGML_UNARY_OP_SILU) {
        return false;
    }

    if (conv->type != GGML_TYPE_F32 || un->type != GGML_TYPE_F32 || !ggml_is_contiguous_rows(un)) {
        return false;
    }

    return true;
}

// ADD x N: each ADD reads the previous ADD as src0, and all addends must share layout
// (and, in FULL mode, live in the same Metal buffer)
static bool ggml_metal_fusion_check_add_chain(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_ASSERT(fusion->ops.size() >= 2);

    for (int j = 1; j < (int) fusion->ops.size(); j++) {
        if (nodes[j]->src[0] != nodes[j - 1]) {
            return false;
        }

        if (!ggml_are_same_layout(nodes[j]->src[1], nodes[j - 1]->src[1])) {
            return false;
        }

        if (mode == GGML_METAL_FUSION_FULL) {
            if (!ggml_metal_fusion_same_buffer(nodes[j]->src[1], nodes[0]->src[1])) {
                return false;
            }
        }
    }

    return true;
}

// GATED_DELTA_NET + CPY: the trailing cpy scatters the gdn state snapshots into the recurrent
// cache, so the gdn kernel writes them straight to the cache and the cpy is elided.
// mirrors ggml_metal_op_can_fuse_gdn_cache (PR #25788). the gdn output has other consumers (the
// attn scores view), so unlike the other patterns this is not an elision chain: the structural
// checks live entirely in this callback (unsafe = true).
static bool ggml_metal_fusion_check_gdn_cache(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * gdn = nodes[0];
    const ggml_tensor * cpy = nodes[1];

    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    if (cpy->op != GGML_OP_CPY || (cpy->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }

    const int64_t S_v      = gdn->src[2]->ne[0];
    const int64_t H        = gdn->src[2]->ne[1];
    const int64_t n_tokens = gdn->src[2]->ne[2];
    const int64_t n_seqs   = gdn->src[2]->ne[3];
    const int64_t K        = ggml_get_op_params_i32(gdn, 0);
    const size_t  tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    const int64_t D         = S_v * S_v * H;
    const int64_t n_written = std::min<int64_t>(n_tokens, K);

    const ggml_tensor * src = cpy->src[0]; // gdn snapshot tail view
    const ggml_tensor * dst = cpy->src[1]; // cache view

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn ||
        src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        return false;
    }

    const int64_t expected_ne[GGML_MAX_DIMS] = { D, n_seqs, n_written, 1 };
    if (dst->type != GGML_TYPE_F32 ||
        !std::equal(expected_ne, expected_ne + GGML_MAX_DIMS, dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        // the cache must be allocated so the kernel can write straight to its buffer
        if (dst->data == nullptr) {
            return false;
        }
    }

    return true;
}

// MUL + SIN + SQR + MUL + ADD (snake activation)
static bool ggml_metal_fusion_check_snake(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(fusion);
    GGML_UNUSED(mode);
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);

    const ggml_tensor * mul0     = nodes[0];
    const ggml_tensor * sin_node = nodes[1];
    const ggml_tensor * sqr      = nodes[2];
    const ggml_tensor * mul1     = nodes[3];
    const ggml_tensor * add      = nodes[4];

    // x carries the full activation shape, a is the broadcast operand
    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

    // mul1 reads sqr and inv_b in either operand order
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    // closure check: the trailing add reads the same x as the leading mul
    const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

    // x is in the supported whitelist and every chain intermediate shares x's type.
    // a and inv_b bind as device const float * in the kernel, so they stay F32.
    const bool types_ok =
        (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
        (a->type    == GGML_TYPE_F32) && (inv_b->type    == GGML_TYPE_F32) &&
        (mul0->type == x->type)       && (sin_node->type == x->type) &&
        (sqr->type  == x->type)       && (mul1->type     == x->type) &&
        (add->type  == x->type);

    // a / inv_b collapse to [1, C, 1, 1], x and add stay 2D
    const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];
    const bool dim_ok =
        (x->ne[2]     == 1) && (x->ne[3]     == 1) &&
        (add->ne[2]   == 1) && (add->ne[3]   == 1) &&
        (a->ne[2]     == 1) && (a->ne[3]     == 1) &&
        (inv_b->ne[2] == 1) && (inv_b->ne[3] == 1);

    // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
    const bool contig_ok =
        ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
        ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

    return types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x;
}

#define GGML_METAL_TOPK_MOE_MAX_EXPERTS 1024

static const std::vector<ggml_op> ops_topk_moe = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_norm_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_norm_scale_bias = {
    GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sp = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sp_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sp_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_SQRT, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static const std::vector<ggml_op> ops_topk_moe_sig = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS
};
static const std::vector<ggml_op> ops_topk_moe_sig_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE
};
static const std::vector<ggml_op> ops_topk_moe_sig_norm_scale_bias = {
    GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS,
    GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_SCALE, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE
};

static bool ggml_metal_fusion_check_topk_moe(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_ASSERT(fusion->ops.size() >= 3);
    GGML_UNUSED(nodes);

    const int n_ops = (int) fusion->ops.size();

    bool with_norm = false;
    for (int j = 0; j < n_ops; ++j) {
        with_norm |= fusion->ops[j] == GGML_OP_SUM_ROWS;
    }
    const bool with_scale = fusion->ops[n_ops - 1] == GGML_OP_SCALE;

    // the fusion table operates on the non-empty node sequence; the raw graph also
    // contains the RESHAPE/VIEW nodes that the fused kernel elides.
    const std::vector<ggml_op> & ops_all = fusion->ops_all;

    const int raw_start = node_idxs[idx];
    int       raw_end   = node_idxs[idx + n_ops - 1];

    // the norm variant ends with a RESHAPE that the non-empty sequence filters out;
    // include it so the output use-count check sees the real final routing tensor
    if (with_norm && !with_scale) {
        if (raw_end + 1 >= gf->n_nodes) {
            return false;
        }
        const ggml_tensor * trailing_reshape = gf->nodes[raw_end + 1];
        if (trailing_reshape->op != GGML_OP_RESHAPE || trailing_reshape->src[0] != gf->nodes[raw_end]) {
            return false;
        }
        raw_end++;
    }

    const int raw_count = raw_end - raw_start + 1;
    if (raw_count != (int) ops_all.size()) {
        return false;
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = raw_start + i;
        if (gf->nodes[raw_start + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * first = gf->nodes[raw_start];
    const ggml_tensor * out   = gf->nodes[raw_end];

    // the unary chains start at the sigmoid or the softplus node and rank biased scores
    const bool with_unary = first->op == GGML_OP_UNARY;
    const bool with_sp    = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SOFTPLUS;
    const bool with_sig   = with_unary && ggml_get_unary_op(first) == GGML_UNARY_OP_SIGMOID;

    if (with_unary && !with_sp && !with_sig) {
        return false;
    }

    const ggml_tensor * probs;          // the scoring output
    const ggml_tensor * scores;         // the node the argsort ranks (adds the bias for the unary chains)
    const ggml_tensor * probs_reshaped;
    const ggml_tensor * argsort;
    const ggml_tensor * ids;
    const ggml_tensor * get_rows;
    const ggml_tensor * logits = first->src[0];
    int ids_off;

    if (with_unary) {
        // the softplus chains add a SQRT node, the sigmoid chains start at the sigmoid node
        const int base = with_sp ? 1 : 0;

        probs          = gf->nodes[raw_start + base + 0];
        probs_reshaped = gf->nodes[raw_start + base + 1];
        scores         = gf->nodes[raw_start + base + 2];
        argsort        = gf->nodes[raw_start + base + 3];
        ids            = gf->nodes[raw_start + base + 4];
        get_rows       = gf->nodes[raw_start + base + 5];
        ids_off        = base + 4;

        if (scores->op != GGML_OP_ADD || scores->src[0] != probs ||
            (with_sp && (probs->op != GGML_OP_SQRT || probs->src[0] != first))) {
            return false;
        }

        // the selection bias is a per-expert vector shared by all rows
        const ggml_tensor * bias = scores->src[1];
        if (bias->type != GGML_TYPE_F32 || scores->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias) ||
            bias->ne[0] != probs->ne[0] || bias->ne[1] != 1 || bias->ne[2] != 1 || bias->ne[3] != 1) {
            return false;
        }
    } else {
        probs          = first;
        scores         = first;
        probs_reshaped = gf->nodes[raw_start + 1];
        argsort        = gf->nodes[raw_start + 2];
        ids            = gf->nodes[raw_start + 3];
        get_rows       = gf->nodes[raw_start + 4];
        ids_off        = 3;

        // the fused kernel implements plain softmax only
        float scale    = 1.0f;
        float max_bias = 0.0f;
        memcpy(&scale,    ((const int32_t *) first->op_params) + 0, sizeof(scale));
        memcpy(&max_bias, ((const int32_t *) first->op_params) + 1, sizeof(max_bias));
        if (scale != 1.0f || max_bias != 0.0f || first->src[1] || first->src[2]) {
            return false;
        }
    }

    if (logits->type != GGML_TYPE_F32 || probs->type != GGML_TYPE_F32 ||
        out->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_expert <= 0 || n_expert_used <= 0 || n_expert_used > n_expert ||
        n_expert > GGML_METAL_TOPK_MOE_MAX_EXPERTS || n_expert_used > GGML_METAL_TOPK_MOE_MAX_EXPERTS) {
        return false;
    }

    if (logits->ne[2] != 1 || logits->ne[3] != 1 ||
        ids->ne[1] != n_tokens || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        out->ne[0] != 1 || out->ne[1] != n_expert_used || out->ne[2] != n_tokens || out->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(logits) || !ggml_is_contiguous(out) ||
        ids->nb[0] != ggml_type_size(GGML_TYPE_I32) ||
        ids->nb[1] != ggml_type_size(GGML_TYPE_I32) * n_expert) {
        return false;
    }

    // the top-k gathers the unbiased scores, the argsort ranks the biased ones
    if (probs_reshaped->src[0] != probs || argsort->src[0] != scores ||
        ids->src[0] != argsort || get_rows->src[0] != probs_reshaped || get_rows->src[1] != ids) {
        return false;
    }

    if (with_norm) {
        const int norm_off = with_unary ? 6 + (with_sp ? 1 : 0) : 5;

        const ggml_tensor * weights_reshaped = gf->nodes[raw_start + norm_off + 0];
        const ggml_tensor * sum_rows         = gf->nodes[raw_start + norm_off + 1];
        const ggml_tensor * denom            = gf->nodes[raw_start + norm_off + 2];
        const ggml_tensor * div              = gf->nodes[raw_start + norm_off + 3];
        const ggml_tensor * out_reshaped     = gf->nodes[raw_start + norm_off + 4];

        if (weights_reshaped->src[0] != get_rows || sum_rows->src[0] != weights_reshaped ||
            denom->src[0] != sum_rows || div->src[0] != weights_reshaped || div->src[1] != denom ||
            out_reshaped->src[0] != div) {
            return false;
        }

        // the kernel applies the sum adjustment as a lower clamp or an additive bias, never a scale
        if (denom->op == GGML_OP_SCALE && ggml_get_op_params_f32(denom, 0) != 1.0f) {
            return false;
        }

        if (with_scale && out->src[0] != out_reshaped) {
            return false;
        }
    } else if (with_scale && out->src[0] != get_rows) {
        return false;
    }

    const int outputs[2] = { raw_start + ids_off, raw_end };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 2)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!logits->data || !out->data || !ids->data) {
            return false;
        }
    }

    return true;
}

#define GGML_METAL_MOE_REDUCE_MAX_EXPERTS 8

struct ggml_metal_moe_reduce_match {
    const ggml_tensor * experts;
    const ggml_tensor * weights;
    const ggml_tensor * dst;
    int node_count;
};

static bool ggml_metal_fusion_match_moe_reduce(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops_all,
        ggml_metal_moe_reduce_match * match) {
    if (match == nullptr || node_idx < 0 || node_idx + (int) ops_all.size() > gf->n_nodes) {
        return false;
    }

    const ggml_tensor * mul = gf->nodes[node_idx];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32) {
        return false;
    }

    // MUL, then one VIEW per expert, then one ADD per additional expert
    const int raw_count     = (int) ops_all.size();
    const int n_expert_used = raw_count / 2;

    if (n_expert_used < 2 || n_expert_used > GGML_METAL_MOE_REDUCE_MAX_EXPERTS ||
        raw_count != 2 * n_expert_used) {
        return false;
    }

    int n_views = 0;
    while (node_idx + 1 + n_views < gf->n_nodes &&
           gf->nodes[node_idx + 1 + n_views]->op == GGML_OP_VIEW) {
        n_views++;
    }

    if (n_views != n_expert_used) {
        return false;
    }

    for (int i = n_expert_used + 1; i < raw_count; ++i) {
        if (gf->nodes[node_idx + i]->op != GGML_OP_ADD) {
            return false;
        }
    }

    int raw_idxs[GGML_METAL_FUSION_MAX];
    for (int i = 0; i < raw_count; ++i) {
        raw_idxs[i] = node_idx + i;
        if (gf->nodes[node_idx + i]->op != ops_all[i]) {
            return false;
        }
    }

    const ggml_tensor * experts = mul->src[0];
    const ggml_tensor * weights = mul->src[1];
    const ggml_tensor * dst     = gf->nodes[node_idx + raw_count - 1];

    if (experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const int64_t n_embd   = experts->ne[0];
    const int64_t n_tokens = experts->ne[2];

    // note: n_tokens == 0 (no-output batch) must match so that the packing stays shape-independent
    if (n_embd <= 0 || experts->ne[1] != n_expert_used || experts->ne[3] != 1 ||
        weights->ne[0] != 1 || weights->ne[1] != n_expert_used || weights->ne[2] != n_tokens || weights->ne[3] != 1 ||
        dst->ne[0] != n_embd || dst->ne[1] != n_tokens || dst->ne[2] != 1 || dst->ne[3] != 1) {
        return false;
    }

    if (!ggml_is_contiguous(experts) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(dst)) {
        return false;
    }

    for (int i = 1; i <= n_expert_used; ++i) {
        const ggml_tensor * view = gf->nodes[node_idx + i];
        if (view->view_src != mul || view->src[0] != mul ||
            view->view_offs != (size_t) (i - 1) * mul->nb[1] ||
            view->ne[0] != n_embd || view->ne[1] != n_tokens ||
            view->nb[1] != mul->nb[2]) {
            return false;
        }
    }

    const ggml_tensor * prev_add = nullptr;
    for (int j = 1; j < n_expert_used; ++j) {
        const ggml_tensor * add = gf->nodes[node_idx + n_expert_used + j];
        const ggml_tensor * rhs = gf->nodes[node_idx + j + 1];
        const ggml_tensor * lhs = j == 1 ? gf->nodes[node_idx + 1] : prev_add;
        if (add->src[0] != lhs || add->src[1] != rhs) {
            return false;
        }
        prev_add = add;
    }

    const int outputs[1] = { node_idx + raw_count - 1 };
    if (!ggml_can_fuse_subgraph_ext(gf, raw_idxs, raw_count, ops_all.data(), outputs, 1)) {
        return false;
    }

    match->experts    = experts;
    match->weights    = weights;
    match->dst        = dst;
    match->node_count = raw_count;
    return true;
}

static bool ggml_metal_fusion_check_moe_reduce(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(nodes);

    ggml_metal_moe_reduce_match match;
    if (!ggml_metal_fusion_match_moe_reduce(gf, node_idxs[idx], fusion->ops_all, &match)) {
        return false;
    }

    if ((int) fusion->ops.size() != match.experts->ne[1]) {
        return false;
    }

    const int raw_end = node_idxs[idx] + match.node_count - 1;
    if (node_idxs[idx + (int) fusion->ops.size() - 1] != raw_end) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_FULL) {
        if (!match.experts->data || !match.weights->data || !match.dst->data) {
            return false;
        }

        // the fused kernel reads experts and weights while writing dst, so dst must not overlap either input
        if (ggml_metal_fusion_overlap(match.dst, match.experts) || ggml_metal_fusion_overlap(match.dst, match.weights)) {
            return false;
        }
    }

    return true;
}

// true if t is or views a tensor in a buffer marked as weights, such as a bias; the model loader marks its buffers before
// any graph is optimized, and tensors in unmarked or not yet allocated buffers count as non-weights in both phases
static bool ggml_metal_tensor_is_weight(const struct ggml_tensor * t) {
    const ggml_tensor * base = t->view_src != NULL ? t->view_src : t;

    return base->buffer != NULL && ggml_backend_buffer_get_usage(base->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_operand(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    if (add->op != GGML_OP_ADD || (add->src[0] == mm) == (add->src[1] == mm)) {
        return NULL;
    }

    const ggml_tensor * other = add->src[0] == mm ? add->src[1] : add->src[0];

    const bool ok = other->type == GGML_TYPE_F32 && add->type == GGML_TYPE_F32 && !ggml_metal_tensor_is_weight(other);

    return ok ? other : NULL;
}

static const struct ggml_tensor * ggml_metal_mul_mat_add_residual(const struct ggml_tensor * mm, const struct ggml_tensor * add) {
    const ggml_tensor * res = ggml_metal_mul_mat_add_operand(mm, add);

    const bool ok = res != NULL && ggml_are_same_shape(res, mm) &&
        ggml_is_contiguous(res) && ggml_is_contiguous(mm) && ggml_is_contiguous(add);

    return ok ? res : NULL;
}

// MUL_MAT + ADD of an f32 non-weight: the reorder packs it without reading row counts, so ubatch sizes share one order;
// the encoder fuses only a same-shape residual in the few-row MMA store, which the sum may overlap only in place
static bool ggml_metal_fusion_check_mul_mat_add(
        const ggml_metal_fusion       * fusion,
        const ggml_tensor * const     * nodes,
        const ggml_cgraph             * gf,
        const int                     * node_idxs,
              int                       idx,
              ggml_metal_fusion_mode    mode) {
    GGML_UNUSED(gf);
    GGML_UNUSED(node_idxs);
    GGML_UNUSED(idx);
    GGML_UNUSED(fusion);

    const ggml_tensor * mm  = nodes[0];
    const ggml_tensor * add = nodes[1];

    if (ggml_metal_mul_mat_add_operand(mm, add) == nullptr ||
        !ggml_metal_op_mul_mat_may_use_mma(mm)) {
        return false;
    }

    if (mode == GGML_METAL_FUSION_STRUCTURAL) {
        return true;
    }

    const ggml_tensor * res = ggml_metal_mul_mat_add_residual(mm, add);

    if (res == nullptr || !ggml_metal_op_mul_mat_use_mma(mm)) {
        return false;
    }

    return !ggml_metal_fusion_overlap(add, mm->src[0]) && !ggml_metal_fusion_overlap(add, mm->src[1]) &&
        (add->data == res->data || !ggml_metal_fusion_overlap(add, res));
}

static bool ggml_metal_fusion_check_ssm_conv_state(
        const ggml_metal_fusion *, const ggml_tensor * const * nodes,
        const ggml_cgraph * gf, const int * node_idxs, int idx, ggml_metal_fusion_mode mode) {
    const int i = node_idxs[idx];
    if (node_idxs[idx + 4] != i + 7) {
        return false;
    }
    // The destination view of CPY is an external input. Only the window and its tail are elided.
    const int members[] = { i, i+1, i+2, i+3, i+4, i+5, i+7 };
    const int outputs[] = { i+4, i+7 };
    static const ggml_op pattern[] = { GGML_OP_CONCAT, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD,
                                       GGML_OP_SCALE, GGML_OP_VIEW, GGML_OP_CPY };
    if (!ggml_can_fuse_subgraph_ext(gf, members, 7, pattern, outputs, 2)) {
        return false;
    }

    const auto * window = nodes[0];
    const auto * conv   = nodes[1];
    const auto * flat   = gf->nodes[i+2];
    const auto * add    = nodes[2];
    const auto * out    = nodes[3];
    const auto * tail   = gf->nodes[i+5];
    const auto * state  = nodes[4];
    const auto * past   = window->src[0];
    const auto * step   = window->src[1];
    const auto * w      = conv->src[1];
    const auto * cur    = add->src[0];
    if (ggml_get_op_params_i32(window, 0) != 0 || conv->src[0] != window || w == window || w->view_src == window || flat->src[0] != conv ||
        add->src[1] != flat || add->view_src || out->src[0] != add || out->view_src ||
        step->op != GGML_OP_TRANSPOSE || step->src[0]->op != GGML_OP_RESHAPE || step->src[0]->src[0] != cur ||
        tail->src[0] != window || state->src[0] != tail || state->src[1] != gf->nodes[i+6] || state->src[1]->op != GGML_OP_VIEW) {
        return false;
    }
    const int64_t nc = past->ne[1], nt = step->ne[0], ns = past->ne[2];
    if (past->type != GGML_TYPE_F32 || cur->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || state->type != GGML_TYPE_F32 ||
        !ggml_is_3d(past) || past->ne[0] != 3 || nc < 1 || nc > INT32_MAX || nt < 1 || nt > INT32_MAX || ns < 1 || ns > INT32_MAX ||
        w->ne[0] != 4 || w->ne[1] != nc || !ggml_is_contiguous(w) || !ggml_is_contiguous(cur) ||
        !ggml_is_matrix(cur) || cur->ne[0] != nc || cur->ne[1] != nt*ns || !ggml_are_same_shape(flat, cur) ||
        past->nb[0] != sizeof(float) || past->nb[1] != 3*sizeof(float) || !ggml_are_same_shape(tail, past) ||
        tail->view_offs != size_t(nt)*sizeof(float) || tail->nb[0] != sizeof(float) ||
        tail->nb[1] != window->nb[1] || (ns > 1 && tail->nb[2] != window->nb[2]) ||
        !ggml_is_matrix(state) || state->ne[0] != 3*nc || state->ne[1] != ns || state->nb[0] != sizeof(float) ||
        (ns > 1 && state->nb[1] < state->ne[0]*sizeof(float))) {
        return false;
    }
    if (mode == GGML_METAL_FUSION_FULL) {
        for (const auto * src : { past, cur, w }) {
            if (ggml_metal_fusion_overlap(state, src) || (ggml_metal_fusion_overlap(out, src) &&
                    !(nt == 1 && src == cur && out->data == cur->data))) {
                return false;
            }
        }
        if (ggml_metal_fusion_overlap(out, state)) {
            return false;
        }
    }
    return true;
}

static bool ggml_metal_fusion_check_gated_residual(
        const ggml_metal_fusion * fusion, const ggml_tensor * const * nodes,
        const ggml_cgraph * gf, const int * node_idxs, int idx, ggml_metal_fusion_mode mode) {
    const auto * cur = nodes[1]->src[1];
    const auto * dst = nodes[5];
    const int last = node_idxs[idx + 5];
    if (ggml_get_unary_op(nodes[0]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(nodes[3]) != GGML_UNARY_OP_TANH ||
        nodes[1]->src[0] != nodes[0] || nodes[2]->src[0] != cur || nodes[2]->src[1] != nodes[1] ||
        nodes[4]->src[0] != nodes[3] || dst->src[0] != nodes[2] || dst->src[1] != nodes[4] ||
        !ggml_is_contiguous(dst) || ggml_nelements(dst) > INT32_MAX ||
        !ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, 6, fusion->ops.data(), &last, 1)) {
        return false;
    }
    for (int j = 0; j < 6; ++j) {
        if (nodes[j]->type != GGML_TYPE_F32 || (j < 5 && nodes[j]->view_src)) {
            return false;
        }
    }
    for (const ggml_tensor * src : {nodes[0]->src[0], nodes[3]->src[0], nodes[1]->src[1], nodes[4]->src[1]}) {
        if (src->type != GGML_TYPE_F32 || !ggml_are_same_shape(src, dst) || src->nb[1] % sizeof(float) ||
            !ggml_is_contiguous_1(src) || ((src == nodes[0]->src[0] || src == nodes[3]->src[0]) && !ggml_is_contiguous(src)) ||
            (mode == GGML_METAL_FUSION_FULL && !(src->data == dst->data && ggml_is_contiguous(src)) && ggml_metal_fusion_overlap(src, dst))) {
            return false;
        }
        for (int j = 0; j < 5; ++j) {
            if (src == nodes[j] || src->view_src == nodes[j]) {
                return false;
            }
        }
    }
    return true;
}

static bool ggml_metal_fusion_check_situ(
        const ggml_metal_fusion * fusion, const ggml_tensor * const * nodes,
        const ggml_cgraph * gf, const int * node_idxs, int idx, ggml_metal_fusion_mode mode) {
    const auto * gate = nodes[0]->src[0];
    const auto * up   = nodes[5]->src[0];
    const auto * dst  = nodes[8];
    const int last = node_idxs[idx + 8];
    if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, 9, fusion->ops.data(), &last, 1) ||
        ggml_get_unary_op(nodes[1]) != GGML_UNARY_OP_TANH || ggml_get_unary_op(nodes[3]) != GGML_UNARY_OP_SIGMOID ||
        ggml_get_unary_op(nodes[6]) != GGML_UNARY_OP_TANH || nodes[1]->src[0] != nodes[0] || nodes[2]->src[0] != nodes[1] ||
        nodes[3]->src[0] != gate || nodes[4]->src[0] != nodes[2] || nodes[4]->src[1] != nodes[3] ||
        nodes[6]->src[0] != nodes[5] || nodes[7]->src[0] != nodes[6] || dst->src[0] != nodes[4] || dst->src[1] != nodes[7]) {
        return false;
    }
    for (int i = 0; i < 9; ++i) {
        if (nodes[i]->type != GGML_TYPE_F32 || (i < 8 && nodes[i]->view_src) ||
            (nodes[i]->op == GGML_OP_SCALE && ggml_get_op_params_f32(nodes[i], 1) != 0.0f)) {
            return false;
        }
        if (i < 5 && (up == nodes[i] || up->view_src == nodes[i])) {
            return false;
        }
    }
    for (const auto * src : {gate, up}) {
        if (src->type != GGML_TYPE_F32 || src->nb[0] != sizeof(float) || src->nb[1] % sizeof(float) ||
            !ggml_are_same_shape(src, dst) || !ggml_is_contiguous_1(src) ||
            (mode == GGML_METAL_FUSION_FULL && !(src->data == dst->data && ggml_is_contiguous(src)) && ggml_metal_fusion_overlap(src, dst))) {
            return false;
        }
    }
    return ggml_is_contiguous(dst) && dst->ne[0] <= INT32_MAX && ggml_nrows(dst) <= INT32_MAX;
}

static bool ggml_metal_fusion_check_clip_norm(
        const ggml_metal_fusion * fusion, const ggml_tensor * const * nodes,
        const ggml_cgraph * gf, const int * node_idxs, int idx, ggml_metal_fusion_mode mode) {
    const auto * x = nodes[0]->src[0];
    const auto * dst = nodes[7];
    const int last = node_idxs[idx + 7];
    if (x->type != GGML_TYPE_F32 || x->ne[0] > 128 || x->ne[0] % 4 || ggml_nrows(x) > INT32_MAX ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(dst) || !ggml_are_same_shape(x, dst) ||
        !ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, 8, fusion->ops.data(), &last, 1)) {
        return false;
    }
    for (int i = 0; i < 8; ++i) {
        if (nodes[i]->type != GGML_TYPE_F32 || (i < 7 && nodes[i]->view_src)) {
            return false;
        }
    }
    return nodes[1]->src[0] == nodes[0] && nodes[2]->src[0] == nodes[1] &&
           nodes[3]->src[0] == nodes[2] && ggml_get_op_params_f32(nodes[3], 0) == 1.0f &&
           nodes[4]->src[0] == nodes[3] && nodes[5]->src[0] == nodes[4] && nodes[5]->src[1] == nodes[3] &&
           nodes[6]->src[0] == nodes[5] && ggml_get_op_params_f32(nodes[6], 0) == 0.0f &&
           ggml_get_op_params_f32(nodes[6], 1) == 1.0f && dst->src[0] == x && dst->src[1] == nodes[6] &&
           (mode == GGML_METAL_FUSION_STRUCTURAL || x->data == dst->data || !ggml_metal_fusion_overlap(x, dst));
}

static bool ggml_metal_fusion_check_cont_mul_mat(
        const ggml_metal_fusion * fusion, const ggml_tensor * const * nodes,
        const ggml_cgraph * gf, const int * node_idxs, int idx, ggml_metal_fusion_mode mode) {
    const auto * copy = nodes[0];
    const auto * mm = nodes[1];
    const auto * a = copy->src[0];
    const auto * b = mm->src[1];
    const int last = node_idxs[idx + 1];
    if (mm->src[0] != copy || b == copy || b->view_src == copy || !ggml_are_same_shape(a, copy) ||
        a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || a->ne[0] > 8 || a->ne[1] > INT32_MAX ||
        a->nb[1] != sizeof(float) || b->nb[0] != sizeof(float) || a->ne[2] != b->ne[2] || b->ne[1] > INT32_MAX ||
        b->ne[2] > INT32_MAX || !ggml_is_3d(a) || !ggml_is_3d(b) || ggml_get_op_params_i32(mm, 1) != GGML_HINT_NONE ||
        ggml_prec_get_acc(mm) != GGML_PREC_UNDEFINED ||
        ggml_prec_get_src(mm, 1) != GGML_PREC_UNDEFINED || !ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, 2, fusion->ops.data(), &last, 1)) {
        return false;
    }
    return mode == GGML_METAL_FUSION_STRUCTURAL || (!ggml_metal_fusion_overlap(mm, a) && !ggml_metal_fusion_overlap(mm, b));
}

static bool ggml_metal_fusion_check_rel_mask(
        const ggml_metal_fusion * fusion, const ggml_tensor * const * nodes,
        const ggml_cgraph * gf, const int * node_idxs, int idx, ggml_metal_fusion_mode mode) {
    const bool project = nodes[0]->op == GGML_OP_MUL_MAT;
    const int gather_idx = (int) fusion->ops.size() - 3;
    const int start = node_idxs[idx];
    const int first = node_idxs[idx + gather_idx];
    const int last  = node_idxs[idx + gather_idx + 2];
    if (last - start + 1 != (int) fusion->ops_all.size() || last != first + 4 || node_idxs[idx + gather_idx + 1] != first + 2 ||
        !ggml_can_fuse_subgraph(gf, start, last - start + 1, fusion->ops_all.data(), &last, 1)) {
        return false;
    }

    const auto * gather  = nodes[gather_idx];
    const auto * reshape = gf->nodes[first + 1];
    const auto * add     = nodes[gather_idx + 1];
    const auto * perm    = gf->nodes[first + 3];
    const auto * dst     = nodes[gather_idx + 2];
    const auto * table   = gather->src[0];
    const auto * ids     = gather->src[1];
    const auto * mask    = add->src[1];
    if (gather_idx > 0) {
        if (table != nodes[gather_idx - 1] || !ggml_are_same_shape(table, table->src[0]) || mask == table || mask->view_src == table) {
            return false;
        }
        table = table->src[0];
    } else if (table->nb[0] != sizeof(float)) {
        return false;
    }
    if (reshape->src[0] != gather ||
        add->src[0] != reshape || (add->view_src && add->view_src != gather) || mask == gather || mask->view_src == gather || perm->src[0] != add ||
        dst->src[0] != perm || dst->src[1] != dst || !ggml_are_same_shape(perm, dst) ||
        table->type != GGML_TYPE_F32 || mask->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16 ||
        !ggml_is_contiguous(ids) || !ggml_is_contiguous(dst)) {
        return false;
    }

    const int64_t keys = ids->ne[0], queries = dst->ne[1], heads = table->ne[0], streams = dst->ne[3];
    if (keys > INT32_MAX || queries > INT32_MAX || heads < 1 || heads > INT32_MAX || streams > INT32_MAX || table->ne[1] > INT32_MAX ||
        !ggml_is_3d(ids) || !((ids->ne[1] == queries*streams && ids->ne[2] == 1) || (ids->ne[1] == queries && ids->ne[2] == streams)) ||
        table->ne[2] != ids->ne[1] || table->ne[3] != ids->ne[2] || dst->ne[0] != keys || dst->ne[2] != heads ||
        reshape->ne[0] != heads || reshape->ne[1] != keys || reshape->ne[2] != queries || reshape->ne[3] != streams ||
        perm->nb[0] != add->nb[1] || perm->nb[1] != add->nb[2] || perm->nb[2] != add->nb[0] || perm->nb[3] != add->nb[3] ||
        mask->ne[0] != 1 || mask->ne[1] != keys || mask->ne[2] != queries || mask->ne[3] != streams ||
        mask->nb[1] != sizeof(float) || (queries > 1 && mask->nb[2] != size_t(keys)*sizeof(float))) {
        return false;
    }

    if (project) {
        const auto * mm = nodes[0];
        const auto * sub = nodes[1];
        const auto * basis = mm->src[0];
        const auto * features = mm->src[1];
        const auto * sink = sub->src[1];
        if (sub->src[0] != gf->nodes[start + 1] || sub->src[0]->src[0] != mm || (sub->view_src && sub->view_src != mm) ||
            table->op != GGML_OP_PERMUTE || table->src[0] != sub || !ggml_is_matrix(ids) ||
            !ggml_is_matrix(basis) || !ggml_is_matrix(features) || basis->ne[0] != 16 || basis->ne[1] < 1 || basis->ne[1] != table->ne[1] ||
            !ggml_is_contiguous(basis) || !ggml_is_contiguous(features) || features->ne[1] % heads || features->ne[1]/heads != ids->ne[1] ||
            basis->type != GGML_TYPE_F32 || features->type != GGML_TYPE_F32 || sink->type != GGML_TYPE_F32 ||
            sub->ne[0] != basis->ne[1] || sub->ne[1] != heads || sub->ne[2] != ids->ne[1] || !ggml_is_3d(sub) ||
            table->nb[0] != sub->nb[1] || table->nb[1] != sub->nb[0] || table->nb[2] != sub->nb[2] ||
            sink->ne[0] != 1 || sink->ne[1] != heads || sink->ne[2] != ids->ne[1] || !ggml_is_3d(sink) || sink->nb[1] != sizeof(float) ||
            sink == mm || sink->view_src == mm || ggml_get_op_params_i32(mm, 1) != GGML_HINT_NONE ||
            ggml_prec_get_acc(mm) != GGML_PREC_UNDEFINED || ggml_prec_get_src(mm, 1) != GGML_PREC_UNDEFINED) {
            return false;
        }
        for (const auto * src : { basis, features, sink, ids, mask }) {
            for (int i = start; i < last; ++i) {
                if (src == gf->nodes[i] || src->view_src == gf->nodes[i]) {
                    return false;
                }
            }
            if (mode == GGML_METAL_FUSION_FULL && ggml_metal_fusion_overlap(dst, src)) {
                return false;
            }
        }
        return true;
    }
    return mode == GGML_METAL_FUSION_STRUCTURAL || (!ggml_metal_fusion_overlap(dst, table) &&
           !ggml_metal_fusion_overlap(dst, ids) && !ggml_metal_fusion_overlap(dst, mask));
}

// ---- patterns ------------------------------------------------------------

static const std::vector<ggml_op> ops_norm_mul         = { GGML_OP_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_norm_mul_add     = { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_norm_scale       = { GGML_OP_NORM, GGML_OP_SCALE };
static const std::vector<ggml_op> ops_rms_norm_mul     = { GGML_OP_RMS_NORM, GGML_OP_MUL };
static const std::vector<ggml_op> ops_rms_norm_mul_add = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD };
static const std::vector<ggml_op> ops_rms_norm_scale   = { GGML_OP_RMS_NORM, GGML_OP_SCALE };

static const std::vector<ggml_op> ops_add_2 = { GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_3 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_4 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_5 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_6 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_add_7 = { GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD };
static const std::vector<ggml_op> ops_snake = { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD };

static const std::vector<ggml_op> ops_gdn_cache = { GGML_OP_GATED_DELTA_NET, GGML_OP_CPY };

static const std::vector<ggml_op> ops_ssm_conv_silu = { GGML_OP_SSM_CONV, GGML_OP_UNARY };
static const std::vector<ggml_op> ops_ssm_conv_state = {
    GGML_OP_CONCAT, GGML_OP_SSM_CONV, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_SCALE, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY,
};

static const std::vector<ggml_op> ops_moe_reduce_2 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_3 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_4 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_5 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_6 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_7 = {
    GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};
static const std::vector<ggml_op> ops_moe_reduce_8 = {
    GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_VIEW,
    GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD, GGML_OP_ADD
};

static const std::vector<ggml_op> ops_mul_mat_add = { GGML_OP_MUL_MAT, GGML_OP_ADD };
static const std::vector<ggml_op> ops_cont_mul_mat = { GGML_OP_CONT, GGML_OP_MUL_MAT };
static const std::vector<ggml_op> ops_rel_mask_cont = { GGML_OP_CONT, GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_PERMUTE, GGML_OP_CPY };
static const std::vector<ggml_op> ops_rel_mask_project = {
    GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_SUB, GGML_OP_PERMUTE, GGML_OP_CONT,
    GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_PERMUTE, GGML_OP_CPY,
};
static const std::vector<ggml_op> ops_rel_mask = { GGML_OP_GET_ROWS, GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_PERMUTE, GGML_OP_CPY };
static const std::vector<ggml_op> ops_clip_norm = {
    GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL,
};
static const std::vector<ggml_op> ops_situ = {
    GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_MUL,
    GGML_OP_VIEW, GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_MUL,
};
static const std::vector<ggml_op> ops_gated_residual = {
    GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_SUB, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD,
};

static const std::vector<ggml_metal_fusion> ggml_metal_fusions = {
    { GGML_METAL_FUSION_NORM_MUL,     ops_norm_mul,         {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_norm_mul_add,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_norm_scale,       {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL,     ops_rms_norm_mul,     {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_MUL_ADD, ops_rms_norm_mul_add, {}, false, ggml_metal_fusion_check_norm },
    { GGML_METAL_FUSION_NORM_SCALE,   ops_rms_norm_scale,   {}, false, ggml_metal_fusion_check_norm },

    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_2, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_3, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_4, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_5, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_6, {}, false, ggml_metal_fusion_check_add_chain },
    { GGML_METAL_FUSION_ADD_CHAIN, ops_add_7, {}, false, ggml_metal_fusion_check_add_chain },

    { GGML_METAL_FUSION_SNAKE, ops_snake, {}, false, ggml_metal_fusion_check_snake },

    { GGML_METAL_FUSION_GDN_CACHE, ops_gdn_cache, {}, true, ggml_metal_fusion_check_gdn_cache },

    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe,                     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_scale,               {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm,                {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale,          {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_bias,           {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_norm_scale_bias,     {1}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp,                  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_scale,            {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm,             {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale,       {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_bias,        {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sp_norm_scale_bias,  {3}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig,                 {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_scale,           {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm,            {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale,      {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_bias,       {2}, true, ggml_metal_fusion_check_topk_moe },
    { GGML_METAL_FUSION_TOPK_MOE, ops_topk_moe_sig_norm_scale_bias, {2}, true, ggml_metal_fusion_check_topk_moe },

    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_2, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_3, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_4, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_5, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_6, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_7, {}, true, ggml_metal_fusion_check_moe_reduce },
    { GGML_METAL_FUSION_MOE_REDUCE, ops_moe_reduce_8, {}, true, ggml_metal_fusion_check_moe_reduce },

    { GGML_METAL_FUSION_SSM_CONV_SILU, ops_ssm_conv_silu, {}, false, ggml_metal_fusion_check_ssm_conv_silu },
    { GGML_METAL_FUSION_SSM_CONV_STATE, ops_ssm_conv_state, {3}, true, ggml_metal_fusion_check_ssm_conv_state },

    { GGML_METAL_FUSION_MUL_MAT_ADD, ops_mul_mat_add, {}, false, ggml_metal_fusion_check_mul_mat_add },
    { GGML_METAL_FUSION_CONT_MUL_MAT, ops_cont_mul_mat, {}, true, ggml_metal_fusion_check_cont_mul_mat },
    { GGML_METAL_FUSION_REL_MASK, ops_rel_mask_project, {}, true, ggml_metal_fusion_check_rel_mask },
    { GGML_METAL_FUSION_REL_MASK, ops_rel_mask_cont,    {}, true, ggml_metal_fusion_check_rel_mask },
    { GGML_METAL_FUSION_REL_MASK, ops_rel_mask, {}, true, ggml_metal_fusion_check_rel_mask },
    { GGML_METAL_FUSION_CLIP_NORM, ops_clip_norm, {}, true, ggml_metal_fusion_check_clip_norm },
    { GGML_METAL_FUSION_SITU, ops_situ, {}, true, ggml_metal_fusion_check_situ },
    { GGML_METAL_FUSION_GATED_RESIDUAL, ops_gated_residual, {}, true, ggml_metal_fusion_check_gated_residual },
};

static const std::vector<const ggml_metal_fusion *> & ggml_metal_fusion_candidates(ggml_op op) {
    static const auto candidates = [] {
        std::array<std::vector<const ggml_metal_fusion *>, GGML_OP_COUNT> result;
        for (const auto & fusion : ggml_metal_fusions) {
            result[fusion.ops.front()].push_back(&fusion);
        }
        return result;
    }();
    return candidates[op];
}

// ---- alloc deps -----------------------------------------------------------

static bool ggml_metal_fusion_match_raw_pattern(
        const ggml_cgraph * gf, int node_idx, const std::vector<ggml_op> & ops) {
    if (node_idx < 0 || node_idx + (int) ops.size() > gf->n_nodes) {
        return false;
    }

    for (int i = 0; i < (int) ops.size(); ++i) {
        if (gf->nodes[node_idx + i]->op != ops[i]) {
            return false;
        }
    }

    return true;
}

static void ggml_metal_fusion_add_pattern_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf,
        const ggml_metal_fusion * fusion,
        int node_idx) {
    const int last_node = node_idx + (int) fusion->ops_all.size() - 1;

    // keep all external inputs alive until the fused output
    std::set<ggml_tensor *> seen;
    for (int j = 0; j < (int) fusion->ops_all.size(); ++j) {
        ggml_tensor * node = gf->nodes[node_idx + j];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            ggml_tensor * src = node->src[s];
            if (src && seen.insert(src).second) {
                add_alloc_dep(user_data, src, gf->nodes[last_node]);
            }
        }
        seen.insert(node);
    }
}

void ggml_metal_fusion_add_alloc_deps(
        void * user_data,
        void (*add_alloc_dep)(void *, ggml_tensor *, ggml_tensor *),
        ggml_cgraph * gf) {
    for (int i = 0; i < gf->n_nodes; ++i) {
        const ggml_metal_fusion * best = nullptr;
        int best_raw = 0;

        for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
            if ((int) fusion.ops_all.size() <= best_raw) {
                continue;
            }
            if (ggml_metal_fusion_match_raw_pattern(gf, i, fusion.ops_all)) {
                best = &fusion;
                best_raw = (int) fusion.ops_all.size();
            }
        }

        if (best) {
            ggml_metal_fusion_add_pattern_alloc_deps(user_data, add_alloc_dep, gf, best, i);
            i += best_raw - 1;
        }
    }
}

// ---- shared fusion info ---------------------------------------------------

static std::string ggml_metal_fusion_label(const ggml_metal_fusion * fusion) {
    GGML_ASSERT(fusion != nullptr);

    std::string label;
    for (int j = 0; j < (int) fusion->ops.size(); j++) {
        if (j > 0) {
            label += '+';
        }
        label += ggml_op_name(fusion->ops[j]);
    }
    return label;
}

struct ggml_metal_fusion_info {
    std::vector<std::string> labels;
    std::vector<uint64_t>    counts;
    bool enabled;
    bool stats;
    bool labels_set;
    int  debug;
};

struct ggml_metal_fusion_info * ggml_metal_fusion_info_init(bool enabled, int debug) {
    ggml_metal_fusion_info * finfo = new ggml_metal_fusion_info;
    finfo->enabled    = enabled;
    finfo->stats      = debug > 0;
    finfo->labels_set = false;
    finfo->debug      = debug;

    if (finfo->stats) {
        ggml_metal_fusion_info_labels_init(finfo);
    }

    return finfo;
}

void ggml_metal_fusion_info_free(ggml_metal_fusion_info * finfo) {
    delete finfo;
}

bool ggml_metal_fusion_info_enabled(const ggml_metal_fusion_info * finfo) {
    return finfo->enabled;
}

bool ggml_metal_fusion_info_stats(const ggml_metal_fusion_info * finfo) {
    return finfo->stats;
}

int ggml_metal_fusion_info_debug(const ggml_metal_fusion_info * finfo) {
    return finfo->debug;
}

int ggml_metal_fusion_info_n_fusions(const ggml_metal_fusion_info * finfo) {
    return (int) finfo->labels.size();
}

const char * ggml_metal_fusion_info_label(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->labels.size());
    return finfo->labels[idx].c_str();
}

uint64_t ggml_metal_fusion_info_count(const ggml_metal_fusion_info * finfo, int idx) {
    GGML_ASSERT(idx >= 0 && idx < (int) finfo->counts.size());
    return finfo->counts[idx];
}

void ggml_metal_fusion_info_count_fusion(ggml_metal_fusion_info * finfo, const ggml_metal_fusion * fusion) {
    if (!finfo->stats || fusion == nullptr) {
        return;
    }

    const ptrdiff_t idx = fusion - ggml_metal_fusions.data();
    if (idx >= 0 && idx < (ptrdiff_t) finfo->counts.size()) {
        finfo->counts[idx]++;
    }
}

void ggml_metal_fusion_info_set_enabled(ggml_metal_fusion_info * finfo, bool enabled) {
    finfo->enabled = enabled;
}

void ggml_metal_fusion_info_labels_init(ggml_metal_fusion_info * finfo) {
    if (finfo->labels_set) {
        return;
    }

    finfo->labels.clear();
    finfo->counts.assign(ggml_metal_fusions.size(), 0);
    finfo->labels.reserve(ggml_metal_fusions.size());

    for (const ggml_metal_fusion & fusion : ggml_metal_fusions) {
        finfo->labels.emplace_back(ggml_metal_fusion_label(&fusion));
    }

    finfo->labels_set = true;
}

void ggml_metal_fusion_info_stats_init(ggml_metal_fusion_info * finfo) {
    finfo->stats = true;
    ggml_metal_fusion_info_labels_init(finfo);
}

void ggml_metal_fusion_info_stats_reset(ggml_metal_fusion_info * finfo) {
    std::fill(finfo->counts.begin(), finfo->counts.end(), 0);
}

int ggml_metal_fusion_info_stats_get(const ggml_metal_fusion_info * finfo, const char ** labels, uint64_t * counts, int n) {
    const int n_fusions = (int) finfo->labels.size();

    if (labels == nullptr) {
        return n_fusions;
    }

    const int n_fill = std::min(n, n_fusions);
    for (int i = 0; i < n_fill; i++) {
        labels[i] = finfo->labels[i].c_str();
        if (counts != nullptr) {
            counts[i] = finfo->counts[i];
        }
    }

    return n_fill;
}

// ---- memory-range checks -------------------------------------------------

// reject fusions where an external source overlaps any fused output. the fused
// kernels elide intermediate nodes, so only sources that are not part of the
// fused subgraph can cause read/write races with the output.
static bool ggml_metal_fusion_check_memory_ranges(
        const ggml_metal_fusion      * fusion,
        const ggml_tensor * const    * nodes,
        int                            node_count) {
    // some fused kernels write through a tensor that also appears as a source (e.g. the gdn
    // cache cpy), so a source that is the same memory as the output is not an external read
    // source
    auto same_memory = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->data && b->data && a->data == b->data) {
            return true;
        }
        for (const ggml_tensor * v = a; v; v = v->view_src) {
            if (v == b) {
                return true;
            }
        }
        for (const ggml_tensor * v = b; v; v = v->view_src) {
            if (v == a) {
                return true;
            }
        }
        return false;
    };

    auto nodes_overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (!a || !b || !a->data || !b->data || !a->buffer || !b->buffer) {
            return false;
        }

        if (a->buffer != b->buffer) {
            return false;
        }

        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        return (b_start <= a_start && a_start < b_end) ||
               (a_start <= b_start && b_start < a_end);
    };

    auto is_intermediate = [](const ggml_tensor * src, const ggml_tensor * const * nodes, int j) {
        for (int k = 0; k < j; ++k) {
            if (src == nodes[k]) {
                return true;
            }
            for (const ggml_tensor * view_src = src->view_src; view_src; view_src = view_src->view_src) {
                if (view_src == nodes[k]) {
                    return true;
                }
            }
        }
        return false;
    };

    auto check_dst = [&](const ggml_tensor * dst) {
        for (int j = 0; j < node_count; ++j) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * src = nodes[j]->src[s];
                if (!src || src->op == GGML_OP_NONE || same_memory(src, dst)) {
                    continue;
                }

                if (nodes_overlap(dst, src) && !is_intermediate(src, nodes, j)) {
                    return false;
                }
            }
        }
        return true;
    };

    if (!check_dst(nodes[node_count - 1])) {
        return false;
    }

    for (int offset : fusion->outs) {
        GGML_ASSERT(offset >= 0 && offset < node_count);
        if (!check_dst(nodes[offset])) {
            return false;
        }
    }

    return true;
}

// ---- queries -------------------------------------------------------------

// find the longest pattern matching the node sequence starting at idx
// (idx is a position in node_idxs, which maps to graph node indices)
const ggml_metal_fusion * ggml_metal_fusion_next(
        const ggml_cgraph * gf,
        const int * node_idxs,
        int n_idxs,
        int idx,
        ggml_metal_fusion_mode mode,
        int * n_out) {
    const ggml_metal_fusion * res = nullptr;
    int best = 1;

    if (idx >= n_idxs) {
        *n_out = best;
        return res;
    }

    for (const auto * candidate : ggml_metal_fusion_candidates(gf->nodes[node_idxs[idx]]->op)) {
        const auto & fusion = *candidate;
        const int n_ops = (int) fusion.ops.size();

        // only look for a longer match than the current best
        if (n_ops <= best) {
            continue;
        }
        if (idx + n_ops > n_idxs) {
            continue;
        }

        const ggml_tensor * nodes[GGML_METAL_FUSION_MAX];

        // the op sequence must match exactly
        bool ok = true;
        for (int j = 0; j < n_ops; j++) {
            nodes[j] = gf->nodes[node_idxs[idx + j]];
            if (nodes[j]->op != fusion.ops[j]) {
                ok = false;
                break;
            }
        }
        if (!ok) {
            continue;
        }

        if (!fusion.unsafe) {
            // common element-wise chain constraints: each node reads the previous one,
            // and all nodes have the same shape
            for (int j = 1; j < n_ops && ok; j++) {
                if (nodes[j]->src[0] != nodes[j - 1] && nodes[j]->src[1] != nodes[j - 1]) {
                    ok = false;
                    break;
                }
                if (!ggml_are_same_shape(nodes[j], nodes[j - 1])) {
                    ok = false;
                    break;
                }
            }
            if (!ok) {
                continue;
            }

            // primary output is the last node; additional outputs come from fusion.outs
            int outputs_buf[GGML_METAL_FUSION_MAX];
            outputs_buf[0] = node_idxs[idx + n_ops - 1];
            for (size_t i = 0; i < fusion.outs.size(); ++i) {
                const int out_offset = fusion.outs[i];
                GGML_ASSERT(out_offset >= 0 && out_offset < n_ops);
                outputs_buf[i + 1] = node_idxs[idx + out_offset];
            }

            const int n_outputs = 1 + (int) fusion.outs.size();

            // structural subgraph checks (op sequence, elidable uses, view containment)
            if (!ggml_can_fuse_subgraph_ext(gf, node_idxs + idx, n_ops, fusion.ops.data(), outputs_buf, n_outputs)) {
                continue;
            }
        }

        // pattern-specific checks (the sole validator for unsafe patterns)
        if (fusion.check && !fusion.check(&fusion, nodes, gf, node_idxs, idx, mode)) {
            continue;
        }

        // the compute phase has allocated tensors and can detect aliasing between
        // external sources and fused outputs; the optimizer phase cannot do this yet
        if (mode == GGML_METAL_FUSION_FULL &&
                !ggml_metal_fusion_check_memory_ranges(&fusion, nodes, n_ops)) {
            continue;
        }

        best = n_ops;
        res = &fusion;
    }

    *n_out = best;

    return res;
}

// optimize phase: maximum number of nodes starting at idx (a raw sequential graph index) that
// could be fused, chaining patterns back-to-back. matching runs on the same filtered (view
// transparent) node sequence that the compute phase uses, so the returned count is the raw index
// span from idx to the last matched node (intermediate views are packed along).
int ggml_metal_fusion_max(const ggml_cgraph * gf, int idx) {
    if (ggml_metal_fusion_candidates(gf->nodes[idx]->op).empty()) {
        return 1;
    }

    // collect the non-view node indices starting at idx; 0-element tensors are included so
    // that empty graphs pack like their non-empty counterparts (see ggml_metal_fusion_filter_ops)
    int idxs[GGML_METAL_FUSION_MAX];
    int n_idxs = 0;
    for (int i = idx; i < gf->n_nodes && n_idxs < GGML_METAL_FUSION_MAX; i++) {
        if (!ggml_op_is_empty(gf->nodes[i]->op)) {
            idxs[n_idxs++] = i;
        }
    }
    if (n_idxs == 0) {
        return 1;
    }

    int total = 0;
    int i_f = 0;

    while (i_f < n_idxs && total < GGML_METAL_FUSION_MAX) {
        int len = 1;
        const ggml_metal_fusion * fusion = ggml_metal_fusion_next(gf, idxs, n_idxs, i_f, GGML_METAL_FUSION_STRUCTURAL, &len);
        if (!fusion || total + len > GGML_METAL_FUSION_MAX) {
            break;
        }

        total += len;
        i_f += len;
    }

    if (i_f == 0) {
        return 1;
    }

    // map the matched non-empty nodes back to the raw index span (views are included)
    return std::min(GGML_METAL_FUSION_MAX, idxs[i_f - 1] - idx + 1);
}
