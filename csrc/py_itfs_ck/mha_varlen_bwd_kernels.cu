// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "py_itfs_common.h"
#include "mha_common.h"

#include "mha_bwd.h"

namespace aiter {
namespace torch_itfs {
fmha_bwd_args get_ck_fmha_varlen_bwd_args(const mask_info &mask,
                                          // sizes
                                          const int b,
                                          const int max_seqlen_q,
                                          const int max_seqlen_k,
                                          const int h,
                                          const int h_k,
                                          const int hdim_q,
                                          const int hdim_v,
                                          // device pointers
                                          const at::Tensor q,
                                          const at::Tensor k,
                                          const at::Tensor v,
                                          const at::Tensor seqlens_q,
                                          const at::Tensor seqlens_k,
                                          std::optional<const at::Tensor> &alibi_slopes_,
                                          const at::Tensor out,
                                          const at::Tensor softmax_lse,
                                          const at::Tensor dout,
                                          at::Tensor dq_acc,
                                          at::Tensor d,
                                          at::Tensor dq,
                                          at::Tensor dk,
                                          at::Tensor dv,
                                          float softmax_scale,
                                          float p_dropout,
                                          std::pair<uint64_t*, uint64_t*> drop_seed_offset)
{
    ck_tile::index_t total_q = q.size(0);
    ck_tile::index_t total_k = k.size(0);

    // q: (total_q, nheads, hdim_q)
    ck_tile::index_t batch_stride_q = 0;
    ck_tile::index_t stride_q = q.stride(0);
    ck_tile::index_t nhead_stride_q = q.stride(1);

    // k: (total_k, nheads_k, hdim_q)
    ck_tile::index_t batch_stride_k = 0;
    ck_tile::index_t stride_k = k.stride(0);
    ck_tile::index_t nhead_stride_k = k.stride(1);

    // v: (total_k, nheads_k, hdim_v)
    ck_tile::index_t batch_stride_v = 0;
    ck_tile::index_t stride_v = v.stride(0);
    ck_tile::index_t nhead_stride_v = v.stride(1);

    // o: (total_q, nheads, hdim_v)
    ck_tile::index_t batch_stride_o = 0;
    ck_tile::index_t stride_o = out.stride(0);
    ck_tile::index_t nhead_stride_o = out.stride(1);

    // lse: (nheads, total_q)
    ck_tile::index_t batch_stride_lse = 0;
    ck_tile::index_t nhead_stride_lse = softmax_lse.stride(0);

    // do: (total_q, nheads, hdim)
    ck_tile::index_t batch_stride_do = 0;
    ck_tile::index_t stride_do = dout.stride(0);
    ck_tile::index_t nhead_stride_do = dout.stride(1);

    // d: (batch_size, nheads, max_seqlen_q)
    // CK assume d share the same stride with lse

    // dq: (total_q, nheads, hdim_q)
    ck_tile::index_t batch_stride_dq = 0;
    ck_tile::index_t stride_dq = dq.stride(0);
    ck_tile::index_t nhead_stride_dq = dq.stride(1);


    // dk_expanded: (total_k, nheads, hdim_q)
    ck_tile::index_t batch_stride_dk = 0;
    ck_tile::index_t stride_dk = dk.stride(0);
    ck_tile::index_t nhead_stride_dk = dk.stride(1);

    // dv_expanded: (total_k, nheads, hdim_v)
    ck_tile::index_t batch_stride_dv = 0;
    ck_tile::index_t stride_dv = dv.stride(0);
    ck_tile::index_t nhead_stride_dv = dv.stride(1);

    // dq_acc: (split, total_q, nheads, hdim_v)
    ck_tile::index_t split_stride_dq_acc = dq_acc.stride(0);
    ck_tile::index_t batch_stride_dq_acc = 0;
    ck_tile::index_t stride_dq_acc = dq_acc.stride(1);
    ck_tile::index_t nhead_stride_dq_acc = dq_acc.stride(2);

    float p_undrop = 1.0 - p_dropout;

    void *alibi_slopes_ptr = nullptr;
    ck_tile::index_t stride_alibi_slopes = 0;

    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        TORCH_CHECK(alibi_slopes.sizes() == torch::IntArrayRef({h}) || alibi_slopes.sizes() == torch::IntArrayRef({b, h}));
        alibi_slopes_ptr = alibi_slopes.data_ptr();
        // alibi_slopes:(batch_size, nheads) or (nhead)
        stride_alibi_slopes = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    }

    return fmha_bwd_args{q.data_ptr(),
                         k.data_ptr(),
                         v.data_ptr(),
                         alibi_slopes_ptr, // bias
                         out.data_ptr(),
                         softmax_lse.data_ptr(),
                         dout.data_ptr(),
                         d.data_ptr(),
                         nullptr, // rand_val
                         dq.data_ptr(),
                         dk.data_ptr(),
                         dv.data_ptr(),
                         nullptr, // dbias
                         dq_acc.data_ptr(), // dq_acc
                         seqlens_q.data_ptr(), // seqstart_q
                         seqlens_k.data_ptr(), // seqstart_k
                         nullptr, // seqlen_k_ptr
                         total_q,
                         total_k,
                         b,
                         max_seqlen_q, // max_seqlen_q
                         max_seqlen_k, // max_seqlen_k
                         hdim_q, // hdim_q
                         hdim_v, // hdim_v
                         h, // nhead
                         h_k, // nhead_k
                         softmax_scale,
                         stride_q,
                         stride_k,
                         stride_v,
                         stride_alibi_slopes,
                         stride_o,
                         0, // stride_randval
                         stride_do,
                         stride_dq_acc,
                         stride_dq,
                         stride_dk,
                         stride_dv,
                         0, // stride_dbias
                         nhead_stride_q,
                         nhead_stride_k,
                         nhead_stride_v,
                         0, // nhead_stride_bias
                         nhead_stride_o,
                         0, // nhead_stride_randval
                         nhead_stride_do,
                         nhead_stride_lse,
                         nhead_stride_dq_acc,
                         nhead_stride_dq,
                         nhead_stride_dk,
                         nhead_stride_dv,
                         0, // nhead_stride_dbias
                         batch_stride_q,
                         batch_stride_k,
                         batch_stride_v,
                         0  , // batch_stride_bias
                         batch_stride_o,
                         0, // batch_stride_randval
                         batch_stride_do,
                         batch_stride_lse,
                         batch_stride_dq_acc,
                         batch_stride_dq,
                         batch_stride_dk,
                         batch_stride_dv,
                         0  , // batch_stride_dbias, FA without dbias
                         split_stride_dq_acc,
                         mask.left,
                         mask.right,
                         static_cast<ck_tile::index_t>(mask.type),
                         p_dropout,
                         p_undrop,
                         drop_seed_offset};
}

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,         // [total_q, hq, d_v]
               const at::Tensor &q,            // [total_q, hq, d_q]
               const at::Tensor &k,            // [total_k, hk, d_q]
               const at::Tensor &v,            // [total_k, hk, d_v]
               const at::Tensor &out,          // [total_q, hq, d_v]
               const at::Tensor &softmax_lse,  // [b, hq, sq]
               const at::Tensor &cu_seqlens_q, // [b+1]
               const at::Tensor &cu_seqlens_k, // [b+1]
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const bool deterministic,
               std::optional<at::Tensor> dq_,                 // [total_q, hq, d_q]
               std::optional<at::Tensor> dk_,                 // [total_k, hk, d_q]
               std::optional<at::Tensor> dv_,                 // [total_k, hk, d_v]
               std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
               std::optional<const at::Tensor> rng_state_,
               std::optional<at::Generator> gen_)
{
    if (is_causal) { window_size_right = 0; }

    bool is_dropout = p_dropout > 0.0;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");

    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");
    TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype int32");

    std::string q_dtype_str = q_dtype == torch::kFloat16 ? "fp16" : "bf16";

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    CHECK_DEVICE(cu_seqlens_q); CHECK_DEVICE(cu_seqlens_k);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    CHECK_CONTIGUOUS(cu_seqlens_q);
    CHECK_CONTIGUOUS(cu_seqlens_k);

    const auto sizes = q.sizes();

    const int total_q = sizes[0];
    const int batch_size = cu_seqlens_q.numel() - 1;
    const int num_heads = sizes[1];
    const int head_size_q = sizes[2];
    const int head_size_v = v.size(2);
    const int total_k = k.size(0);
    const int num_heads_k = k.size(1);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size_q % 8 == 0, "head_size_q should be a multiple of 8");
    TORCH_CHECK(head_size_v % 8 == 0, "head_size_v should be a multiple of 8");
    TORCH_CHECK(head_size_q <= 256, "CK FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(head_size_v <= 256, "CK FlashAttention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (window_size_left >= max_seqlen_k) { window_size_left = -1; }
    if (window_size_right >= max_seqlen_k) { window_size_right = -1; }

    mask_info mask;
    if (is_causal) {
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + "0";
        mask = mask_info::decode(mask_identify, max_seqlen_q, max_seqlen_k); // casual
    }
    else if (window_size_left == -1 && window_size_right == -1) {
        mask = mask_info::decode("0", max_seqlen_q, max_seqlen_k); // no mask
    }
    else {
        // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
        std::string mask_identify = "b:" + std::to_string(window_size_left) + "," + std::to_string(window_size_right);
        mask = mask_info::decode(mask_identify, max_seqlen_q, max_seqlen_k); // local
    }

    // q, k, v, out had been padded in mha_fwd
    // dq_, dk_, dv_ are also padded tensor
    CHECK_SHAPE(q, total_q, num_heads, head_size_q);
    CHECK_SHAPE(k, total_k, num_heads_k, head_size_q);
    CHECK_SHAPE(v, total_k, num_heads_k, head_size_v);
    CHECK_SHAPE(out, total_q, num_heads, head_size_v);
    CHECK_SHAPE(dout, total_q, num_heads, head_size_v);
    CHECK_SHAPE(cu_seqlens_q, batch_size + 1);
    CHECK_SHAPE(cu_seqlens_k, batch_size + 1);

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, total_q, num_heads, head_size_q);
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, total_k, num_heads_k, head_size_q);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, total_k, num_heads_k, head_size_v);
    } else {
        dv = torch::empty_like(v);
    }

    at::cuda::CUDAGuard device_guard{q.device()};

    auto opts = q.options();
    auto softmax_d = torch::empty({batch_size, num_heads, max_seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;

    if (!deterministic) {
        dq_accum = torch::zeros({1, total_q, num_heads, head_size_v}, opts.dtype(at::kFloat));
    } else {
        const ck_tile::index_t kN0 = head_size_q <= 128 ? 128 : 64;
        const ck_tile::index_t nsplits = ck_tile::integer_divide_ceil(max_seqlen_k, kN0);
        dq_accum = torch::zeros({nsplits, total_q, num_heads, head_size_v}, opts.dtype(at::kFloat));
    }

    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_expanded = torch::empty({total_k, num_heads, head_size_q}, opts);
        dv_expanded = torch::empty({total_k, num_heads, head_size_v}, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if(zero_tensors) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    bias_enum bias_type = alibi_slopes_.has_value() ? bias_enum::alibi : bias_enum::no_bias;

    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        gen_, at::cuda::detail::getDefaultCUDAGenerator());

    int64_t counter_offset = batch_size * num_heads * ck_tile::get_warp_size();
    at::Tensor rng_state;

    if (rng_state_.has_value()) {
        rng_state = rng_state_.value();
    } else if(is_dropout) {
        rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        auto philox_args = gen->philox_cuda_state(counter_offset);
        hipLaunchKernelGGL(
            aiter::ParsePhiloxCudaState, dim3(1), dim3(64), 0, 0,
            philox_args, reinterpret_cast<uint64_t*>(rng_state.data_ptr()));
    } else {
        rng_state = torch::empty({2}, opts.dtype(torch::kInt64));
    }

    if (max_seqlen_q > 0) {
        auto rng_state_ptr = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
        auto drop_seed_offset = std::make_pair(rng_state_ptr, rng_state_ptr + 1);
        ck_tile::stream_config stream_config{stream};

        auto args =
            get_ck_fmha_varlen_bwd_args(
                mask,
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads,
                num_heads_k,
                head_size_q,
                head_size_v,
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                alibi_slopes_,
                out,
                softmax_lse,
                dout,
                dq_accum,
                softmax_d,
                dq,
                dk_expanded,
                dv_expanded,
                softmax_scale,
                p_dropout,
                drop_seed_offset);

        float t = aiter::mha_bwd(args,
                                 stream_config,
                                 q_dtype_str,
                                 true,  //is_group_mode
                                 mask.type,
                                 bias_type,
                                 false,  // has_dbias
                                 false,  // is_store_randval
                                 deterministic,
                                 false,  // use_ext_asm
                                 false,  // is_v3_atomic_fp32
                                 0);     // how_v3_bf16_cvt
        TORCH_CHECK(t >= 0, "invalid argument for fmha_bwd");
    } else {
        // If seqlen_q == 0, then we have an empty tensor. We need to set the output to 0.
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        at::sum_out(dk, at::reshape(dk_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_q}), {2});
        at::sum_out(dv, at::reshape(dv_expanded, {total_k, num_heads_k, num_heads / num_heads_k, head_size_v}), {2});
    }

    return { dq, dk, dv, softmax_d };
}

} // namespace torch_itfs
} // namespace aiter
