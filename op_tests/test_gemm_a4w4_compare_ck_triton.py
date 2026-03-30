# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)


def shuffle_scales(scales: torch.Tensor):
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape
    scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled


@perftest(num_iters=5)
def run_torch(x, w, x_scales, w_scales, dtype):
    m, k = x.shape
    n, k = w.shape
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@benchmark()
def test_gemm(dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ["gfx950"]:
        return

    ret = {}
    flops = M * N * K * 2

    # --- Quantize inputs (shared by CK and Triton) ---
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((M, K), dtype=dtype)
    w = torch.randn((N, K), dtype=dtype)
    _, x_scales = quant_func(x, shuffle=False)
    _, w_scales = quant_func(w, shuffle=False)
    x, x_scales_shuffle = quant_func(x, shuffle=True)
    w, w_scales_shuffle = quant_func(w, shuffle=True)
    wshuffle = shuffle_weight(w, layout=(16, 16))
    x_scales = x_scales.view(torch.uint8)
    w_scales = w_scales.view(torch.uint8)

    # --- Torch reference ---
    ref, _ = run_torch(x, w, x_scales, w_scales, dtype)

    # --- CK W4A4 GEMM ---
    ck_out, ck_us = run_perftest(
        aiter.gemm_a4w4,
        x,
        wshuffle,
        x_scales_shuffle,
        w_scales_shuffle,
        bpreshuffle=True,
    )
    ck_err = checkAllclose(ref, ck_out, msg="CK a4w4")
    ret["ck us"] = ck_us
    ret["ck TFLOPS"] = flops / ck_us / 1e6
    ret["ck err"] = ck_err

    # --- Triton W4A4 GEMM (preshuffled) ---
    # Prepare preshuffled weight and scales for Triton kernel
    weight_shuffle_layout = (16, 16)
    w_triton = shuffle_weight(
        w, layout=weight_shuffle_layout, use_int4=False
    ).reshape(
        w.shape[0] // weight_shuffle_layout[0],
        w.shape[1] * weight_shuffle_layout[0],
    )
    w_scales_triton = shuffle_scales(w_scales)
    if M >= 32:
        x_scales_triton = shuffle_scales(x_scales)
    else:
        x_scales_triton = x_scales.contiguous()
    y_triton = torch.empty((M, N), dtype=dtype, device="cuda")

    triton_out, triton_us = run_perftest(
        gemm_afp4wfp4_preshuffle,
        x.view(torch.uint8),
        w_triton.view(torch.uint8),
        x_scales_triton.view(torch.uint8),
        w_scales_triton.view(torch.uint8),
        dtype,
        y_triton,
        use_aot=True,
    )
    if triton_out.dim() == 3:
        triton_out = triton_out.sum(dim=0).to(dtype)
    triton_err = checkAllclose(ref, triton_out, msg="Triton a4w4")
    ret["triton us"] = triton_us
    ret["triton TFLOPS"] = flops / triton_us / 1e6
    ret["triton err"] = triton_err

    # --- BF16 GEMM baseline ---
    x_bf16 = torch.randn((M, K), dtype=dtype, device="cuda")
    w_bf16 = torch.randn((N, K), dtype=dtype, device="cuda")
    _, bf16_us = run_perftest(F.linear, x_bf16, w_bf16)
    ret["bf16 us"] = bf16_us
    ret["bf16 TFLOPS"] = flops / bf16_us / 1e6

    # --- Speedups ---
    ret["ck vs bf16"] = bf16_us / ck_us
    ret["triton vs bf16"] = bf16_us / triton_us
    ret["ck vs triton"] = triton_us / ck_us

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Compare CK vs Triton vs BF16 GEMM performance",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    choices=[dtypes.d_dtypes["bf16"]],
    metavar="{bf16}",
    default=[dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
_default_csv = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "aiter", "configs", "a4w4_blockscale_untuned_gemm.csv",
)
_default_shapes = []
if os.path.exists(_default_csv):
    _df = pd.read_csv(_default_csv, skipinitialspace=True)
    _default_shapes = [
        (int(row["M"]), int(row["N"]), int(row["K"])) for _, row in _df.iterrows() if (int(row["M"]) <= 16) 
    ]

parser.add_argument(
    "-mnk",
    "--shape",
    type=dtypes.str2tuple,
    nargs="*",
    default=_default_shapes,
    help="""Shape of mnk.
    e.g. -mnk 1280,8192,1024""",
)

args = parser.parse_args()

df = []
for dtype in args.dtype:
    for m, n, k in args.shape:
        ret = test_gemm(dtype, m, n, k)
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("CK vs Triton vs BF16 summary (markdown):\n%s", df_md)
