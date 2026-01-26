# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest
from aiter.utility import fp4_utils

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_GROUP_SIZE = 32
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)


@perftest(num_iters=5)
def run_torch(x, w, x_scales, w_scales, dtype):
    m, k = x.shape
    n, k = w.shape
    # First convert the x and w inputs to f32.
    x_f32 = fp4_utils.mxfp4_to_f32(x)
    w_f32 = fp4_utils.mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales[:m]
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    x_scales_f32 = fp4_utils.e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales[:n]
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1)
    w_scales_f32 = fp4_utils.e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)[:m, :n]


@perftest()
def run_gemm_ck(x, weight, x_scale, w_scale, out):
    return aiter.gemm_a4w4_blockscale(x, weight, x_scale, w_scale, out)


@perftest()
def run_triton(x, w, x_scales, w_scales, out, dtype=dtypes.bf16):
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4

    gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, out)
    return out


@perftest()
def run_gemm_asm(
    x,
    weightshuffle,
    x_scale,
    w_scale,
    out,
    kernelName="",
    bias=None,
    dtype=dtypes.bf16,
    bpreshuffle=True,
    log2_k_split=None,
):
    # if log2_k_split is not None and log2_k_split > 0:
    #     out_reset = torch.zeros(
    #         (out.shape[0] + 31) // 32 * 32, out.shape[1], dtype=dtype
    #     )
    #     out = out_reset

    aiter.gemm_a4w4_asm(
        x,
        weightshuffle,
        x_scale,
        w_scale,
        out,
        kernelName,
        bias,
        bpreshuffle=bpreshuffle,
        log2_k_split=log2_k_split,
    )
    return out


@benchmark()
def test_gemm(dtype, M, N, K):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ["gfx950"]:
        return
    ret = {}
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
    a, avg_a = run_torch(x, w, x_scales, w_scales, dtype)
    # out1 = torch.empty(M, N, dtype=dtype)
    # b, avg_b = run_triton(x, w.T, x_scales, w_scales, out1, dtype)
    # b, avg_b = a, 0
    # err_b = checkAllclose(a, b, msg="triton        ")

    c, us = run_perftest(
        aiter.gemm_a4w4,
        x,
        wshuffle,
        x_scales_shuffle,
        w_scales_shuffle,
        bpreshuffle=True,
    )
    err = checkAllclose(a, c, msg="unified api")
    ret["us"] = us
    ret["TFLOPS"] = M * N * K * 2 / us / 1e6
    ret["TB/s"] = (x.nbytes + w.nbytes) / us / 1e6
    ret["err"] = err

    # kernelName = "" # "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"
    # log2_k_split = 1
    # out2 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    # d, us = run_gemm_asm(
    #     x,
    #     wshuffle,
    #     x_scales_shuffle,
    #     w_scales_shuffle,
    #     out2,
    #     kernelName,
    #     bias_f32,
    #     bpreshuffle=True,
    #     log2_k_split=log2_k_split,
    # )
    # err = checkAllclose(a, d[:M], msg=f"asm {kernelName} log2_k_split_{log2_k_split}")
    # tag = "asm_dbg"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    # out3 = torch.empty((M + 31) // 32 * 32, N, dtype=dtype)
    # e, us = run_gemm_ck(x, wshuffle, x_scales_shuffle, w_scales_shuffle, out3)
    # err = checkAllclose(a, e[:M], msg="ck            ")
    # tag = "ck"
    # ret[f"us {tag}"] = us
    # ret[f"TFLOPS {tag}"] = M * N * K * 2 / us / 1e6
    # ret[f"TB/s {tag}"] = (x.nbytes + w.nbytes) / us / 1e6
    # ret[f"err {tag}"] = err

    return ret


l_dtype = ["bf16"]

# Load GEMM shapes from CSV file
import os
csv_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "aiter/configs/a4w4_blockscale_untuned_gemm.csv"
)
df_shapes = pd.read_csv(csv_path)
l_mnk = [(row['M'], row['N'], row['K']) for _, row in df_shapes.iterrows()]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-mnk",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Shape of mnk.
    e.g. -mnk 1280,8192,1024""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.shape is not None:
    l_mnk = [args.shape]

df = []
for dtype in l_dtype:
    for m, n, k in l_mnk:
        ret = test_gemm(dtype, m, n, k)
        df.append(ret)
df = pd.DataFrame(df)
aiter.logger.info(f"summary:\n{df}")
