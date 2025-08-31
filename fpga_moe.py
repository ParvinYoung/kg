from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
    moe_sum_reduce_torch_compile, 
    moe_sum_reduce_triton,
)
import torch
from typing import List, Optional
import os
import sys
import time
sys.path.append("~/sglang/fpga-kernel/build")
from fpga import (
    launch_moe_single_expert, 
    wait, 
    copy_h2d, 
    get_input,
    get_scale, 
    get_weight, 
    ptr_to_capsule
)
from sglang.srt.utils import (
    is_cuda,
    is_hip,
)

import ctypes
import pybind11
from sglang.srt.layers.quantization.fp8_kernel import (
    per_token_group_quant_fp8,
    scaled_fp8_quant,
    sglang_per_token_group_quant_fp8,
)

_is_hip = is_hip()
_is_cuda = is_cuda()

padding_size = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0

if _is_cuda or _is_hip:
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

if _is_cuda:
    from sgl_kernel import (
        sgl_per_tensor_quant_fp8,
        sgl_per_token_group_quant_fp8,
        sgl_per_token_quant_fp8,
    )

enable_moe_align_block_size_triton = bool(
    int(os.getenv("ENABLE_MOE_ALIGN_BLOCK_SIZE_TRITON", "0"))
)

def FPGAMoE(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layerid: int,
    E : int, 
    inplace: bool = False,
    use_fp8_w8a8: bool = True,
    use_int4_w4a16: bool = False,
    inter_dim : int = 2048,
    dim : int = 7168,
    block_shape: Optional[List[int]] = [1, 128],
    per_channel_quant: bool = True,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
):

    # Check constraints.
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens, _ = hidden_states.shape
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 64 * 1024
    M = min(num_tokens, CHUNK_SIZE)


    intermediate_cache3 = torch.empty(
        M * topk_ids.shape[1] * dim,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    ).view(
        (M, topk_ids.shape[1], dim),
    )

    fpga_result_on_cpu = torch.empty(
                (M*topk_ids.shape[1], dim),
                device="cpu",
                dtype=hidden_states.dtype,
            )

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk_ids.shape[1], dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif topk_ids.shape[1] == 1:
        out_hidden_states = torch.empty(
            (num_tokens, topk_ids.shape[1], dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    else:
        out_hidden_states = torch.empty_like(hidden_states)
    
    input_ptr = get_input()
    input_scale_ptr = get_scale()
    input_weight_ptr = get_weight()

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            fpga_result_on_cpu = fpga_result_on_cpu[:tokens_in_chunk]

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        
        counts = torch.bincount(curr_topk_ids.flatten(), minlength=E).tolist()
        count = 0

        for i in range(E):
            if counts[i] == 0:
                continue

            idx, top = torch.where(curr_topk_ids == i)
            batch = len(idx)
            if use_fp8_w8a8:
                if block_shape is None:
                    # activations apply per-token quantization when weights apply per-channel quantization by default
                    input_scaled, scale = scaled_fp8_quant(
                        curr_hidden_states[idx], use_per_token_if_dynamic=per_channel_quant
                    )
                else:
                    # activation block-wise fp8 quantization
                    assert len(block_shape) == 2
                    _, block_k = block_shape[0], block_shape[1]
                    if _is_cuda:
                        input_scaled, scale = sglang_per_token_group_quant_fp8(curr_hidden_states[idx], block_k)
                    else:
                        input_scaled, scale = per_token_group_quant_fp8(curr_hidden_states[idx], block_k)
            else:            
                input_scaled, scale = curr_hidden_states[idx], torch.ones(batch, device=hidden_states.device, dtype=torch.float32)
            if scale.shape[0] != batch:
                scale = scale.repeat(batch).view(-1, 1)
            temp_out = torch.empty((batch, dim), device="cpu", dtype=hidden_states.dtype)
            curr_cpp_hidden = ptr_to_capsule(input_scaled.contiguous().cpu().data_ptr())
            curr_cpp_scale = ptr_to_capsule(scale.contiguous().cpu().data_ptr())
            curr_cpp_weight = ptr_to_capsule(curr_topk_weights[idx, top].contiguous().cpu().data_ptr())
            curr_cpp_out = ptr_to_capsule(temp_out.data_ptr())

            copy_h2d(curr_cpp_hidden, input_ptr, batch*dim)
            copy_h2d(curr_cpp_scale, input_scale_ptr, batch)
            copy_h2d(curr_cpp_weight, input_weight_ptr, batch)

            launch_moe_single_expert(
                input_ptr,
                input_scale_ptr,
                input_weight_ptr,
                batch,
                layerid,
                i,
                curr_cpp_out
            )
            count += 1
            fpga_result_on_cpu[idx.cpu()] = temp_out

        # while True:
        #     if wait() == count:
        #         break
        #     time.sleep(0.01)
        #get_tensor(batch, curr_cpp_out).copy_(fpga_result_on_cpu[:, count, :])

        if not no_combine and topk_ids.shape[1] != 1:
            intermediate_cache3 = fpga_result_on_cpu.view(M, topk_ids.shape[1], dim).to(device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            out_hidden_states[begin_chunk_idx:end_chunk_idx] = fpga_result_on_cpu.view(M, topk_ids.shape[1], dim).to(device=hidden_states.device, dtype=hidden_states.dtype)

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0].to(device=hidden_states.device),
                    intermediate_cache3[:, 1].to(device=hidden_states.device),
                    out=out_hidden_states[begin_chunk_idx:end_chunk_idx],
                ).squeeze(dim=1)
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if tokens_in_chunk <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape).to(device=hidden_states.device),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape).to(device=hidden_states.device),
                        out_hidden_states[begin_chunk_idx:end_chunk_idx],
                        routed_scaling_factor,
                    )

    return out_hidden_states