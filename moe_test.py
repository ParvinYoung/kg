# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
"""Inference-only DeepseekV2 model."""

import logging
import os
from enum import IntEnum, auto
from typing import Any, Dict, Iterable, Optional, Tuple
from sglang.srt.layers.moe.fused_moe_triton import FPGAMoE
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    parallel_state,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.amx_utils import PackWeightMethod
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    enable_moe_dense_fully_dp,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    get_local_attention_dp_size,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE, get_moe_impl_class
from sglang.srt.layers.moe.ep_moe.token_dispatcher import DeepEPDispatcher
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization import deep_gemm_wrapper
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_kernel import (
    is_fp8_fnuz,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_dequant,
    block_quant_to_tensor_quant,
    channel_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
    requant_weight_ue8m0_inplace,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope, get_rope_wrapper
from sglang.srt.layers.utils import is_sm100_supported
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.two_batch_overlap import (
    MaybeTboDeepEPDispatcher,
    model_forward_maybe_tbo,
)
import json
from sglang.srt.utils import (
    BumpAllocator,
    DeepEPMode,
    LazyValue,
    add_prefix,
    bind_or_assign,
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    get_int_env_var,
    is_cpu,
    is_cuda,
    is_flashinfer_available,
    is_hip,
    is_non_idle_and_non_empty,
    log_info_on_rank0,
    use_intel_amx_backend,
)
from sglang.srt.layers.utils import get_layer_id, get_expert_id, get_proj_type
import sys
sys.path.append("~/sglang/fpga-kernel/build")

_is_hip = is_hip()
_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()

if _is_cuda:
    from sgl_kernel import (
        awq_dequantize,
        bmm_fp8,
        dsv3_fused_a_gemm,
        dsv3_router_gemm,
        merge_state_v2,
    )
elif _is_cpu and _is_cpu_amx_available:
    pass
else:
    from vllm._custom_ops import awq_dequantize

if _is_hip:
    from sglang.srt.layers.attention.triton_ops.rocm_mla_decode_rope import (
        decode_attention_fwd_grouped_rope,
    )

from fpga import (
    load_ds_moe_weights,
    ptr_to_capsule
)
_is_flashinfer_available = is_flashinfer_available()
_is_sm100_supported = is_cuda() and is_sm100_supported()


logger = logging.getLogger(__name__)


class AttnForwardMethod(IntEnum):
    # Use multi-head attention
    MHA = auto()

    # Use absorbed multi-latent attention
    MLA = auto()

    # Use multi-head attention, but with KV cache chunked.
    # This method can avoid OOM when prefix lengths are long.
    MHA_CHUNKED_KV = auto()

    # Use MLA but with fused RoPE
    MLA_FUSED_ROPE = auto()

    # Use MLA with fused RoPE kernel for CPU
    MLA_FUSED_ROPE_CPU = auto()


class DeepseekV2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size

        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=0,
            tp_size=1,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=0,
            tp_size=1,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x, forward_batch=None, can_fuse_mlp_allreduce=False):
        if (self.tp_size == 1) and x.shape[0] == 0:
            return x

        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x, can_fuse_mlp_allreduce=can_fuse_mlp_allreduce)
        return x


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
        is_nextn: bool = False,
    ):
        super().__init__()
        self.is_nextn = is_nextn
        self.weight = nn.Parameter(
            torch.empty((config.n_routed_experts, config.hidden_size))
        )

        if config.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty((config.n_routed_experts))
            )
        else:
            self.e_score_correction_bias = None
        if _is_cpu and _is_cpu_amx_available:
            self.quant_method = PackWeightMethod(weight_names=["weight"])

    def forward(self, hidden_states):
        if use_intel_amx_backend(self):
            return torch.ops.sgl_kernel.weight_packed_linear(
                hidden_states,
                self.weight,
                None,  # bias
                True,  # is_vnni
            )

        # NOTE: For some unknown reason, router_gemm seems degrade accept length.
        if (
            _is_cuda
            and not self.is_nextn
            and hidden_states.shape[0] < 4
            and hidden_states.shape[1] == 7168
            and self.weight.shape[0] == 256
            and _device_sm >= 90
        ):
            logits = dsv3_router_gemm(hidden_states, self.weight).to(
                hidden_states.dtype
            )
        else:
            logits = F.linear(hidden_states, self.weight, None)

        return logits


class DeepseekV2MoE(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ):
        super().__init__()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.num_fused_shared_experts = (
            0
        )
        self.config = config
        self.layer_id = layer_id

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(
            config=config, prefix=add_prefix("gate", prefix), is_nextn=is_nextn
        )

        # self.experts = None
        self.num_experts=config.n_routed_experts + self.num_fused_shared_experts + global_server_args_dict["ep_num_redundant_experts"]
        self.inter_dim=config.moe_intermediate_size
        self.dim= config.hidden_size

        self.shared_experts_is_int8 = False
        self.shared_experts_is_fp8 = False
        self.shared_experts_weight_block_size = None
        if config.n_shared_experts is not None and self.num_fused_shared_experts == 0:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            # disable tp for shared experts when enable deepep moe
            self.shared_experts = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if global_server_args_dict["enable_deepep_moe"]
                    else {}
                ),
            )
            is_packed_weight = hasattr(
                self.shared_experts.gate_up_proj.quant_method, "quant_config"
            ) and self.shared_experts.gate_up_proj.quant_method.quant_config.get_name() in {
                "awq",
                "moe_wna16",
            }
            self.shared_experts_is_int8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.int8
            )
            self.shared_experts_is_fp8 = (
                not is_packed_weight
                and self.shared_experts.gate_up_proj.weight.dtype == torch.float8_e4m3fn
            )
            if self.shared_experts_is_fp8:
                assert (
                    self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                    == self.shared_experts.down_proj.quant_method.quant_config.weight_block_size
                )
                self.shared_experts_weight_block_size = (
                    self.shared_experts.gate_up_proj.quant_method.quant_config.weight_block_size
                )

        self.top_k = config.num_experts_per_tok

        if global_server_args_dict["enable_deepep_moe"]:
            # TODO: we will support tp < ep in the future
            self.ep_size = get_tensor_model_parallel_world_size()
            self.num_experts = (
                config.n_routed_experts
                + global_server_args_dict["ep_num_redundant_experts"]
            )
            self.renormalize = config.norm_topk_prob
            self.topk_group = config.topk_group
            self.num_expert_group = config.n_group
            self.correction_bias = (
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            )

            self.deepep_dispatcher = MaybeTboDeepEPDispatcher(
                group=parallel_state.get_tp_group().device_group,
                router_topk=self.top_k,
                permute_fusion=True,
                num_experts=self.num_experts,
                num_local_experts=config.n_routed_experts,
                hidden_size=config.hidden_size,
                params_dtype=config.torch_dtype,
                deepep_mode=DeepEPMode[global_server_args_dict["deepep_mode"]],
                async_finish=True,
                return_recv_hook=True,
            )

        self._enable_deepep_moe = global_server_args_dict["enable_deepep_moe"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        can_fuse_mlp_allreduce: bool = False,
    ) -> torch.Tensor:
        return self.forward_normal(hidden_states, can_fuse_mlp_allreduce)

    def forward_normal(
        self, hidden_states: torch.Tensor, can_fuse_mlp_allreduce: bool = False
    ) -> torch.Tensor:
        if hasattr(self, "shared_experts") and use_intel_amx_backend(
            self.shared_experts.gate_up_proj
        ):
            return self.forward_cpu(hidden_states)

        #shared_output = self._forward_shared_experts(hidden_states.to(dtype=torch.float8_e4m3fn))
        #print("shared_output", shared_output.shape if shared_output is not None else None)
        shared_output = None
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_weights, topk_idx = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                use_grouped_topk=True,
                top_k=self.config.num_experts_per_tok + self.num_fused_shared_experts,
                renormalize=self.config.norm_topk_prob,
                topk_group=self.config.topk_group,
                num_expert_group=self.config.n_group,
                num_fused_shared_experts=self.num_fused_shared_experts,
                custom_routing_function=None,
                correction_bias=self.gate.e_score_correction_bias,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        import time
        start = time.time()
        final_hidden_states = FPGAMoE(hidden_states, topk_weights, topk_idx, self.layer_id, self.num_experts, inter_dim=self.inter_dim, dim=self.dim)
        end = time.time()
        print("fpga moe time:", end - start)
        
        if not _is_cuda and not _use_aiter:
            # fused in biased_grouped_topk so we can skip here
            final_hidden_states *= self.routed_scaling_factor
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states
    
    def _forward_shared_experts(self, hidden_states):
        if self.num_fused_shared_experts == 0:
            return self.shared_experts(hidden_states)
        else:
            return None
        
    def load_weights(self, weights: Dict[str, torch.Tensor]):
        #print(weights.keys())
        # if self.num_fused_shared_experts == 0:
        #     self.shared_experts.load_state_dict(
        #     {
        #         "gate_up_proj.weight": weights["model.layers.3.mlp.shared_experts.up_proj.weight"],
        #         "down_proj.weight": weights["model.layers.3.mlp.shared_experts.down_proj.weight"],
        #     }
        # )
        self.gate.load_state_dict({
            "weight": weights["model.layers.3.mlp.gate.weight"],
            "e_score_correction_bias": weights["model.layers.3.mlp.gate.e_score_correction_bias"]
            })
        
        for i in range(256):
            weight_tensor = weights[f"model.layers.3.mlp.experts.{i}.up_proj.weight"]
            scale_tensor = weights[f"model.layers.3.mlp.experts.{i}.up_proj.weight_scale_inv"]
            weight_ptr = ptr_to_capsule(weight_tensor.data_ptr())
            scale_ptr = ptr_to_capsule(scale_tensor.data_ptr())
            load_ds_moe_weights(weight_ptr,scale_ptr, 3, i, 0)
            
            weight_tensor = weights[f"model.layers.3.mlp.experts.{i}.down_proj.weight"]
            scale_tensor = weights[f"model.layers.3.mlp.experts.{i}.down_proj.weight_scale_inv"]
            load_ds_moe_weights(weight_ptr,scale_ptr, 3, i, 2)
            
            weight_tensor = weights[f"model.layers.3.mlp.experts.{i}.gate_proj.weight"]
            scale_tensor = weights[f"model.layers.3.mlp.experts.{i}.gate_proj.weight_scale_inv"]
            load_ds_moe_weights(weight_ptr,scale_ptr, 3, i, 1)
        
def load_configs():
    with open(f"config.json", "r") as f:
        config_dict = json.load(f)
    config = PretrainedConfig.from_dict(config_dict)

    # with open(f"quant_config.json", "r") as f:
    #     quant_dict = json.load(f)
    # if quant_dict is not None:
    #     quant_config = QuantizationConfig.get_from_keys(quant_dict)
    # else:
    #     quant_config = None

    return config
    


if __name__ == "__main__":
    config = load_configs()
    
    model = DeepseekV2MoE(config=config,
                #quant_config=quant_config,
                layer_id=3).cuda()
    
    from safetensors.torch import load_file
    import torch

    state_dict = {}

    for i in range(1,5):
        shard_path = f"/data/models/DeepSeek-R1-0528/model-0000{i}-of-000163.safetensors"    
        shard_state = load_file(shard_path)
        state_dict.update(shard_state) 

    model.load_weights(state_dict)
    
    hidden_states = torch.randn(4, 7168, device='cuda')
    for _ in range(10):
        model(hidden_states)
    
    torch.cuda.synchronize()
    #hidden_states = hidden_states.to(dtype=torch.float8_e4m3fn)
    g = torch.cuda.CUDAGraph()  # 创建图对象
    #model(hidden_states)
    with torch.cuda.graph(g):
        model(hidden_states)

    g.replay()