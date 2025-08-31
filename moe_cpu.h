#pragma once
#include <ATen/core/Tensor.h>

#define NUM_EXPERTS 256
#define NUM_LAYERS  61

namespace infiniai::fpga {

int load_ds_moe_weights(
    void* weight,         // fp8 e4m3, [2048, 7168] for up/gate, [7168, 2048] for down
    void* weight_scale,   // fp32,     [2048] for up/gate,       [7168] for down
    const int layer_id,
    const int expert_id,
    const int mlp_type         // 0 for up, 1 for gate, 2 for down
);

int launch_moe_single_expert(
    void* input,          // fp8,     [bs, 7168]
    void* input_scale,    // fp32,    [bs,], scale used to dequantize fp8 input
    void* input_weights,  // fp32,     [bs]
    const int batch_size,
    const int layer_id,
    const int expert_id,
    void* output          // fp32,     [bs, 7168]
);

int wait();

// get input fpga address;
void* get_input();
void* get_scale();
void* get_weight();

// copy from cpu to fpga
void copy_h2d(void* h_addr, void* d_addr, size_t size);

py::capsule ptr_to_capsule(uint64_t ptr);
}
