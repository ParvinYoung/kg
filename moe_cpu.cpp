#include <unordered_map>
#include <cstdint>
#include <torch/torch.h>
#include <torch/extension.h>
#include "moe_cpu.h"


#define MAX_SEQ_LEN 4096
#define MAX_BATCH   4



namespace infiniai::fpga {

std::unordered_map<int, at::Tensor> weights;
std::unordered_map<int, void*> mem_map;

int finish_count = 0;

at::Tensor dequantize_fp8e4m3_to_fp32(const at::Tensor& fp8_tensor, const at::Tensor& scales) {
    TORCH_CHECK(fp8_tensor.scalar_type() == torch::kFloat8_e4m3fn, 
               "Input tensor must be FP8 E4M3 format");

    auto fp32_tensor = fp8_tensor.to(torch::kFloat32);
    auto group_size = fp8_tensor.size(1) / scales.size(1); // should be 128

    auto expanded_scales = scales.repeat_interleave(group_size, 0).repeat_interleave(group_size, 1);
    auto res = fp32_tensor * expanded_scales;
    return res;
}

// add 1 to the result if you want to get scale instead of weight
int get_key(int layer_id, int expert_id, int weight_type) {
    return layer_id * 100000 + expert_id * 100 + weight_type * 10;
}

int load_ds_moe_weights(
    void* weight,           // fp8 weight, cpu
    void* weight_scale,     // [16, 56] for up/gate, [56, 16] for down,
    const int layer_id,
    const int expert_id,
    const int weight_type
) {
    auto key = get_key(layer_id, expert_id, weight_type);
    at::Tensor tensor = torch::from_blob(
        weight, 
        weight_type == 2 ? std::initializer_list<int64_t>{7168, 2048} : std::initializer_list<int64_t>{2048, 7168},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn)
    ).clone();
    at::Tensor scale = torch::from_blob(
        weight_scale, 
        weight_type == 2 ? std::initializer_list<int64_t>{56, 16} : std::initializer_list<int64_t>{16, 56},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).clone();
    // const bool load_with_fp8 = std::getenv("LOAD_WITH_FP8") != nullptr;
    // if (load_with_fp8) {
    //     weights[key] = tensor;
    //     weights[key + 1] = scale;
    // } else {
    //     weights[key] = dequantize_fp8e4m3_to_fp32(tensor, scale);
    // }
    weights[key] = tensor;
    weights[key + 1] = scale;

    return 0;
}

int launch_moe_single_expert(
    void* input,          // fp8,  [bs, 7168], attention output
    void* input_scale,    // fp32, [bs,], scale used to dequantize fp8 input
    void* input_weights,  // fp32, [bs]
    const int batch_size,
    const int layer_id,
    const int expert_id,
    void* output          // bf16 or fp16,     [bs, 7168]
) {
    // allocate cpu mem for output buffer and register addr with expert id
    
    // const bool load_with_fp8 = std::getenv("LOAD_WITH_FP8") != nullptr;
    bool load_with_fp8 = true;
    at::Tensor mlp_up;
    at::Tensor mlp_gate;
    at::Tensor mlp_down;
    if (load_with_fp8) {
        at::Tensor mlp_up_fp8     = weights[get_key(layer_id, expert_id, 0)].cpu();       // [2048, 7168], fp32
        at::Tensor mlp_gate_fp8   = weights[get_key(layer_id, expert_id, 1)].cpu();       // [2048, 7168], fp32
        at::Tensor mlp_down_fp8   = weights[get_key(layer_id, expert_id, 2)].cpu();       // [7168, 2048], fp32
        at::Tensor mlp_up_scale   = weights[get_key(layer_id, expert_id, 0) + 1].cpu();   // [16, 56], fp32
        at::Tensor mlp_gate_scale = weights[get_key(layer_id, expert_id, 1) + 1].cpu();   // [16, 56], fp32
        at::Tensor mlp_down_scale = weights[get_key(layer_id, expert_id, 2) + 1].cpu();   // [16, 56], fp32
        mlp_up = dequantize_fp8e4m3_to_fp32(mlp_up_fp8, mlp_up_scale);
        mlp_gate = dequantize_fp8e4m3_to_fp32(mlp_gate_fp8, mlp_gate_scale);
        mlp_down = dequantize_fp8e4m3_to_fp32(mlp_down_fp8, mlp_down_scale);
    } else {
        mlp_up     = weights[get_key(layer_id, expert_id, 0)].cpu();       // [2048, 7168], fp32
        mlp_gate   = weights[get_key(layer_id, expert_id, 1)].cpu();       // [2048, 7168], fp32
        mlp_down   = weights[get_key(layer_id, expert_id, 2)].cpu();       // [7168, 2048], fp32
    }

    // int dim_in = 7168;
    // int dim_hidden = 2048;

    // at::Tensor mlp_up     = torch::randn({dim_hidden, dim_in}, torch::kFloat32);
    // at::Tensor mlp_gate   = torch::randn({dim_hidden, dim_in}, torch::kFloat32);
    // at::Tensor mlp_down   = torch::randn({dim_in, dim_hidden}, torch::kFloat32);

    // Read Tensor from input pointer
    at::Tensor attn_out = torch::from_blob(
        input, 
        std::initializer_list<int64_t>{batch_size, 7168},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn)
    ).cpu();

    at::Tensor scales = torch::from_blob(
        input_scale, 
        std::initializer_list<int64_t>{batch_size, 56},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).cpu();

    at::Tensor expert_weights = torch::from_blob(
        input_weights, 
        std::initializer_list<int64_t>{batch_size},
        torch::TensorOptions().dtype(torch::kFloat32)
    ).cpu();

    // Cast FP8 to FP32
    auto attn_out_fp32 = attn_out.to(torch::kFloat32);
    auto expanded_scales = scales.repeat_interleave(128, 1);
    attn_out_fp32 =  attn_out_fp32 * expanded_scales;

    // calculate

    auto x = attn_out_fp32.contiguous();
    auto up_proj = torch::mm(x, mlp_up.t());
    
    auto gate_proj = torch::mm(x, mlp_gate.t());
    auto hidden_states = up_proj * torch::silu(gate_proj);

    auto expert_output = torch::mm(hidden_states, mlp_down.t()); 
    expert_weights = expert_weights.view({-1, 1});
    expert_output = (expert_output * expert_weights).to(torch::kBFloat16);
    // copy result to output buffer
    memcpy(output, expert_output.data_ptr(), expert_output.numel() * expert_output.element_size());
    finish_count++;
    return 0;

}

void* get_input() {
    size_t total_size = MAX_BATCH * MAX_SEQ_LEN * 7168 * sizeof(unsigned char);  // [bs = 4, seq = 4096, 7168]
    
    void* fp8_buffer = malloc(total_size);
    if (fp8_buffer == NULL) {
        return NULL;
    }
    memset(fp8_buffer, 0, total_size);

    // return dummy fpga address
    return fp8_buffer;
}

void* get_scale() {
    size_t total_size = MAX_BATCH * MAX_SEQ_LEN * sizeof(unsigned char);
    void* buffer = malloc(total_size);
    memset(buffer, 0, total_size);
    return buffer;
}

void* get_weight() {
    size_t total_size = MAX_BATCH * MAX_SEQ_LEN * sizeof(unsigned char);
    void* buffer = malloc(total_size);
    memset(buffer, 0, total_size);
    return buffer;
}

// copy N bytes from cpu address to dummy fpga address
void copy_h2d(void* h_addr, void* d_addr, size_t size) {
    memcpy(d_addr, h_addr, size);
}

int wait() {
    int res = finish_count;
    finish_count = 0;
    return res;
}

py::capsule ptr_to_capsule(uint64_t ptr) {
    return py::capsule(reinterpret_cast<void*>(ptr));
}

}
PYBIND11_MODULE(fpga, m) {
    m.def("load_ds_moe_weights", &infiniai::fpga::load_ds_moe_weights, "Load DS MoE weights");
    m.def("launch_moe_single_expert", &infiniai::fpga::launch_moe_single_expert, "Launch MoE single expert");
    m.def("get_input", &infiniai::fpga::get_input, "Get input buffer");
    m.def("get_scale", &infiniai::fpga::get_scale, "Get scale buffer");
    m.def("get_weight", &infiniai::fpga::get_weight, "Get route weight buffer");
    m.def("copy_h2d", &infiniai::fpga::copy_h2d, "Copy data from host to device");
    m.def("wait", &infiniai::fpga::wait, "Wait for completion");
    m.def("ptr_to_capsule", &infiniai::fpga::ptr_to_capsule, "Convert pointer to capsule");
}
