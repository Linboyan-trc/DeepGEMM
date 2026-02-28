#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../heuristics/sm90.hpp"
#include "../heuristics/sm100.hpp"
#include "runtime_utils.hpp"

#include <iostream>

namespace deep_gemm {

// 1. SMXXFP8MQALogitsRuntime类
// 1.1 继承自父类LaunchRuntime<SMXXFP8MQALogitsRuntime>
class SMXXFP8MQALogitsRuntime final: public LaunchRuntime<SMXXFP8MQALogitsRuntime> {
public:
    struct Args {
        int seq_len;
        int seq_len_kv;
        int max_seqlen_k;
        int stride_logits;
        int num_heads, head_dim;
        bool is_compressed_logits;

        int num_q_stages;
        int num_kv_stages;
        int block_q;
        int block_kv;

        int* cu_seq_len_k_start;
        int* cu_seq_len_k_end;
        float* logits;
        float softmax_scale;

        CUtensorMap tensor_map_q;
        CUtensorMap tensor_map_kv;
        CUtensorMap tensor_map_kv_scales;
        CUtensorMap tensor_map_weights;

        int num_specialized_threads;
        int num_math_threads;

        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        // TODO: optimize performance by tuning args
        // Block sizes are fixed in this kernel
        DG_HOST_ASSERT(128 % args.num_heads == 0);
        const auto& arch = device_runtime->get_arch(true);

        return fmt::format(R"(
#include <deep_gemm/impls/sm{}_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm{}_fp8_mqa_logits<
        {}, {},
        {},
        {}, {},
        {}, {},
        {}, {}
    >);
}};
)", arch, arch,
    args.num_heads, args.head_dim,
    args.is_compressed_logits,
    args.block_q, args.block_kv,
    args.num_q_stages, args.num_kv_stages,
    args.num_specialized_threads, args.num_math_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.seq_len, args.seq_len_kv,
            args.max_seqlen_k, static_cast<int64_t>(args.stride_logits),
            args.cu_seq_len_k_start, args.cu_seq_len_k_end,
            args.logits,
            args.tensor_map_q, args.tensor_map_kv,
            args.tensor_map_kv_scales, args.tensor_map_weights
        ));
    }
}; 

// 1. fp8_mqa_logits()算子
// 1.1 参数:                  q.shape = [2048, 64, 128]
// 1.1 参数:                 kv.shape = [4096, 128]
// 1.1 参数:          kv_scales.shape = [4096]
// 1.1 参数:            weights.shape = [2048, 64]
// 1.1 参数: cu_seq_len_k_start.shape = [2048]，元素值为0
// 1.1 参数:   cu_seq_len_k_end.shape = [2048]，元素值为2048, 2049, ... 4095
// 1.2 参数:             logits.shape = [2048, 4096]
// 1.2 参数: seq_len            = 2048,  seq_len_kv = 4096,     max_seqlen_k = 0
// 1.2 参数: stride_logits      = 4352,  num_heads  = 64,       head_dim     = 128,
// 1.2 参数:  seq_len_alignment = 2048
// 1.3 无返回值
static void smxx_fp8_mqa_logits(
    const torch::Tensor& q,
    const torch::Tensor& kv, 
    const torch::Tensor& kv_scales,
    const torch::Tensor& weights,
    const torch::Tensor& cu_seq_len_k_start,
    const torch::Tensor& cu_seq_len_k_end,

    const torch::Tensor& logits,
    const int& seq_len, 
    const int& seq_len_kv,
    const int& max_seqlen_k, 
    const int& stride_logits,
    const int& num_heads, 
    const int& head_dim,
    const int& seq_len_alignment
) {
    // 2. 一个block计算128个q和256个kv
    // 2.1 128个线程专门负责搬数据
    // 2.2 Q和KV都用3级流水线缓存
    // 2.3 SM90用512个算数线程
    // 2.4 每个block一次处理同一个head的2个q token，一共64头，一共128个token
    constexpr int block_qh = 128;
    constexpr int block_kv = 256;
    constexpr int num_specialized_threads = 128;
    constexpr int num_q_stages = 3, num_kv_stages = 3;
    const int num_math_threads = (device_runtime->get_arch_major() == 10 ? 256 : 512);
    const int block_q = block_qh / num_heads;
    DG_HOST_ASSERT(block_qh % num_heads == 0);
    DG_HOST_ASSERT(seq_len_alignment % block_q == 0);

    // 3. is_compressed_logits = false
    const bool is_compressed_logits = (max_seqlen_k > 0);

    // 4. 维度是128
    DG_HOST_ASSERT(head_dim == 32 or head_dim == 64 or head_dim == 128);


    // 5. 搬运q张量
    // 5.1 q 是一个 131072行 × 128列 的矩阵
    // 5.2 每次搬 128 行
    const auto& tensor_map_q = make_tma_2d_desc(
        q,                      // 5.1 q.shape = [2048, 64, 128]
        head_dim,               // 5.2 128
        seq_len * num_heads,    // 5.3 2048 * 64
        head_dim,               // 5.4 128
        block_qh,               // 5.5 128
        head_dim,               // 5.6 128
        head_dim                // 5.7 128
    );

    // 6. 搬运kv张量
    // 6.1 kv 是一个 4096行 × 128列 的矩阵
    // 6.2 每次搬 256 行
    const auto& tensor_map_kv = make_tma_2d_desc(
        kv,                     // 6.1 kv.shape = [4096, 128]
        head_dim,               // 6.2 128
        seq_len_kv,             // 6.3 4096
        head_dim,               // 6.4 128
        block_kv,               // 6.5 256
        head_dim,               // 6.6 128
        head_dim                // 6.7 128
    );

    // According to the driver API, the minimal alignment is 256 bytes
    // So it is safe for us to do a 16-byte OOB
    // 7. 搬运kv_scales张量
    // 7.1 kv_scales 是一个 4096行 × 1列 的矩阵
    // 7.2 每次搬 256 行
    const auto& tensor_map_kv_scales = make_tma_2d_desc(
        kv_scales,                                                                      // 7.1 kv_scales.shape = [4096]
        get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),   // 7.2 ???
        1,                                                                              // 7.3 1
        block_kv,                                                                       // 7.4 256
        1,                                                                              // 7.5 1
        0,                                                                              // 7.6 0
        0                                                                               // 7.7 0
    );

    // 8. 搬运weights张量
    const auto& tensor_map_weights = make_tma_2d_desc(
        weights,
        num_heads, 
        seq_len,
        num_heads, 
        block_q, 
        num_heads, 
        0
    );

    // Calculate shared memory size
    int smem_size = 0;
    const int smem_q_size_per_stage = block_q * num_heads * head_dim * static_cast<int>(q.element_size());
    const int smem_weight_size_per_stage = block_q * num_heads * static_cast<int>(weights.element_size());
    const int smem_kv_size_per_stage = block_kv * head_dim * static_cast<int>(kv.element_size());
    const int kv_scale_size_per_stage = block_kv * static_cast<int>(kv_scales.element_size());
    smem_size += num_q_stages * smem_q_size_per_stage;
    smem_size += num_kv_stages * smem_kv_size_per_stage;
    smem_size += num_q_stages * smem_weight_size_per_stage;
    smem_size += num_kv_stages * kv_scale_size_per_stage;
    smem_size += (num_q_stages * 2 + num_kv_stages * 2 + (num_math_threads / 128) * 2) * 8;
    smem_size += 4;
    DG_HOST_ASSERT(smem_size <= SM90ArchSpec::smem_capacity);
    DG_HOST_ASSERT(smem_size <= SM100ArchSpec::smem_capacity);

    // Launch
    // 9. Kernel参数
    // 9.1 seq_len              = 2048
    // 9.2 seq_len_kv           = 4096
    // 9.3 max_seqlen_k         = 0
    // 9.4 stride_logits        = 4352
    // 9.5 num_heads            = 64
    // 9.6 head_dim             = 128
    // 9.7 is_compressed_logits = false, 0
    // 9.8 num_q_stages         = 3
    // 9.9 num_kv_stages        = 3
    // 9.10 block_q             = 2
    // 9.11 block_kv            = 256
    // 9.12 num_specialized_threads = 128
    // 9.13 num_math_threads    = 512
    // 9.14 launch_args = LaunchArgs(132, 640, 152228)

    // 9.11 args.cu_seq_len_k_start是一个一维数组指针，指向cu_seq_len_k_start.shape = [2048]
    // 9.12 args.cu_seq_len_k_end是一个一维数组指针，指向cu_seq_len_k_end.shape = [2048]
    // 9.13 args.logits是一个二维数组指针，指向logits.shape = [2048, 4096]
    // 9.21 args.tensor_map_q是CUtensorMap描述符，是一个描述如何搬运张量的信息体

    std::cout << "11. prepare kernel args" << std::endl;
    std::cout << seq_len << std::endl;
    std::cout << seq_len_kv << std::endl;
    std::cout << max_seqlen_k << std::endl;
    std::cout << stride_logits << std::endl;
    std::cout << num_heads << std::endl;
    std::cout << head_dim << std::endl;
    std::cout << is_compressed_logits << std::endl;
    std::cout << num_q_stages << std::endl;
    std::cout << num_kv_stages << std::endl;
    std::cout << block_q << std::endl;
    std::cout << block_kv << std::endl;
    std::cout << num_specialized_threads << std::endl;
    std::cout << num_math_threads << std::endl;
    std::cout << device_runtime->get_num_sms() << std::endl;
    std::cout << num_specialized_threads + num_math_threads << std::endl;
    std::cout << smem_size << std::endl;
    const SMXXFP8MQALogitsRuntime::Args& args = {
        .seq_len = seq_len,
        .seq_len_kv = seq_len_kv,
        .max_seqlen_k = max_seqlen_k,
        .stride_logits = stride_logits,
        .num_heads = num_heads, 
        .head_dim = head_dim,
        .is_compressed_logits = is_compressed_logits,
        .num_q_stages = num_q_stages,
        .num_kv_stages = num_kv_stages,
        .block_q = block_q,
        .block_kv = block_kv,
        .cu_seq_len_k_start = cu_seq_len_k_start.data_ptr<int>(),
        .cu_seq_len_k_end = cu_seq_len_k_end.data_ptr<int>(),
        .logits = logits.data_ptr<float>(),
        .tensor_map_q = tensor_map_q,
        .tensor_map_kv = tensor_map_kv,
        .tensor_map_kv_scales = tensor_map_kv_scales,
        .tensor_map_weights = tensor_map_weights,
        .num_specialized_threads = num_specialized_threads,
        .num_math_threads = num_math_threads,
        .launch_args = LaunchArgs(
            device_runtime->get_num_sms(),
            num_specialized_threads + num_math_threads,
            smem_size
        )
    };

    std::cout << "12. generate kernel" << std::endl;
    const auto& code = SMXXFP8MQALogitsRuntime::generate(args);
    const auto& runtime = compiler->build("smxx_fp8_mqa_logits", code);

    std::cout << "13. launch kernel" << std::endl;
    SMXXFP8MQALogitsRuntime::launch(runtime, args);
}

} // namespace deep_gemm
