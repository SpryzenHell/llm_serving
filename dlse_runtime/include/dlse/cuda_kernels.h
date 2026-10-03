#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

namespace dlse {

cudaError_t launch_paged_attention_decode(
    const void* q,
    const void* k_pages,
    const void* v_pages,
    const std::int32_t* block_table,
    const std::int32_t* seq_lens,
    void* output,
    int batch,
    int q_heads,
    int kv_heads,
    int head_dim,
    int page_size,
    int max_blocks_per_seq,
    int max_seq_len,
    cudaStream_t stream);

cudaError_t launch_contiguous_attention_decode(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const std::int32_t* seq_lens,
    void* output,
    int batch,
    int q_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len,
    cudaStream_t stream);

}
