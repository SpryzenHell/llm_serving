#include "dlse/cuda_kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>

namespace {

constexpr int kThreads = 256;

__device__ float as_float(half value) {
    return __half2float(value);
}

__device__ half as_half(float value) {
    return __float2half_rn(value);
}

template<bool Paged>
__global__ void attention_decode_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const std::int32_t* __restrict__ block_table,
    const std::int32_t* __restrict__ seq_lens,
    half* __restrict__ output,
    int batch,
    int q_heads,
    int kv_heads,
    int head_dim,
    int page_size,
    int max_blocks_per_seq,
    int max_seq_len) {

    extern __shared__ float scores[];

    const int head_index = blockIdx.x;
    const int total_heads = batch * q_heads;

    if (head_index >= total_heads) {
        return;
    }

    const int batch_index = head_index / q_heads;
    const int q_head = head_index % q_heads;
    const int repetition = q_heads / kv_heads;
    const int kv_head = q_head / repetition;
    const int sequence_len =
        min(seq_lens[batch_index], max_seq_len);

    const int tid = threadIdx.x;

    const half* q_ptr =
        q +
        (static_cast<std::size_t>(
             batch_index * q_heads + q_head)
         * head_dim);

    half* output_ptr =
        output +
        (static_cast<std::size_t>(
             batch_index * q_heads + q_head)
         * head_dim);

    const float scale =
        rsqrtf(static_cast<float>(head_dim));

    for (int pos = tid;
         pos < sequence_len;
         pos += blockDim.x) {

        int physical_page = 0;
        int page_offset = pos;

        if constexpr (Paged) {
            const int page_index =
                pos / page_size;

            page_offset =
                pos - page_index * page_size;

            physical_page =
                block_table[
                    batch_index *
                        max_blocks_per_seq +
                    page_index];
        }

        const std::size_t token_base =
            Paged
                ? (
                    (static_cast<std::size_t>(
                        physical_page) *
                     page_size +
                     page_offset) *
                    static_cast<std::size_t>(
                        kv_heads * head_dim)
                  )
                : (
                    (static_cast<std::size_t>(
                        batch_index) *
                     max_seq_len +
                     pos) *
                    static_cast<std::size_t>(
                        kv_heads * head_dim)
                  );

        const std::size_t kv_base =
            token_base +
            static_cast<std::size_t>(
                kv_head * head_dim);

        float dot = 0.0f;

        for (int d = 0;
             d < head_dim;
             ++d) {
            dot +=
                as_float(q_ptr[d]) *
                as_float(k[kv_base + d]);
        }

        scores[pos] =
            dot * scale;
    }

    __syncthreads();

    __shared__ float max_score;
    __shared__ float denominator;

    if (tid == 0) {
        float max_value =
            -std::numeric_limits<float>::infinity();

        for (int pos = 0;
             pos < sequence_len;
             ++pos) {
            max_value =
                fmaxf(max_value, scores[pos]);
        }

        max_score = max_value;

        float sum = 0.0f;

        for (int pos = 0;
             pos < sequence_len;
             ++pos) {
            sum += expf(
                scores[pos] -
                max_value);
        }

        denominator =
            fmaxf(sum, 1e-20f);
    }

    __syncthreads();

    for (int d = tid;
         d < head_dim;
         d += blockDim.x) {

        float value = 0.0f;

        for (int pos = 0;
             pos < sequence_len;
             ++pos) {

            int physical_page = 0;
            int page_offset = pos;

            if constexpr (Paged) {
                const int page_index =
                    pos / page_size;

                page_offset =
                    pos -
                    page_index * page_size;

                physical_page =
                    block_table[
                        batch_index *
                            max_blocks_per_seq +
                        page_index];
            }

            const std::size_t token_base =
                Paged
                    ? (
                        (static_cast<std::size_t>(
                            physical_page) *
                         page_size +
                         page_offset) *
                        static_cast<std::size_t>(
                            kv_heads * head_dim)
                      )
                    : (
                        (static_cast<std::size_t>(
                            batch_index) *
                         max_seq_len +
                         pos) *
                        static_cast<std::size_t>(
                            kv_heads * head_dim)
                      );

            const std::size_t kv_base =
                token_base +
                static_cast<std::size_t>(
                    kv_head * head_dim);

            const float weight =
                expf(
                    scores[pos] -
                    max_score) /
                denominator;

            value +=
                weight *
                as_float(v[kv_base + d]);
        }

        output_ptr[d] =
            as_half(value);
    }
}

bool invalid_common(
    int batch,
    int q_heads,
    int kv_heads,
    int head_dim,
    int max_seq_len) {

    return
        batch <= 0 ||
        q_heads <= 0 ||
        kv_heads <= 0 ||
        q_heads % kv_heads != 0 ||
        head_dim <= 0 ||
        max_seq_len <= 0;
}

}

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
    cudaStream_t stream) {

    if (invalid_common(
            batch, q_heads, kv_heads,
            head_dim, max_seq_len) ||
        page_size <= 0 ||
        max_blocks_per_seq <= 0) {
        return cudaErrorInvalidValue;
    }

    const std::size_t shared_bytes =
        static_cast<std::size_t>(
            max_seq_len) *
        sizeof(float);

    attention_decode_kernel<true>
        <<<batch * q_heads,
           kThreads,
           shared_bytes,
           stream>>>(
            static_cast<const half*>(q),
            static_cast<const half*>(k_pages),
            static_cast<const half*>(v_pages),
            block_table,
            seq_lens,
            static_cast<half*>(output),
            batch,
            q_heads,
            kv_heads,
            head_dim,
            page_size,
            max_blocks_per_seq,
            max_seq_len);

    return cudaGetLastError();
}

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
    cudaStream_t stream) {

    if (invalid_common(
            batch, q_heads, kv_heads,
            head_dim, max_seq_len)) {
        return cudaErrorInvalidValue;
    }

    const std::size_t shared_bytes =
        static_cast<std::size_t>(
            max_seq_len) *
        sizeof(float);

    attention_decode_kernel<false>
        <<<batch * q_heads,
           kThreads,
           shared_bytes,
           stream>>>(
            static_cast<const half*>(q),
            static_cast<const half*>(k_cache),
            static_cast<const half*>(v_cache),
            nullptr,
            seq_lens,
            static_cast<half*>(output),
            batch,
            q_heads,
            kv_heads,
            head_dim,
            0,
            0,
            max_seq_len);

    return cudaGetLastError();
}

}
