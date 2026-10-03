#include "dlse/cuda_kernels.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

static void check(
    cudaError_t error,
    const char* label) {
    if (error != cudaSuccess) {
        std::cerr << label << ": "
                  << cudaGetErrorString(error)
                  << "\n";
        std::exit(1);
    }
}

int main() {
    constexpr int batch = 4;
    constexpr int q_heads = 8;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int page_size = 16;
    constexpr int max_seq_len = 129;
    constexpr int max_blocks =
        (max_seq_len + page_size - 1) /
        page_size;

    const std::size_t q_elems =
        static_cast<std::size_t>(batch) *
        q_heads * head_dim;

    const std::size_t contig_elems =
        static_cast<std::size_t>(batch) *
        max_seq_len * kv_heads * head_dim;

    const std::size_t page_count =
        static_cast<std::size_t>(batch) *
        max_blocks;

    const std::size_t paged_elems =
        page_count *
        page_size *
        kv_heads *
        head_dim;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float>
        dist(-1.0f, 1.0f);

    std::vector<half> host_q(q_elems);
    std::vector<half> host_k(contig_elems);
    std::vector<half> host_v(contig_elems);

    for (auto* values :
         {&host_q, &host_k, &host_v}) {
        for (auto& value : *values) {
            value = __float2half(
                dist(rng));
        }
    }

    const std::vector<int> lengths{
        129, 65, 33, 17
    };

    std::vector<std::int32_t> host_len(
        lengths.begin(),
        lengths.end());

    std::vector<std::int32_t>
        host_table(
            static_cast<std::size_t>(batch) *
            max_blocks,
            -1);

    std::vector<half> host_k_paged(
        paged_elems);

    std::vector<half> host_v_paged(
        paged_elems);

    for (int b = 0;
         b < batch;
         ++b) {
        for (int page = 0;
             page < max_blocks;
             ++page) {

            const int physical =
                b * max_blocks + page;

            host_table[
                static_cast<std::size_t>(b) *
                max_blocks + page] =
                physical;

            for (int offset = 0;
                 offset < page_size;
                 ++offset) {

                const int pos =
                    page * page_size +
                    offset;

                if (pos >= max_seq_len) {
                    break;
                }

                const std::size_t source =
                    (static_cast<std::size_t>(b) *
                     max_seq_len +
                     pos) *
                    static_cast<std::size_t>(
                        kv_heads * head_dim);

                const std::size_t destination =
                    (static_cast<std::size_t>(
                         physical) *
                     page_size +
                     offset) *
                    static_cast<std::size_t>(
                        kv_heads * head_dim);

                std::copy_n(
                    host_k.data() + source,
                    kv_heads * head_dim,
                    host_k_paged.data() +
                        destination);

                std::copy_n(
                    host_v.data() + source,
                    kv_heads * head_dim,
                    host_v_paged.data() +
                        destination);
            }
        }
    }

    half* device_q = nullptr;
    half* device_k = nullptr;
    half* device_v = nullptr;
    half* device_k_paged = nullptr;
    half* device_v_paged = nullptr;
    half* output_contig = nullptr;
    half* output_paged = nullptr;

    std::int32_t* device_len = nullptr;
    std::int32_t* device_table = nullptr;

    check(cudaMalloc(
        &device_q,
        q_elems * sizeof(half)),
        "alloc q");

    check(cudaMalloc(
        &device_k,
        contig_elems * sizeof(half)),
        "alloc k");

    check(cudaMalloc(
        &device_v,
        contig_elems * sizeof(half)),
        "alloc v");

    check(cudaMalloc(
        &device_k_paged,
        paged_elems * sizeof(half)),
        "alloc paged k");

    check(cudaMalloc(
        &device_v_paged,
        paged_elems * sizeof(half)),
        "alloc paged v");

    check(cudaMalloc(
        &output_contig,
        q_elems * sizeof(half)),
        "alloc output");

    check(cudaMalloc(
        &output_paged,
        q_elems * sizeof(half)),
        "alloc paged output");

    check(cudaMalloc(
        &device_len,
        batch * sizeof(std::int32_t)),
        "alloc lengths");

    check(cudaMalloc(
        &device_table,
        host_table.size() *
            sizeof(std::int32_t)),
        "alloc table");

    check(cudaMemcpy(
        device_q,
        host_q.data(),
        q_elems * sizeof(half),
        cudaMemcpyHostToDevice),
        "copy q");

    check(cudaMemcpy(
        device_k,
        host_k.data(),
        contig_elems * sizeof(half),
        cudaMemcpyHostToDevice),
        "copy k");

    check(cudaMemcpy(
        device_v,
        host_v.data(),
        contig_elems * sizeof(half),
        cudaMemcpyHostToDevice),
        "copy v");

    check(cudaMemcpy(
        device_k_paged,
        host_k_paged.data(),
        paged_elems * sizeof(half),
        cudaMemcpyHostToDevice),
        "copy paged k");

    check(cudaMemcpy(
        device_v_paged,
        host_v_paged.data(),
        paged_elems * sizeof(half),
        cudaMemcpyHostToDevice),
        "copy paged v");

    check(cudaMemcpy(
        device_len,
        host_len.data(),
        batch * sizeof(std::int32_t),
        cudaMemcpyHostToDevice),
        "copy lengths");

    check(cudaMemcpy(
        device_table,
        host_table.data(),
        host_table.size() *
            sizeof(std::int32_t),
        cudaMemcpyHostToDevice),
        "copy table");

    for (int i = 0; i < 10; ++i) {
        check(
            dlse::launch_contiguous_attention_decode(
                device_q,
                device_k,
                device_v,
                device_len,
                output_contig,
                batch,
                q_heads,
                kv_heads,
                head_dim,
                max_seq_len,
                nullptr),
            "warmup contiguous");

        check(
            dlse::launch_paged_attention_decode(
                device_q,
                device_k_paged,
                device_v_paged,
                device_table,
                device_len,
                output_paged,
                batch,
                q_heads,
                kv_heads,
                head_dim,
                page_size,
                max_blocks,
                max_seq_len,
                nullptr),
            "warmup paged");
    }

    check(
        cudaDeviceSynchronize(),
        "warmup sync");

    const int iterations = 100;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    check(cudaEventCreate(&start),
          "event start");
    check(cudaEventCreate(&stop),
          "event stop");

    check(cudaEventRecord(start),
          "record contig start");

    for (int i = 0;
         i < iterations;
         ++i) {
        check(
            dlse::launch_contiguous_attention_decode(
                device_q,
                device_k,
                device_v,
                device_len,
                output_contig,
                batch,
                q_heads,
                kv_heads,
                head_dim,
                max_seq_len,
                nullptr),
            "benchmark contiguous");
    }

    check(cudaEventRecord(stop),
          "record contig stop");

    check(cudaEventSynchronize(stop),
          "sync contig");

    float contiguous_ms = 0.0f;

    check(cudaEventElapsedTime(
        &contiguous_ms,
        start,
        stop),
        "elapsed contig");

    check(cudaEventRecord(start),
          "record paged start");

    for (int i = 0;
         i < iterations;
         ++i) {
        check(
            dlse::launch_paged_attention_decode(
                device_q,
                device_k_paged,
                device_v_paged,
                device_table,
                device_len,
                output_paged,
                batch,
                q_heads,
                kv_heads,
                head_dim,
                page_size,
                max_blocks,
                max_seq_len,
                nullptr),
            "benchmark paged");
    }

    check(cudaEventRecord(stop),
          "record paged stop");

    check(cudaEventSynchronize(stop),
          "sync paged");

    float paged_ms = 0.0f;

    check(cudaEventElapsedTime(
        &paged_ms,
        start,
        stop),
        "elapsed paged");

    std::vector<half>
        host_output_contig(q_elems);

    std::vector<half>
        host_output_paged(q_elems);

    check(cudaMemcpy(
        host_output_contig.data(),
        output_contig,
        q_elems * sizeof(half),
        cudaMemcpyDeviceToHost),
        "copy output contig");

    check(cudaMemcpy(
        host_output_paged.data(),
        output_paged,
        q_elems * sizeof(half),
        cudaMemcpyDeviceToHost),
        "copy output paged");

    float max_abs_error = 0.0f;

    for (std::size_t i = 0;
         i < q_elems;
         ++i) {
        max_abs_error =
            std::max(
                max_abs_error,
                std::fabs(
                    __half2float(
                        host_output_contig[i]) -
                    __half2float(
                        host_output_paged[i])));
    }

    std::cout
        << "contiguous_ms_per_iter="
        << contiguous_ms /
            static_cast<float>(
                iterations)
        << " paged_ms_per_iter="
        << paged_ms /
            static_cast<float>(
                iterations)
        << " max_abs_error="
        << max_abs_error << "\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(device_q);
    cudaFree(device_k);
    cudaFree(device_v);
    cudaFree(device_k_paged);
    cudaFree(device_v_paged);
    cudaFree(output_contig);
    cudaFree(output_paged);
    cudaFree(device_len);
    cudaFree(device_table);

    return max_abs_error < 1e-2f
        ? 0
        : 2;
}
