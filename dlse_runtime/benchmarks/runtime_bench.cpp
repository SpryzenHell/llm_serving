#include "dlse/runtime.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

int main() {
    dlse::ContinuousBatchScheduler scheduler({
        8, 64, 32
    });

    std::vector<dlse::RequestId> decoders;

    for (int i = 0; i < 8; ++i) {
        decoders.push_back(
            scheduler.submit(0, 1000)
        );
    }

    // Activate the eight decode streams.
    (void)scheduler.dispatch();

    const auto long_prompt =
        scheduler.submit(4096, 16);

    std::vector<std::size_t>
        decode_iterations;

    std::size_t prefetched = 0;

    for (int i = 0; i < 200; ++i) {
        const auto plan =
            scheduler.dispatch();

        for (const auto& item :
             plan.decode) {
            if (std::find(
                    decoders.begin(),
                    decoders.end(),
                    item.request_id
                ) != decoders.end()) {
                decode_iterations.push_back(
                    static_cast<std::size_t>(
                        plan.iteration
                    )
                );
            }
        }

        for (const auto& item :
             plan.prefill) {
            if (item.request_id ==
                long_prompt) {
                prefetched += item.token_count;
            }
        }
    }

    std::size_t max_gap = 0;

    for (std::size_t i = 1;
         i < decode_iterations.size();
         ++i) {
        max_gap = std::max(
            max_gap,
            decode_iterations[i] -
            decode_iterations[i - 1]
        );
    }

    // Example FP16 KV footprint for a
    // 32-layer GQA model. This measures
    // KV reservation, not whole-process VRAM.
    constexpr std::size_t page_size = 16;
    constexpr std::size_t max_seq_len = 4096;
    constexpr std::size_t batch = 8;
    constexpr std::size_t bytes_per_token = 4096;

    dlse::PagedKVCache cache({
        page_size, 4096, bytes_per_token
    });

    const std::vector<std::size_t> lengths{
        128, 256, 512, 768,
        1024, 128, 64, 512
    };

    for (std::size_t i = 0;
         i < lengths.size();
         ++i) {
        const auto id =
            static_cast<dlse::RequestId>(
                i + 1
            );

        cache.create(id);
        cache.append(id, lengths[i]);
    }

    const auto stats =
        cache.stats();

    const std::size_t static_bytes =
        batch *
        max_seq_len *
        bytes_per_token;

    const double reclaim =
        static_bytes == 0
            ? 0.0
            : 1.0 -
                static_cast<double>(
                    stats.reserved_bytes
                ) /
                static_cast<double>(
                    static_bytes
                );

    std::cout
        << "max_decode_gap_iterations="
        << max_gap << "\n"
        << "long_prompt_prefilled="
        << prefetched << "\n"
        << "kv_logical_tokens="
        << stats.logical_tokens << "\n"
        << "kv_reserved_tokens="
        << stats.reserved_tokens << "\n"
        << "kv_capacity_reclaim="
        << reclaim << "\n";
}
