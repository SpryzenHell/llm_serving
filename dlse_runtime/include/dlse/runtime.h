#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

namespace dlse {

using RequestId = std::uint64_t;
using PhysicalPage = std::uint32_t;

enum class WorkKind : std::uint8_t { Prefill, Decode };

struct Request {
    RequestId id{};
    std::size_t prompt_tokens{};
    std::size_t max_new_tokens{};
    std::size_t prefill_cursor{};
    std::size_t generated_tokens{};
    bool finished{false};
};

struct WorkItem {
    RequestId request_id{};
    WorkKind kind{};
    std::size_t token_offset{};
    std::size_t token_count{};
};

struct IterationPlan {
    std::uint64_t iteration{};
    std::vector<WorkItem> prefill;
    std::vector<WorkItem> decode;
};

struct SchedulerConfig {
    std::size_t max_batch_size{8};
    std::size_t max_batched_tokens{128};
    std::size_t prefill_chunk_tokens{32};
};

struct SchedulerSnapshot {
    std::uint64_t iteration{};
    std::size_t waiting{};
    std::size_t active{};
    std::size_t finished{};
    std::size_t submitted{};
};

class ContinuousBatchScheduler {
public:
    explicit ContinuousBatchScheduler(SchedulerConfig config);

    RequestId submit(
        std::size_t prompt_tokens,
        std::size_t max_new_tokens);

    IterationPlan dispatch();

    void finish(RequestId id);
    void abort(RequestId id);

    std::optional<Request> get(RequestId id) const;
    SchedulerSnapshot snapshot() const;

private:
    struct Entry {
        Request request;
        bool active{false};
    };

    SchedulerConfig config_;
    RequestId next_id_{1};
    std::uint64_t iteration_{0};
    std::size_t finished_{0};

    std::unordered_map<RequestId, Entry> entries_;
    std::deque<RequestId> waiting_prefill_;
    std::deque<RequestId> active_decode_;

    void maybe_activate(RequestId id);
};

struct KvCacheConfig {
    std::size_t page_size_tokens{16};
    std::size_t max_pages{4096};
    std::size_t bytes_per_token{0};
};

struct KvCacheStats {
    std::size_t total_pages{};
    std::size_t free_pages{};
    std::size_t used_pages{};
    std::size_t logical_tokens{};
    std::size_t reserved_tokens{};
    std::size_t reserved_bytes{};
    double internal_fragmentation{0.0};
};

class PagedKVCache {
public:
    explicit PagedKVCache(KvCacheConfig config);

    void create(RequestId id);
    void append(RequestId id, std::size_t tokens);
    void release(RequestId id);

    const std::vector<PhysicalPage>& block_table(
        RequestId id) const;

    std::size_t logical_tokens(RequestId id) const;
    std::size_t pages_for_tokens(
        std::size_t tokens) const noexcept;
    KvCacheStats stats() const;

private:
    struct Sequence {
        std::vector<PhysicalPage> pages;
        std::size_t logical_tokens{0};
    };

    KvCacheConfig config_;
    std::vector<PhysicalPage> free_pages_;
    std::unordered_map<RequestId, Sequence> sequences_;
    std::size_t used_pages_{0};
};

} // namespace dlse
