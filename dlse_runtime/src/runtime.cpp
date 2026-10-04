#include "dlse/runtime.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace dlse {

ContinuousBatchScheduler::ContinuousBatchScheduler(
    SchedulerConfig config)
    : config_(config) {
    if (config_.max_batch_size == 0 ||
        config_.max_batched_tokens == 0 ||
        config_.prefill_chunk_tokens == 0) {
        throw std::invalid_argument("scheduler limits must be positive");
    }

    if (config_.max_batched_tokens < config_.max_batch_size) {
        throw std::invalid_argument(
            "max_batched_tokens must cover one decode token per slot");
    }
}

RequestId ContinuousBatchScheduler::submit(
    std::size_t prompt_tokens,
    std::size_t max_new_tokens) {
    if (max_new_tokens == 0) {
        throw std::invalid_argument(
            "max_new_tokens must be positive");
    }

    const RequestId id = next_id_++;
    Entry entry;
    entry.request.id = id;
    entry.request.prompt_tokens = prompt_tokens;
    entry.request.max_new_tokens = max_new_tokens;
    entries_.emplace(id, entry);

    if (prompt_tokens == 0) {
        maybe_activate(id);
    } else {
        waiting_prefill_.push_back(id);
    }

    return id;
}

void ContinuousBatchScheduler::maybe_activate(RequestId id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }

    auto& entry = it->second;

    if (entry.request.finished || entry.active) {
        return;
    }

    if (entry.request.prefill_cursor >=
        entry.request.prompt_tokens) {
        entry.active = true;
        active_decode_.push_back(id);
    }
}

IterationPlan ContinuousBatchScheduler::dispatch() {
    ++iteration_;

    IterationPlan plan;
    plan.iteration = iteration_;

    std::size_t budget =
        config_.max_batched_tokens;

    std::size_t decode_slots =
        std::min(config_.max_batch_size, budget);

    // Decode first: active generations receive one token
    // before the remaining budget is given to prefills.
    const std::size_t active_at_start =
        active_decode_.size();

    for (std::size_t i = 0;
         i < active_at_start && decode_slots > 0;
         ++i) {
        const RequestId id =
            active_decode_.front();

        active_decode_.pop_front();

        auto it = entries_.find(id);

        if (it == entries_.end() ||
            it->second.request.finished) {
            continue;
        }

        auto& request = it->second.request;

        plan.decode.push_back({
            id,
            WorkKind::Decode,
            request.generated_tokens,
            1
        });

        ++request.generated_tokens;
        --decode_slots;
        --budget;

        if (request.generated_tokens >=
            request.max_new_tokens) {
            request.finished = true;
            it->second.active = false;
            ++finished_;
        } else {
            active_decode_.push_back(id);
        }
    }

    // Round-robin bounded prefill: no waiting request
    // receives more than one configured chunk per iteration.
    std::size_t guard =
        waiting_prefill_.size();

    while (budget > 0 &&
           !waiting_prefill_.empty() &&
           guard-- > 0) {
        const RequestId id =
            waiting_prefill_.front();

        waiting_prefill_.pop_front();

        auto it = entries_.find(id);

        if (it == entries_.end() ||
            it->second.request.finished) {
            continue;
        }

        auto& request = it->second.request;

        const std::size_t remaining =
            request.prompt_tokens -
            request.prefill_cursor;

        const std::size_t chunk =
            std::min({
                remaining,
                config_.prefill_chunk_tokens,
                budget
            });

        if (chunk == 0) {
            maybe_activate(id);
            continue;
        }

        const std::size_t offset =
            request.prefill_cursor;

        request.prefill_cursor += chunk;
        budget -= chunk;

        plan.prefill.push_back({
            id,
            WorkKind::Prefill,
            offset,
            chunk
        });

        if (request.prefill_cursor >=
            request.prompt_tokens) {
            maybe_activate(id);
        } else {
            waiting_prefill_.push_back(id);
        }
    }

    return plan;
}

void ContinuousBatchScheduler::finish(RequestId id) {
    auto it = entries_.find(id);

    if (it == entries_.end() ||
        it->second.request.finished) {
        return;
    }

    it->second.request.finished = true;
    it->second.active = false;
    ++finished_;
}

void ContinuousBatchScheduler::abort(RequestId id) {
    finish(id);
}

std::optional<Request>
ContinuousBatchScheduler::get(RequestId id) const {
    const auto it = entries_.find(id);

    if (it == entries_.end()) {
        return std::nullopt;
    }

    return it->second.request;
}

SchedulerSnapshot
ContinuousBatchScheduler::snapshot() const {
    return {
        iteration_,
        waiting_prefill_.size(),
        active_decode_.size(),
        finished_,
        entries_.size()
    };
}

PagedKVCache::PagedKVCache(
    KvCacheConfig config)
    : config_(config) {
    if (config_.page_size_tokens == 0 ||
        config_.max_pages == 0) {
        throw std::invalid_argument(
            "page size and max pages must be positive");
    }

    if (config_.max_pages >
        static_cast<std::size_t>(
            std::numeric_limits<PhysicalPage>::max())) {
        throw std::invalid_argument(
            "max_pages exceeds PhysicalPage range");
    }

    free_pages_.reserve(
        config_.max_pages);

    for (std::size_t i = 0;
         i < config_.max_pages;
         ++i) {
        free_pages_.push_back(
            static_cast<PhysicalPage>(
                config_.max_pages - 1 - i));
    }
}

std::size_t PagedKVCache::pages_for_tokens(
    std::size_t tokens) const noexcept {
    if (tokens == 0) {
        return 0;
    }

    return (
        tokens +
        config_.page_size_tokens - 1
    ) / config_.page_size_tokens;
}

void PagedKVCache::create(
    RequestId id) {
    if (sequences_.contains(id)) {
        throw std::invalid_argument(
            "request already exists in KV cache");
    }

    sequences_.emplace(
        id,
        Sequence{}
    );
}

void PagedKVCache::append(
    RequestId id,
    std::size_t tokens) {
    auto it = sequences_.find(id);

    if (it == sequences_.end()) {
        throw std::out_of_range(
            "unknown request");
    }

    auto& sequence = it->second;

    if (tokens >
        std::numeric_limits<std::size_t>::max() -
        sequence.logical_tokens) {
        throw std::overflow_error(
            "logical token count overflow");
    }

    const std::size_t target =
        sequence.logical_tokens +
        tokens;

    const std::size_t required_pages =
        pages_for_tokens(target);

    while (sequence.pages.size() <
           required_pages) {
        if (free_pages_.empty()) {
            throw std::runtime_error(
                "KV page pool exhausted");
        }

        sequence.pages.push_back(
            free_pages_.back());
        free_pages_.pop_back();
        ++used_pages_;
    }

    sequence.logical_tokens = target;
}

void PagedKVCache::release(
    RequestId id) {
    auto it = sequences_.find(id);

    if (it == sequences_.end()) {
        return;
    }

    for (const auto page :
         it->second.pages) {
        free_pages_.push_back(page);
    }

    used_pages_ -=
        it->second.pages.size();

    sequences_.erase(it);
}

const std::vector<PhysicalPage>&
PagedKVCache::block_table(
    RequestId id) const {
    const auto it =
        sequences_.find(id);

    if (it == sequences_.end()) {
        throw std::out_of_range(
            "unknown request");
    }

    return it->second.pages;
}

std::size_t PagedKVCache::logical_tokens(
    RequestId id) const {
    const auto it =
        sequences_.find(id);

    if (it == sequences_.end()) {
        throw std::out_of_range(
            "unknown request");
    }

    return it->second.logical_tokens;
}

KvCacheStats PagedKVCache::stats() const {
    std::size_t logical = 0;
    std::size_t reserved = 0;

    for (const auto& [id, sequence] :
         sequences_) {
        (void)id;

        logical +=
            sequence.logical_tokens;

        reserved +=
            sequence.pages.size() *
            config_.page_size_tokens;
    }

    const std::size_t reserved_bytes =
        reserved * config_.bytes_per_token;

    const double fragmentation =
        reserved == 0
            ? 0.0
            : 1.0 -
                  static_cast<double>(
                      logical) /
                  static_cast<double>(
                      reserved);

    return {
        config_.max_pages,
        free_pages_.size(),
        used_pages_,
        logical,
        reserved,
        reserved_bytes,
        fragmentation
    };
}

} // namespace dlse
