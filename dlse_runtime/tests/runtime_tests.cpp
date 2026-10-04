#include "dlse/runtime.h"

#include <cassert>

int main() {
    {
        dlse::ContinuousBatchScheduler scheduler({
            4, 16, 4
        });

        const auto id =
            scheduler.submit(10, 3);

        const auto first =
            scheduler.dispatch();

        assert(first.prefill.size() == 1);
        assert(first.prefill[0].token_count == 4);

        const auto second =
            scheduler.dispatch();

        assert(second.prefill.size() == 1);
        assert(second.prefill[0].token_count == 4);

        const auto third =
            scheduler.dispatch();

        assert(third.prefill.size() == 1);
        assert(third.prefill[0].token_count == 2);

        const auto fourth =
            scheduler.dispatch();

        assert(fourth.decode.size() == 1);
        assert(fourth.decode[0].request_id == id);
    }

    {
        dlse::PagedKVCache cache({
            16, 8, 1024
        });

        const auto id =
            static_cast<dlse::RequestId>(7);

        cache.create(id);
        cache.append(id, 15);

        assert(
            cache.block_table(id).size() == 1
        );

        cache.append(id, 1);

        assert(
            cache.block_table(id).size() == 1
        );

        cache.append(id, 1);

        assert(
            cache.block_table(id).size() == 2
        );

        const auto stats =
            cache.stats();

        assert(stats.logical_tokens == 17);
        assert(stats.reserved_tokens == 32);
        assert(stats.used_pages == 2);

        cache.release(id);

        assert(
            cache.stats().used_pages == 0
        );
    }

    return 0;
}
