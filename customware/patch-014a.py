#!/usr/bin/env python
"""Entry 014, engine part: a position window for per-sequence state I/O of the attention cache, the ATTN_ONLY
state flag, and llama_memory_set_state_range. Exact-string edits; every anchor must match exactly once."""
import sys

BS = chr(92)
TAB = chr(9)

def patch(path, edits):
    src = open(path, encoding='utf-8').read()
    for old, new in edits:
        old = old.replace('%BS%', BS)
        new = new.replace('%BS%', BS)
        n = src.count(old)
        if n != 1:
            print('FAIL %s: anchor found %d times:\n%s' % (path, n, old[:200]))
            sys.exit(1)
        src = src.replace(old, new)
    open(path, 'w', encoding='utf-8', newline='\n').write(src)
    print('patched', path, len(edits), 'edits')

# ---------------------------------------------------------------------------------------------------- llama.h
patch('include/llama.h', [
(
"""// Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load).
// Getting the state for a seq_id with this flag invalidates all prior states gotten for that seq_id with this flag.
#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2
""",
"""// Keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load).
// Getting the state for a seq_id with this flag invalidates all prior states gotten for that seq_id with this flag.
#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2

// customware (entry 014): the attention cells only; a hybrid memory skips its recurrent part. A prefix segment shared
// by several conversations carries no state of its own, so it is written and read with this flag.
#define LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY (1u << 8)
"""),
(
"""    LLAMA_API uint32_t llama_memory_n_cells_owned(llama_memory_t mem, const llama_seq_id * seq_ids, size_t n_seq_ids);
""",
"""    LLAMA_API uint32_t llama_memory_n_cells_owned(llama_memory_t mem, const llama_seq_id * seq_ids, size_t n_seq_ids);

    // customware (entry 014): a position window for the per-sequence state writes and reads of the attention cache(s)
    // behind this memory, until it is set again: a write keeps only the sequence's cells with p0 <= pos < p1 (p1 < 0:
    // no upper bound); a read replaces the sequence's cells from p0 on instead of all of them, so the cells below p0
    // (re-attached from another sequence with llama_memory_seq_cp) stay. -1, -1 = the whole sequence, the default.
    // Whole-memory state I/O ignores the window. Returns false when the memory has no plain attention cache; then
    // every save is whole.
    LLAMA_API bool llama_memory_set_state_range(llama_memory_t mem, llama_pos p0, llama_pos p1);
"""),
])

# ---------------------------------------------------------------------------------------------------- llama-kv-cache.h
patch('src/llama-kv-cache.h', [
(
"""    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in
""",
"""    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // customware (entry 014): the position window of the per-sequence state I/O (see llama_memory_set_state_range)
    void set_state_range(llama_pos p0, llama_pos p1);

    // state_read, plus the cells the restored tokens were placed in
"""),
(
"""    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
""",
"""    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // customware (entry 014): per-sequence state I/O keeps only the cells with state_p0 <= pos < state_p1 (write) and
    // replaces the sequence's cells from state_p0 on (read); -1 = no bound
    llama_pos state_p0 = -1;
    llama_pos state_p1 = -1;

    // this is the SWA type of the cache - not to be confused with the model SWA type
"""),
])

# ---------------------------------------------------------------------------------------------------- llama-kv-cache.cpp
patch('src/llama-kv-cache.cpp', [
(
"""            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
""",
"""            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // customware (entry 014): the position window of a tail or prefix save
            if (add_cell && seq_id != -1 && state_p0 >= 0) {
                const llama_pos pos = cells.pos_get(i);

                add_cell = pos >= state_p0 && (state_p1 < 0 || pos < state_p1);
            }

            // check the cell is not SWA-masked
"""),
(
"""    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;
""",
"""    if (dest_seq_id != -1) {
        // single sequence
        // customware (entry 014): under a position window only the cells from state_p0 on are replaced; the ones below
        // (re-attached from another sequence) stay
        seq_rm(dest_seq_id, state_p0 >= 0 ? state_p0 : -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;
"""),
(
"""void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}
""",
"""void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache::set_state_range(llama_pos p0, llama_pos p1) {
    state_p0 = p0;
    state_p1 = p1;
}
"""),
])

# ---------------------------------------------------------------------------------------------------- hybrid
patch('src/llama-memory-hybrid.cpp', [
(
"""void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}
void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
}
""",
"""void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    // customware (entry 014): a prefix segment carries the attention cells only
    if ((flags & LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY) == 0) {
        mem_recr->state_write(io, seq_id, flags);
    }
}
void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_read(io, seq_id, flags);
    }
    if ((flags & LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY) == 0) {
        mem_recr->state_read(io, seq_id, flags);
    }
}
"""),
])

patch('src/llama-memory-hybrid-idx.cpp', [
(
"""        get_mem_recr()->state_read(io, seq_id, flags);

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
""",
"""        // customware (entry 014): a prefix segment carries the attention (and indexer) cells only
        if ((flags & LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY) == 0) {
            get_mem_recr()->state_read(io, seq_id, flags);
        }

        // [TAG_HYBRID_IDX_STATE] must mirror the write order in state_write
"""),
])

# ---------------------------------------------------------------------------------------------------- llama-context.cpp
patch('src/llama-context.cpp', [
(
"""#include "llama-memory-hybrid.h"
#include "llama-kv-cache.h"
""",
"""#include "llama-memory-hybrid.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-kv-cache.h"
"""),
(
"""#include <stdexcept>
#include <string>
#include <unordered_map>
""",
"""#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
"""),
(
"""            if ((cells->seq_get_all(i) & ~mask).none()) {
                n++;
            }
        }
    }

    return n;
}
""",
"""            if ((cells->seq_get_all(i) & ~mask).none()) {
                n++;
            }
        }
    }

    return n;
}

bool llama_memory_set_state_range(llama_memory_t mem, llama_pos p0, llama_pos p1) {
    if (!mem) {
        return false;
    }

    // only the plain attention cache honours the window; the variants with state paths of their own (dsa, iswa,
    // msa, dsv4) do not, and a window they ignored would save a whole sequence under a tail's label
    const auto plain = [](const llama_kv_cache * kv) {
        return kv != nullptr && typeid(*kv) == typeid(llama_kv_cache);
    };

    llama_kv_cache * attn = nullptr;
    llama_kv_cache * idx  = nullptr;

    if (auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem)) {
        attn = hybrid->get_mem_attn();
        if (auto * hybrid_idx = dynamic_cast<llama_memory_hybrid_idx *>(mem)) {
            idx = hybrid_idx->get_mem_idx();
        }
    } else {
        attn = dynamic_cast<llama_kv_cache *>(mem);
    }

    if (!plain(attn) || (idx != nullptr && !plain(idx))) {
        return false;
    }

    attn->set_state_range(p0, p1);
    if (idx != nullptr) {
        idx->set_state_range(p0, p1);
    }

    return true;
}
"""),
])

print('patch-014a: done')
