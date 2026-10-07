#!/usr/bin/env python
"""Entry 014, engine part 2: the edits patch-014a.py did not reach (its hybrid anchor lacked the blank line between
the two functions): the ATTN_ONLY gate in the hybrid memories and llama_memory_set_state_range in the context."""
import sys

BS = chr(92)

def patch(path, edits):
    src = open(path, encoding='utf-8').read()
    for old, new in edits:
        old = old.replace('%BS%', BS)
        new = new.replace('%BS%', BS)
        n = src.count(old)
        if n != 1:
            print('FAIL %s: anchor found %d times:\n%s' % (path, n, old[:300]))
            sys.exit(1)
        src = src.replace(old, new)
    open(path, 'w', encoding='utf-8', newline='\n').write(src)
    print('patched', path, len(edits), 'edits')

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

print('patch-014a2: done')
