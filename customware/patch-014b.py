#!/usr/bin/env python
"""Entry 014, server part: tail entries and prefix segments in the prompt cache, prefix holders, the restore that
re-attaches a tail's prefix, and the call sites. Exact-string edits; every anchor must match exactly once."""
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

# ============================================================================================ server-task.h
patch('tools/server/server-task.h', [
(
"""struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    // customware: the saved state of a suspended (preempted) slot; never evicted, erased on resume
    bool pinned = false;

    size_t size() const {
""",
"""// customware (entry 014): a sequence in the pool holding the first n_tokens of `tokens` with its cells; a tail entry
// whose prefix one of them covers re-attaches to it on restore (labels only) instead of restoring the prefix
struct server_prefix_holder {
    int                   id       = -1;      // slot id = sequence id
    const server_tokens * tokens   = nullptr;
    int32_t               n_tokens = 0;       // how many of them are in the pool
};

using server_prefix_holders = std::vector<server_prefix_holder>;

// customware (entry 014): what a restore from the cache would cost
struct server_prompt_cache_peek {
    int64_t n_tokens = -1; // the entry's length; -1 if none
    int64_t n_cells  = 0;  // cells the restore allocates: the whole entry, or its tail when a holder covers the prefix
    int     holder   = -1; // the slot whose cells the prefix re-attaches to
};

// customware (entry 014): what a restore did
struct server_prompt_restore {
    int     fork_of   = -1; // the holder the prefix re-attached to
    int32_t n_shared  = 0;  // cells shared with it
    int32_t n_segment = 0;  // cells restored from a prefix segment
    int32_t n_tail    = 0;  // cells restored from the entry itself
};

// customware (entry 014): the position window of the attention cache's per-sequence state I/O, reset on scope exit
struct server_state_range {
    llama_memory_t mem;
    bool ok;

    server_state_range(llama_memory_t mem, llama_pos p0, llama_pos p1) : mem(mem) {
        ok = llama_memory_set_state_range(mem, p0, p1);
    }

    ~server_state_range() {
        if (ok) {
            llama_memory_set_state_range(mem, -1, -1);
        }
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    // customware: the saved state of a suspended (preempted) slot; never evicted, erased on resume
    bool pinned = false;

    // customware (entry 014): data.main holds the attention cells from position p0 on (and the recurrent state at
    // the end); 0 = the whole sequence. A tail (p0 > 0) needs its prefix [0, p0) from a live holder or a segment
    llama_pos p0 = 0;

    // customware (entry 014): a prefix segment: the attention cells [0, tokens.size()) of its tokens, no recurrent
    // state, no checkpoints. Shared by the tails of one family; never matched as a prompt, never restored alone,
    // evicted only after the last tail it covers
    bool segment = false;

    size_t size() const {
"""),
(
"""    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    // customware: the length in tokens of the entry load() would restore for tokens_new, -1 if none. Under a
    // unified KV pool the caller needs that many free cells before it asks for the restore.
    int64_t peek(const server_prompt & prompt, const server_tokens & tokens_new) const;
""",
"""    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    bool load(server_prompt & prompt, const server_tokens & tokens_new, const server_prefix_holders & holders, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, server_prompt_restore & out);

    // customware: what the entry load() would restore for tokens_new costs: its length, the cells it allocates (its
    // tail only when a holder covers the prefix), the holder; n_tokens = -1 if none. Under a unified KV pool the
    // caller needs that many free cells before it asks for the restore.
    server_prompt_cache_peek peek(const server_prompt & prompt, const server_tokens & tokens_new, const server_prefix_holders & holders) const;

    // customware (entry 014): save the sequence `id` (its prompt) as a whole, or as a tail behind the longest prefix
    // (>= tail_min() tokens) that a holder or a segment covers; a segment is written first when none covers it.
    // Pinned for a suspended slot (no checkpoints then; the tokens are kept for the coverage checks).
    server_prompt_cache_state * save(const server_prompt & prompt, const server_prefix_holders & holders, int32_t id, llama_context * ctx_tgt, llama_context * ctx_dft, bool pinned);

    // customware (entry 014): put an entry's state into the sequence id_slot: the prefix of a tail re-attached to a
    // holder (seq_cp) or restored from a segment, then the entry's own cells. `tokens` are the conversation's
    // (the entry's, or the slot's for a pinned one). Nothing of the sequence is left on failure; the entry stays.
    bool restore(server_prompt_cache_state & state, const server_tokens & tokens, const server_prefix_holders & holders, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, server_prompt_restore & out);

    // customware (entry 014): the holder (not id_self) whose cells cover [0, c) of tokens; nullptr if none
    static const server_prefix_holder * find_holder(const server_tokens & tokens, int32_t c, const server_prefix_holders & holders, int32_t id_self);

    // customware (entry 014): a segment whose tokens cover [0, c) of tokens; nullptr if none
    const server_prompt_cache_state * find_segment(const server_tokens & tokens, int32_t c) const;

    // customware (entry 014): shorter shared prefixes are saved whole (env LLAMA_TAIL_MIN, default 4096; 0 = never a tail)
    static int32_t tail_min();
"""),
(
"""    bool load_exact(server_prompt_cache_state * state, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);
""",
"""    bool load_exact(server_prompt_cache_state * state, const server_tokens & tokens, const server_prefix_holders & holders, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, server_prompt_restore & out);
"""),
(
"""private:
    // the entry that best extends tokens_new while keeping most of its own context; states.end() if none
    std::list<server_prompt_cache_state>::const_iterator select(const server_prompt & prompt, const server_tokens & tokens_new, float & f_keep_best, float & f_sim_best) const;

    // evict the oldest entry that is not pinned; false if there is none
    bool evict_oldest();
""",
"""private:
    // the entry that best extends tokens_new while keeping most of its own context; states.end() if none.
    // customware (entry 014): a tail with neither a holder nor a segment for its prefix cannot be restored and is skipped
    std::list<server_prompt_cache_state>::const_iterator select(const server_prompt & prompt, const server_tokens & tokens_new, const server_prefix_holders & holders, float & f_keep_best, float & f_sim_best) const;

    // evict the oldest entry that is not pinned and not a segment some tail still needs; false if there is none
    bool evict_oldest();

    // customware (entry 014): true if an entry (not pinned, not a segment) holds all of tokens
    bool has_prompt(const server_tokens & tokens) const;

    // customware (entry 014): the longest prefix of tokens (>= tail_min()) that a holder (not id_self) or a segment
    // covers; 0 if none. The provider found comes back through holder / segment (a segment covering the chosen
    // prefix is reported even when a holder chose it, so no second one is written).
    int32_t split_point(const server_tokens & tokens, const server_prefix_holders & holders, int32_t id_self, const server_prefix_holder ** holder, const server_prompt_cache_state ** segment) const;

    // customware (entry 014): a segment entry for the first c tokens of tokens, its data sized; nullptr if no room
    server_prompt_cache_state * alloc_segment(const server_tokens & tokens, int32_t c, size_t state_size);

    // customware (entry 014): the segment of the tail being saved, kept through the eviction its allocation may run
    const server_prompt_cache_state * seg_hold = nullptr;
"""),
])

# ============================================================================================ server-task.cpp
patch('tools/server/server-task.cpp', [
# alloc: the two loops skip segments
(
"""    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        // customware (entry 012): a suspended slot's parked state has no token list and takes no part in matching
        if (it->pinned) {
            continue;
        }
""",
"""    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        // customware (entry 012): a suspended slot's parked state has no token list and takes no part in matching
        // customware (entry 014): a segment is not a conversation either
        if (it->pinned || it->segment) {
            continue;
        }
"""),
(
"""        // customware (entry 012): never a pinned entry: its empty token list is "contained" in every prompt, and a
        // suspended slot holds a pointer to it (found by a fork that was suspended, then a purge that saved first)
        if (it->pinned) {
            ++it;
            continue;
        }
""",
"""        // customware (entry 012): never a pinned entry: its empty token list is "contained" in every prompt, and a
        // suspended slot holds a pointer to it (found by a fork that was suspended, then a purge that saved first)
        // customware (entry 014): nor a segment: its tokens are a prefix of every tail it serves
        if (it->pinned || it->segment) {
            ++it;
            continue;
        }
"""),
# select
(
"""std::list<server_prompt_cache_state>::const_iterator server_prompt_cache::select(const server_prompt & prompt, const server_tokens & tokens_new, float & f_keep_best, float & f_sim_best) const {
""",
"""std::list<server_prompt_cache_state>::const_iterator server_prompt_cache::select(const server_prompt & prompt, const server_tokens & tokens_new, const server_prefix_holders & holders, float & f_keep_best, float & f_sim_best) const {
"""),
(
"""        // customware (entry 012): pinned entries belong to suspended slots and are not prompts to load
        if (it->pinned) {
            continue;
        }

        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
""",
"""        // customware (entry 012): pinned entries belong to suspended slots and are not prompts to load
        // customware (entry 014): a segment is a prefix without a state of its own; a tail can be loaded only while
        // a holder or a segment covers its prefix
        if (it->pinned || it->segment) {
            continue;
        }
        if (it->p0 > 0 && find_holder(it->prompt.tokens, it->p0, holders, -1) == nullptr && find_segment(it->prompt.tokens, it->p0) == nullptr) {
            continue;
        }

        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
"""),
# peek
(
"""int64_t server_prompt_cache::peek(const server_prompt & prompt, const server_tokens & tokens_new) const {
    float f_keep_best = 0.0f;
    float f_sim_best  = 0.0f;

    const auto it = select(prompt, tokens_new, f_keep_best, f_sim_best);

    return it == states.cend() ? -1 : (int64_t) it->prompt.tokens.size();
}
""",
"""server_prompt_cache_peek server_prompt_cache::peek(const server_prompt & prompt, const server_tokens & tokens_new, const server_prefix_holders & holders) const {
    float f_keep_best = 0.0f;
    float f_sim_best  = 0.0f;

    server_prompt_cache_peek res;

    const auto it = select(prompt, tokens_new, holders, f_keep_best, f_sim_best);
    if (it == states.cend()) {
        return res;
    }

    res.n_tokens = (int64_t) it->prompt.tokens.size();
    res.n_cells  = res.n_tokens;

    if (it->p0 > 0) {
        if (const auto * holder = find_holder(it->prompt.tokens, it->p0, holders, -1)) {
            res.holder   = holder->id;
            res.n_cells -= it->p0;
        }
    }

    return res;
}
"""),
# load
(
"""bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    float f_keep_best = 0.0f;
    float f_sim_best  = 0.0f;

    // the selection is const; the restore below consumes the entry
    const auto it_sel = select(prompt, tokens_new, f_keep_best, f_sim_best);
""",
"""bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, const server_prefix_holders & holders, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, server_prompt_restore & out) {
    float f_keep_best = 0.0f;
    float f_sim_best  = 0.0f;

    out = server_prompt_restore{};

    // the selection is const; the restore below consumes the entry
    const auto it_sel = select(prompt, tokens_new, holders, f_keep_best, f_sim_best);
"""),
(
"""    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f%BS%n", f_keep_best, f_sim_best);

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu%BS%n", size);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu%BS%n", size);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);
    }

    return true;
}
""",
"""    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f%BS%n", f_keep_best, f_sim_best);

        // customware (entry 014): the prefix of a tail comes from a holder or a segment, the rest from the entry
        if (!restore(*it_best, it_best->prompt.tokens, holders, ctx_tgt, ctx_dft, id_slot, out)) {
            return false;
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);
    }

    return true;
}
"""),
# evict_oldest
(
"""bool server_prompt_cache::evict_oldest() {
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->pinned) {
            continue;
        }

        SRV_WRN(" - cache limit reached, removing oldest entry (size = %.3f MiB)%BS%n", it->size() / (1024.0 * 1024.0));
""",
"""bool server_prompt_cache::evict_oldest() {
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (it->pinned) {
            continue;
        }

        // customware (entry 014): a segment stays while a tail (pinned or not) still needs it, so the tails go first
        if (it->segment) {
            if (&*it == seg_hold) {
                continue;
            }

            bool needed = false;
            for (const auto & other : states) {
                if (!other.segment && other.p0 > 0 && other.prompt.tokens.size() > 0 &&
                        it->prompt.tokens.size() >= (size_t) other.p0 &&
                        (int32_t) it->prompt.tokens.get_common_prefix(other.prompt.tokens) >= other.p0) {
                    needed = true;
                    break;
                }
            }
            if (needed) {
                continue;
            }
        }

        SRV_WRN(" - cache limit reached, removing oldest entry (size = %.3f MiB)%BS%n", it->size() / (1024.0 * 1024.0));
"""),
# load_exact
(
"""bool server_prompt_cache::load_exact(server_prompt_cache_state * state, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    if (!state) {
        return false;
    }

    bool ok = true;
    {
        auto & data = state->data.main;
        const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), data.size(), id_slot, 0);
        if (n != data.size()) {
            SRV_ERR("failed to restore the suspended state with size %zu%BS%n", data.size());
            ok = false;
        }
    }
    if (ok && !state->data.drft.empty()) {
        GGML_ASSERT(ctx_dft);
        auto & data = state->data.drft;
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), data.size(), id_slot, 0);
        if (n != data.size()) {
            SRV_ERR("failed to restore the suspended draft state with size %zu%BS%n", data.size());
            ok = false;
        }
    }

    if (ok) {
        erase(state);
    }

    return ok;
}
""",
"""bool server_prompt_cache::load_exact(server_prompt_cache_state * state, const server_tokens & tokens, const server_prefix_holders & holders, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, server_prompt_restore & out) {
    if (!state) {
        return false;
    }

    // customware (entry 014): a parked tail re-attaches its prefix like a cached one; the parked state stays for
    // another attempt when the restore fails
    if (!restore(*state, tokens, holders, ctx_tgt, ctx_dft, id_slot, out)) {
        return false;
    }

    erase(state);

    return true;
}

// customware (entry 014): tails and prefix segments

int32_t server_prompt_cache::tail_min() {
    static const int32_t n = [] {
        const char * env = getenv("LLAMA_TAIL_MIN");
        return env ? (int32_t) atoi(env) : 4096;
    }();

    return n;
}

bool server_prompt_cache::has_prompt(const server_tokens & tokens) const {
    for (const auto & state : states) {
        if (state.pinned || state.segment) {
            continue;
        }
        if (state.prompt.tokens.get_common_prefix(tokens) == tokens.size()) {
            return true;
        }
    }

    return false;
}

const server_prefix_holder * server_prompt_cache::find_holder(const server_tokens & tokens, int32_t c, const server_prefix_holders & holders, int32_t id_self) {
    if (c <= 0) {
        return nullptr;
    }

    for (const auto & h : holders) {
        if (h.id == id_self || h.tokens == nullptr || h.n_tokens < c) {
            continue;
        }
        if ((int32_t) h.tokens->get_common_prefix(tokens) >= c) {
            return &h;
        }
    }

    return nullptr;
}

const server_prompt_cache_state * server_prompt_cache::find_segment(const server_tokens & tokens, int32_t c) const {
    if (c <= 0) {
        return nullptr;
    }

    for (const auto & state : states) {
        if (!state.segment || state.prompt.tokens.size() < (size_t) c) {
            continue;
        }
        if ((int32_t) state.prompt.tokens.get_common_prefix(tokens) >= c) {
            return &state;
        }
    }

    return nullptr;
}

int32_t server_prompt_cache::split_point(const server_tokens & tokens, const server_prefix_holders & holders, int32_t id_self, const server_prefix_holder ** holder, const server_prompt_cache_state ** segment) const {
    *holder  = nullptr;
    *segment = nullptr;

    if (tail_min() <= 0) {
        return 0;
    }

    int32_t best = 0;

    for (const auto & h : holders) {
        if (h.id == id_self || h.tokens == nullptr) {
            continue;
        }
        const int32_t c = std::min<int32_t>(h.n_tokens, (int32_t) h.tokens->get_common_prefix(tokens));
        if (c > best) {
            best    = c;
            *holder = &h;
        }
    }

    for (const auto & state : states) {
        if (!state.segment) {
            continue;
        }
        const int32_t c = std::min<int32_t>((int32_t) state.prompt.tokens.size(), (int32_t) state.prompt.tokens.get_common_prefix(tokens));
        if (c > best) {
            best     = c;
            *holder  = nullptr;
            *segment = &state;
        }
    }

    if (best < tail_min()) {
        *holder  = nullptr;
        *segment = nullptr;
        return 0;
    }

    if (*segment == nullptr) {
        *segment = find_segment(tokens, best);
    }

    return best;
}

server_prompt_cache_state * server_prompt_cache::alloc_segment(const server_tokens & tokens, int32_t c, size_t state_size) {
    if (limit_size > 0 && state_size > limit_size) {
        SRV_WRN(" - prefix segment size %.3f MiB exceeds cache size limit %.3f MiB, skipping%BS%n",
                state_size / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    if (limit_size > 0) {
        while (!states.empty() && size() + state_size > limit_size) {
            if (!evict_oldest()) {
                break;
            }
        }
    }

    if (!make_room(state_size)) {
        SRV_WRN(" - not saving a prefix segment (%.3f MiB): RAM is below the page-cache reserve%BS%n", state_size / (1024.0 * 1024.0));
        return nullptr;
    }

    std::vector<uint8_t> data;
    try {
        data.resize(state_size);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for a prefix segment: %s%BS%n", e.what());
        return nullptr;
    }

    server_prompt_cache_state st;
    st.prompt.tokens = tokens.clone();
    st.prompt.tokens.keep_first(c);
    st.data.main     = std::move(data);
    st.segment       = true;

    states.push_back(std::move(st));

    return &states.back();
}

server_prompt_cache_state * server_prompt_cache::save(const server_prompt & prompt, const server_prefix_holders & holders, int32_t id, llama_context * ctx_tgt, llama_context * ctx_dft, bool pinned) {
    const int32_t n = (int32_t) prompt.tokens.size();

    if (!pinned && has_prompt(prompt.tokens)) {
        SRV_TRC("%s", " - prompt is already in the cache, skipping%BS%n");
        return nullptr;
    }

    llama_memory_t mem = llama_get_memory(ctx_tgt);

    const server_prefix_holder      * holder  = nullptr;
    const server_prompt_cache_state * segment = nullptr;

    int32_t c = split_point(prompt.tokens, holders, id, &holder, &segment);

    // the window needs a plain attention cache
    if (c > 0 && !llama_memory_set_state_range(mem, -1, -1)) {
        c = 0;
    }

    if (c > 0 && segment == nullptr) {
        // no segment covers the prefix yet: one is written from this sequence's own cells, attention only
        size_t size_seg = 0;
        {
            server_state_range window(mem, 0, c);
            size_seg = llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY);
        }

        auto * seg = alloc_segment(prompt.tokens, c, size_seg);
        if (seg != nullptr) {
            server_state_range window(mem, 0, c);
            llama_state_seq_get_data_ext(ctx_tgt, seg->data.main.data(), size_seg, id, LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY);

            SRV_INF(" - wrote a %d-token prefix segment (%.1f MiB) from slot %d, for the tails of its family%BS%n", c, size_seg / (1024.0 * 1024.0), id);
            segment = seg;
        } else {
            SRV_WRN(" - no room for a %d-token prefix segment: saving the whole sequence%BS%n", c);
            c = 0;
        }
    }

    seg_hold = segment;

    size_t size_tgt = 0;
    {
        server_state_range window(mem, c > 0 ? c : -1, -1);
        size_tgt = llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
    }
    const size_t size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

    SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)%BS%n",
            n, (size_tgt + size_dft) / (1024.0 * 1024.0), size_dft / (1024.0 * 1024.0));

    auto * cur = pinned ? alloc_pinned(size_tgt, size_dft) : alloc(prompt, size_tgt, size_dft);

    seg_hold = nullptr;

    if (cur == nullptr) {
        return nullptr;
    }

    {
        server_state_range window(mem, c > 0 ? c : -1, -1);
        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
    }
    if (ctx_dft) {
        llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
    }

    cur->p0 = c;

    if (c > 0) {
        // a pinned tail keeps its tokens for the coverage checks (the slot keeps them as well)
        if (pinned) {
            cur->prompt.tokens = prompt.tokens.clone();
        }

        if (holder != nullptr) {
            SRV_INF(" - saved slot %d as a tail: %d of %d tokens behind the prefix held by slot %d (%.1f MiB)%BS%n",
                    id, n - c, n, holder->id, (size_tgt + size_dft) / (1024.0 * 1024.0));
        } else {
            SRV_INF(" - saved slot %d as a tail: %d of %d tokens behind a %d-token segment in RAM (%.1f MiB)%BS%n",
                    id, n - c, n, (int) segment->prompt.tokens.size(), (size_tgt + size_dft) / (1024.0 * 1024.0));
        }
    }

    return cur;
}

bool server_prompt_cache::restore(server_prompt_cache_state & state, const server_tokens & tokens, const server_prefix_holders & holders, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot, server_prompt_restore & out) {
    out = server_prompt_restore{};

    llama_memory_t mem = llama_get_memory(ctx_tgt);

    // the sequence starts empty: a whole restore removes it anyway, a windowed one must not find an earlier
    // conversation's cells below p0
    llama_memory_seq_rm(mem, id_slot, -1, -1);

    const int32_t p0 = state.p0;

    if (p0 > 0) {
        if (const auto * holder = find_holder(tokens, p0, holders, id_slot)) {
            // labels only: the prefix cells stay the holder's and become this sequence's too
            llama_memory_seq_cp(mem, holder->id, id_slot, 0, p0);

            out.fork_of  = holder->id;
            out.n_shared = p0;
        } else if (const auto * seg = find_segment(tokens, p0)) {
            const auto & data = seg->data.main;

            server_state_range window(mem, 0, -1);
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), data.size(), id_slot, LLAMA_STATE_SEQ_FLAGS_ATTN_ONLY);
            if (n != data.size()) {
                SRV_ERR("failed to restore the %d-token prefix segment (%zu bytes) of a tail of %d tokens%BS%n",
                        (int) seg->prompt.tokens.size(), data.size(), (int) tokens.size() - p0);
                llama_memory_seq_rm(mem, id_slot, -1, -1);
                return false;
            }

            out.n_segment = (int32_t) seg->prompt.tokens.size();
        } else {
            SRV_WRN("the tail of %d tokens behind %d has neither a holder nor a segment for its prefix: cannot restore it%BS%n",
                    (int) tokens.size() - p0, p0);
            return false;
        }
    }

    {
        auto & data = state.data.main;

        server_state_range window(mem, p0 > 0 ? p0 : -1, -1);
        const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), data.size(), id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (n != data.size()) {
            SRV_ERR("failed to restore state with size %zu%BS%n", data.size());
            llama_memory_seq_rm(mem, id_slot, -1, -1);
            return false;
        }
    }

    {
        auto & data = state.data.drft;

        if (!data.empty()) {
            GGML_ASSERT(ctx_dft);

            const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), data.size(), id_slot, LLAMA_STATE_SEQ_FLAGS_NONE);
            if (n != data.size()) {
                SRV_WRN("failed to restore draft state with size %zu%BS%n", data.size());
                llama_memory_seq_rm(mem, id_slot, -1, -1);
                return false;
            }
        }
    }

    out.n_tail = (int32_t) tokens.size() - p0;

    // the state is in the pool; the entry's buffers go (the caller erases the entry)
    state.data.main.clear();
    state.data.main.shrink_to_fit();
    state.data.drft.clear();
    state.data.drft.shrink_to_fit();

    return true;
}
"""),
])

# ============================================================================================ server-context.cpp
patch('tools/server/server-context.cpp', [
# the slot: holders callback
(
"""    std::function<void(server_slot &)>       callback_on_turn_end; // customware: a generation ended, state still live
""",
"""    std::function<void(server_slot &)>       callback_on_turn_end; // customware: a generation ended, state still live

    // customware (entry 014): the sequences a tail of this slot can re-attach its prefix to (this slot excluded)
    std::function<server_prefix_holders(const server_slot &)> callback_prefix_holders;

    server_prefix_holders prefix_holders() const {
        return callback_prefix_holders ? callback_prefix_holders(*this) : server_prefix_holders{};
    }
"""),
# suspend
(
"""        const size_t size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        auto * st = cache_for_suspend->alloc_pinned(size_tgt, size_dft);
        if (st == nullptr) {
            SLT_ERR(*this, "cannot park the sequence state (%.1f MiB) in the prompt cache%BS%n", (size_tgt + size_dft) / (1024.0 * 1024.0));
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, st->data.main.data(), size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, st->data.drft.data(), size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        mem.seq_rm(id, -1, -1);
""",
"""        // customware (entry 014): a tail when another sequence or a segment covers the prefix: the pinned entry then
        // holds the private cells only
        auto * st = cache_for_suspend->save(prompt, prefix_holders(), id, ctx_tgt, ctx_dft, true);
        if (st == nullptr) {
            SLT_ERR(*this, "%s", "cannot park the sequence state in the prompt cache%BS%n");
            return false;
        }

        const size_t size_tgt = st->data.main.size();
        const size_t size_dft = st->data.drft.size();

        mem.seq_rm(id, -1, -1);
"""),
# resume
(
"""        const bool ok = cache_for_suspend->load_exact(suspended_state, ctx_tgt, ctx_dft, id);

        if (!ok) {
            // the parked state stays in the cache for another attempt
            SLT_ERR(*this, "%s", "failed to restore the parked sequence state%BS%n");
            return false;
        }

        suspended_state = nullptr;
        resume_failures = 0;
""",
"""        server_prompt_restore rs;

        const bool ok = cache_for_suspend->load_exact(suspended_state, prompt.tokens, prefix_holders(), ctx_tgt, ctx_dft, id, rs);

        if (!ok) {
            // the parked state stays in the cache for another attempt
            SLT_ERR(*this, "%s", "failed to restore the parked sequence state%BS%n");
            return false;
        }

        suspended_state = nullptr;
        resume_failures = 0;

        // customware (entry 014): a parked tail came back behind its prefix: shared again, or private from a segment
        fork_of       = rs.n_shared > 0 ? rs.fork_of : -1;
        n_fork_shared = rs.n_shared;
        if (rs.n_shared > 0) {
            SLT_INF(*this, "restored a parked tail of %d tokens behind the %d shared with slot %d%BS%n", rs.n_tail, rs.n_shared, rs.fork_of);
        } else if (rs.n_segment > 0) {
            SLT_INF(*this, "restored a parked tail of %d tokens behind %d cells restored from a prefix segment%BS%n", rs.n_tail, rs.n_segment);
        }
"""),
# prompt_save
(
"""    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        const size_t cur_size_tgt =           llama_state_seq_get_size_ext(ctx_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        const size_t cur_size_dft = ctx_dft ? llama_state_seq_get_size_ext(ctx_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;

        const size_t cur_size = cur_size_tgt + cur_size_dft;

        SRV_TRC(" - saving prompt with length %d, total state size = %.3f MiB (draft: %.3f MiB)%BS%n",
                (int) prompt.tokens.size(), cur_size / (1024.0 * 1024.0), cur_size_dft / (1024.0 * 1024.0));

        auto * cur = prompt_cache.alloc(prompt, cur_size_tgt, cur_size_dft);
        if (cur == nullptr) {
            return false;
        }

        llama_state_seq_get_data_ext(ctx_tgt, cur->data.main.data(), cur_size_tgt, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        if (ctx_dft) {
            llama_state_seq_get_data_ext(ctx_dft, cur->data.drft.data(), cur_size_dft, id, LLAMA_STATE_SEQ_FLAGS_NONE);
        }

        return true;
    }
""",
"""    bool prompt_save(server_prompt_cache & prompt_cache) const {
        if (prompt.tokens.size() == 0) {
            return false;
        }

        // customware (entry 014): a tail when another sequence, or a segment in RAM, covers a prefix of this conversation
        auto * cur = prompt_cache.save(prompt, prefix_holders(), id, ctx_tgt, ctx_dft, false);

        return cur != nullptr;
    }
"""),
# prompt_load
(
"""    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        bool res = prompt_cache.load(prompt, tokens, ctx_tgt, ctx_dft, id);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache%BS%n");
        }

        return res;
    }
""",
"""    bool prompt_load(server_prompt_cache & prompt_cache, const server_tokens & tokens) {
        server_prompt_restore rs;

        bool res = prompt_cache.load(prompt, tokens, prefix_holders(), ctx_tgt, ctx_dft, id, rs);
        if (!res) {
            SLT_WRN(*this, "%s", "failed to load prompt from cache%BS%n");
            return res;
        }

        // customware (entry 014): a tail came back behind its prefix: shared with a live holder, or private from a segment
        fork_of       = rs.n_shared > 0 ? rs.fork_of : -1;
        n_fork_shared = rs.n_shared;
        if (rs.n_shared > 0) {
            SLT_INF(*this, "restored a tail of %d tokens behind the %d shared with slot %d; pool %u of %d cells%BS%n",
                    rs.n_tail, rs.n_shared, rs.fork_of, llama_memory_n_cells_used(llama_get_memory(ctx_tgt)), n_ctx);
        } else if (rs.n_segment > 0) {
            SLT_INF(*this, "restored a tail of %d tokens behind %d cells restored from a prefix segment; pool %u of %d cells%BS%n",
                    rs.n_tail, rs.n_segment, llama_memory_n_cells_used(llama_get_memory(ctx_tgt)), n_ctx);
        }

        return res;
    }
"""),
# init: the callback
(
"""        // customware: suspended slots park their state in the prompt cache's RAM budget
        for (auto & slot : slots) {
            slot.cache_for_suspend = prompt_cache.get();
        }
""",
"""        // customware: suspended slots park their state in the prompt cache's RAM budget
        for (auto & slot : slots) {
            slot.cache_for_suspend = prompt_cache.get();

            // customware (entry 014): the sequences a tail can re-attach its prefix to
            slot.callback_prefix_holders = [this](const server_slot & s) {
                return prefix_holders(&s);
            };
        }
"""),
# get_available_slot: restore the tail next to the idle sibling instead of taking it over
(
"""                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }
        }

        // customware (entry 012): when a busy slot holds a longer prefix of the new prompt than any idle one, the
""",
"""                // if we are about to lose a large portion of the existing context - save it in the prompt cache
                if (f_keep < 0.5f) {
                    update_cache = true;
                }
            }

            // customware (entry 014): when the cache holds more of this conversation than the idle slot does (its
            // tail, behind the prefix that slot holds), the tail is restored into another idle slot and re-attached:
            // both conversations stay whole in the pool, sharing the prefix, instead of the takeover cutting the
            // sibling's tail. Only when another idle slot is there.
            if (ret != nullptr && !update_cache && prompt_cache && params_base.kv_unified &&
                    task.type == SERVER_TASK_TYPE_COMPLETION && task.params.cache_prompt) {
                const size_t lcp_ret = ret->prompt.tokens.get_common_prefix(task.tokens);
                const auto   pk      = prompt_cache->peek(ret->prompt, task.tokens, prefix_holders(nullptr));

                if (pk.n_tokens > 0 && pk.holder >= 0 && (size_t) pk.n_tokens > lcp_ret) {
                    server_slot * dst = nullptr;
                    for (auto & slot : slots) {
                        if (&slot == ret || slot.is_processing()) {
                            continue;
                        }
                        const bool empty = slot.prompt.tokens.size() == 0;
                        if (dst == nullptr ||
                            (empty && dst->prompt.tokens.size() > 0) ||
                            (empty == (dst->prompt.tokens.size() == 0) && slot.t_last_used < dst->t_last_used)) {
                            dst = &slot;
                        }
                    }

                    if (dst != nullptr) {
                        SLT_INF(*dst, "the cache holds %lld tokens of this conversation behind the %zu that slot %d has: restoring the tail here, next to it\n",
                                (long long) pk.n_tokens, lcp_ret, ret->id);
                        ret = dst;
                        update_cache = true;
                    }
                }
            }
        }

        // customware (entry 012): when a busy slot holds a longer prefix of the new prompt than any idle one, the
"""),
# LRU choice avoids the holder
(
"""        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            for (server_slot & slot : slots) {
                // skip the slot if it is not available
                if (slot.is_processing()) {
                    continue;
                }

                // select the current slot if the criteria match
                if (!ret || slot.t_last_used <= t_last) {
                    t_last = slot.t_last_used;
                    ret = &slot;
                }
            }
""",
"""        // find the slot that has been least recently used
        if (ret == nullptr) {
            int64_t t_last = -1;

            // customware (entry 014): a cached tail re-attaches its prefix to a live holder; a restore into the
            // holder's own slot would lose that sharing, so another idle slot is preferred when there is one
            int holder = -1;
            if (prompt_cache && params_base.kv_unified && task.type == SERVER_TASK_TYPE_COMPLETION && task.params.cache_prompt) {
                holder = prompt_cache->peek(server_prompt{}, task.tokens, prefix_holders(nullptr)).holder;
            }

            for (int pass = 0; pass < 2 && ret == nullptr; ++pass) {
                for (server_slot & slot : slots) {
                    // skip the slot if it is not available
                    if (slot.is_processing()) {
                        continue;
                    }
                    if (pass == 0 && slot.id == holder) {
                        continue;
                    }

                    // select the current slot if the criteria match
                    if (!ret || slot.t_last_used <= t_last) {
                        t_last = slot.t_last_used;
                        ret = &slot;
                    }
                }
            }
"""),
# the restore path: need = cells, the holder is never purged for room
(
"""                if (params_base.kv_unified) {
                    const int64_t need = prompt_cache->peek(ret->prompt, task.tokens);
""",
"""                if (params_base.kv_unified) {
                    // customware (entry 014): a tail behind a live holder costs its own cells only, and the holder is
                    // never purged to make room for it
                    const auto    pk   = prompt_cache->peek(ret->prompt, task.tokens, prefix_holders(ret));
                    const int64_t need = pk.n_tokens > 0 ? pk.n_cells : -1;
"""),
(
"""                            server_slot * idle = nullptr;
                            for (auto & slot : slots) {
                                if (&slot == ret || slot.is_processing() || slot.prompt.n_tokens() == 0) {
                                    continue;
                                }
""",
"""                            server_slot * idle = nullptr;
                            for (auto & slot : slots) {
                                if (&slot == ret || slot.is_processing() || slot.prompt.n_tokens() == 0 || slot.id == pk.holder) {
                                    continue;
                                }
"""),
# prefix_holders builder, next to the cells helpers
(
"""    int32_t kv_cells_owned(const std::vector<llama_seq_id> & ids) const {
""",
"""    // customware (entry 014): the sequences in the pool a tail can re-attach its prefix to: every slot with cells
    // (not suspended, not waiting on a parent), `exclude` left out; n_tokens = how many of its tokens are in the pool
    server_prefix_holders prefix_holders(const server_slot * exclude) const {
        server_prefix_holders res;

        llama_memory_t mem = llama_get_memory(ctx_tgt);

        for (const auto & slot : slots) {
            if (&slot == exclude || slot.prompt.tokens.size() == 0 ||
                    slot.state == SLOT_STATE_SUSPENDED || slot.state == SLOT_STATE_WAIT_OTHER) {
                continue;
            }
            if (llama_memory_seq_pos_min(mem, slot.id) != 0) {
                continue;
            }
            const int32_t n = std::min<int32_t>((int32_t) slot.prompt.tokens.size(), llama_memory_seq_pos_max(mem, slot.id) + 1);
            if (n <= 0) {
                continue;
            }
            res.push_back({ slot.id, &slot.prompt.tokens, n });
        }

        return res;
    }

    int32_t kv_cells_owned(const std::vector<llama_seq_id> & ids) const {
"""),
# plan_placement: the cache case
(
"""        if (prompt_cache) {
            const int64_t entry = prompt_cache->peek(server_prompt{}, task.tokens);
            if (entry > 0) {
                // the entry is restored whole, then trimmed and extended
                pl.need = kv_need(std::max((int32_t) entry, n), task.params.n_predict);
                return pl;
            }
        }
""",
"""        if (prompt_cache) {
            const auto pk = prompt_cache->peek(server_prompt{}, task.tokens, prefix_holders(nullptr));
            if (pk.n_tokens > 0) {
                // the entry is restored (whole, or its tail behind a live holder), then trimmed and extended
                pl.need = kv_need((int32_t) pk.n_cells + std::max(0, n - (int32_t) pk.n_tokens), task.params.n_predict);
                return pl;
            }
        }
"""),
# build_sched_input: a parked tail behind a live holder
(
"""        in.now          = ggml_time_us();
        in.t_age        = sched_age_us();

        for (const auto & slot : slots) {
            if (slot.state == SLOT_STATE_SUSPENDED) {
                sched_req r;
                r.id        = slot.task->id;
                r.prio      = slot.task->params.priority;
                r.t_arrival = slot.task->t_arrival;
                r.state     = SCHED_REQ_SUSPENDED;
                r.need      = slot.prompt.n_tokens() + kv_resume_margin();
""",
"""        in.now          = ggml_time_us();
        in.t_age        = sched_age_us();

        const server_prefix_holders holders = prefix_holders(nullptr);

        for (const auto & slot : slots) {
            if (slot.state == SLOT_STATE_SUSPENDED) {
                sched_req r;
                r.id        = slot.task->id;
                r.prio      = slot.task->params.priority;
                r.t_arrival = slot.task->t_arrival;
                r.state     = SCHED_REQ_SUSPENDED;
                r.need      = slot.prompt.n_tokens() + kv_resume_margin();
                // customware (entry 014): a parked tail behind a live holder comes back with its own cells only
                if (slot.suspended_state != nullptr && slot.suspended_state->p0 > 0 &&
                        server_prompt_cache::find_holder(slot.prompt.tokens, slot.suspended_state->p0, holders, slot.id) != nullptr) {
                    r.need -= slot.suspended_state->p0;
                }
"""),
])

print('patch-014b: done')
