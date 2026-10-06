#!/usr/bin/env python
"""Entry 012 server-level patch (kept for the record; the committed source is the result).
Run from the repo root once. %BS% stands for a backslash in the C source."""
p = 'tools/server/server-context.cpp'
s = open(p, encoding='utf-8', newline='').read()
nl = chr(13) + chr(10) if (chr(13) + chr(10)) in s else chr(10)
bs = chr(92)

assert 'n_forks' not in s, 'already patched'


def rep(old, new, count=1):
    global s
    old = old.replace('\n', nl).replace('%BS%', bs)
    new = new.replace('\n', nl).replace('%BS%', bs)
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


# 1. counters
rep('''    int64_t n_kv_preempted = 0; // sequences suspended by a pool overflow
''', '''    int64_t n_kv_preempted = 0; // sequences suspended by a pool overflow
    int64_t n_forks        = 0; // customware (entry 012): requests that started as a fork of a busy slot's conversation
''')

# 2. accounting: share-aware idle_retained + kv_cells_owned; kv_committed with an excluded slot
rep('''    // cells held by idle slots that kept their context (purgeable)
    int32_t kv_idle_retained() const {
        int32_t n = 0;
        for (const auto & slot : slots) {
            if (!slot.is_processing()) {
                n += slot.prompt.n_tokens();
            }
        }
        return n;
    }
''', '''    // cells the given sequences own together (every label in the set): what removing all of them frees. In a
    // unified pool a fork shares its prefix with its source, so this is not the sum of token counts.
    int32_t kv_cells_owned(const std::vector<llama_seq_id> & ids) const {
        if (ids.empty()) {
            return 0;
        }
        return (int32_t) llama_memory_n_cells_owned(llama_get_memory(ctx_tgt), ids.data(), ids.size());
    }

    // cells held by idle slots that kept their context and that purging them all would free; `exclude` is an idle
    // slot about to be reused, whose cells stay
    int32_t kv_idle_retained(const server_slot * exclude = nullptr) const {
        std::vector<llama_seq_id> ids;
        for (const auto & slot : slots) {
            if (!slot.is_processing() && &slot != exclude && slot.prompt.n_tokens() > 0) {
                ids.push_back(slot.id);
            }
        }
        return kv_cells_owned(ids);
    }
''')
rep('''    int32_t kv_committed(bool include_suspended) const {
        const int32_t used = kv_cells_used();
        if (used == 0 && kv_idle_retained() + kv_resident(false) > 0) {
            return kv_resident(include_suspended);
        }
        int32_t n = used - kv_idle_retained();
''', '''    int32_t kv_committed(bool include_suspended, const server_slot * exclude_idle = nullptr) const {
        const int32_t used = kv_cells_used();
        if (used == 0 && kv_idle_retained() + kv_resident(false) > 0) {
            return kv_resident(include_suspended);
        }
        int32_t n = used - kv_idle_retained(exclude_idle);
''')

# 3. pick_victim by owned cells
rep('''        server_slot * best = nullptr;
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.task) {
                continue;
            }
            if (slot.task->is_child() || slot.task->is_parent() || !slot.spec_draft.empty()) {
                continue;
            }
            if (best == nullptr ||
                slot.prompt.n_tokens() <  best->prompt.n_tokens() ||
                (slot.prompt.n_tokens() == best->prompt.n_tokens() && slot.task->id > best->task->id)) {
                best = &slot;
            }
        }
        return best;
''', '''        server_slot * best = nullptr;
        int32_t best_owned = 0;
        for (auto & slot : slots) {
            if (slot.state != SLOT_STATE_GENERATING || !slot.task) {
                continue;
            }
            if (slot.task->is_child() || slot.task->is_parent() || !slot.spec_draft.empty()) {
                continue;
            }
            // customware (entry 012): what a suspension frees is what the slot owns alone, not its token count
            const int32_t owned = kv_cells_owned({ slot.id });
            if (best == nullptr ||
                owned <  best_owned ||
                (owned == best_owned && slot.task->id > best->task->id)) {
                best = &slot;
                best_owned = owned;
            }
        }
        return best;
''')

# 4. try_clear_idle_slots: the idle slot that owns the most cells; none owning any -> nothing to free
rep('''        for (auto & slot : slots) {
            if (slot.is_processing()) {
                continue;
            }

            if (slot.prompt.n_tokens() > 0) {
                SRV_WRN("purging slot %d with %zu tokens%BS%n", slot.id, slot.prompt.tokens.size());

                // customware: a purge is not a cache miss: the prompt goes to the cache first, as a launch would do
                if (prompt_cache && params_base.cache_idle_slots) {
                    slot.prompt_save(*prompt_cache);
                    prompt_cache->update();
                }

                slot.prompt_clear();

                res = true;

                // clear slots one by one
                break;
            }
        }

        return res;
''', '''        // customware (entry 012): purge the idle slot whose cells free the most; one that shares all of its cells
        // with a busy fork source frees nothing and is left alone (its save would only cost RAM)
        server_slot * victim = nullptr;
        int32_t victim_owned = 0;
        for (auto & slot : slots) {
            if (slot.is_processing() || slot.prompt.n_tokens() == 0) {
                continue;
            }
            const int32_t owned = kv_cells_owned({ slot.id });
            if (owned > victim_owned) {
                victim = &slot;
                victim_owned = owned;
            }
        }

        if (victim != nullptr) {
            auto & slot = *victim;

            SRV_WRN("purging slot %d with %zu tokens (%d cells of its own)%BS%n", slot.id, slot.prompt.tokens.size(), victim_owned);

            // customware: a purge is not a cache miss: the prompt goes to the cache first, as a launch would do
            if (prompt_cache && params_base.cache_idle_slots) {
                slot.prompt_save(*prompt_cache);
                prompt_cache->update();
            }

            slot.prompt_clear();

            res = true;
        }

        return res;
''')

# 5. admission: charge the suffix, the slot's own cells are not purgeable
rep('''                        const int32_t need = kv_need(task.n_tokens(), task.params.n_predict);
                        const int32_t used = kv_committed(true);
                        const int32_t pool = kv_pool_size();
                        if (used > 0 && used + need > pool) {
                            SRV_WRN("task %d needs %d cells (prompt %d) but %d of %d are spoken for: waiting for cells%BS%n",
                                    id_task, need, task.n_tokens(), used, pool);
''', '''                        // customware (entry 012): the slot's own cells (a continuation's prefix, private or shared with
                        // a busy fork source) are in the pool already: charge the rest, and do not count them purgeable
                        const int32_t n_held = task.params.cache_prompt ? (int32_t) slot->prompt.tokens.get_common_prefix(task.tokens) : 0;
                        const int32_t need = kv_need(task.n_tokens() - n_held, task.params.n_predict);
                        const int32_t used = kv_committed(true, slot);
                        const int32_t pool = kv_pool_size();
                        if (used > 0 && used + need > pool) {
                            SRV_WRN("task %d needs %d cells (prompt %d, %d of it in the slot) but %d of %d are spoken for: waiting for cells%BS%n",
                                    id_task, need, task.n_tokens(), n_held, used, pool);
''')

# 6. the fork step in get_available_slot, before the LRU fallback
rep('''        // find the slot that has been least recently used
        if (ret == nullptr) {
''', '''        // customware (entry 012): when a busy slot holds a longer prefix of the new prompt than any idle one, the
        // request becomes a fork of that conversation: it shares the prefix cells and starts from the source's
        // checkpoint at the fork point. Idle sources are taken over as before (ties go to idle).
        if (params_base.kv_unified && task.id_slot == -1 && task.type == SERVER_TASK_TYPE_COMPLETION &&
                task.params.cache_prompt && slot_prompt_similarity != 0.0f && task.tokens.size() > 0) {
            server_slot * src = nullptr;
            size_t lcp_src = 0;
            for (auto & slot : slots) {
                if (!slot.is_processing() || slot.state == SLOT_STATE_SUSPENDED || slot.state == SLOT_STATE_WAIT_OTHER) {
                    continue;
                }
                if (slot.prompt.tokens.size() == 0) {
                    continue;
                }
                const size_t lcp = slot.prompt.tokens.get_common_prefix(task.tokens);
                if (lcp > lcp_src) {
                    lcp_src = lcp;
                    src = &slot;
                }
            }

            const size_t lcp_idle = ret ? ret->prompt.tokens.get_common_prefix(task.tokens) : 0;

            if (src != nullptr && lcp_src > lcp_idle && float(lcp_src) / task.tokens.size() > slot_prompt_similarity) {
                // the destination: an empty idle slot, else the least recently used idle one
                server_slot * dst = nullptr;
                for (auto & slot : slots) {
                    if (slot.is_processing()) {
                        continue;
                    }
                    const bool empty = slot.prompt.tokens.size() == 0;
                    if (dst == nullptr ||
                        (empty && dst->prompt.tokens.size() > 0) ||
                        (empty == (dst->prompt.tokens.size() == 0) && slot.t_last_used < dst->t_last_used)) {
                        dst = &slot;
                    }
                }

                if (dst == nullptr) {
                    SRV_INF("task %d shares %zu of its %zu tokens with the busy slot %d and no slot is free to fork into: waiting%BS%n",
                            task.id, lcp_src, task.tokens.size(), src->id);
                    return nullptr;
                }

                // the fork point: the source's latest checkpoint at or before the common prefix, leaving at least one
                // token to process
                const common_prompt_checkpoint * ckpt = nullptr;
                for (const auto & cur : src->prompt.checkpoints) {
                    if (cur.n_tokens <= (int64_t) lcp_src && cur.n_tokens < (int64_t) task.tokens.size() &&
                            (ckpt == nullptr || cur.n_tokens > ckpt->n_tokens)) {
                        ckpt = &cur;
                    }
                }

                if (ckpt == nullptr || float(ckpt->n_tokens) / task.tokens.size() <= slot_prompt_similarity) {
                    SRV_INF("task %d shares %zu tokens with the busy slot %d but it has no checkpoint there to fork from: not forking%BS%n",
                            task.id, lcp_src, src->id);
                } else {
                    const int32_t need      = kv_need((int32_t) task.tokens.size() - (int32_t) ckpt->n_tokens, task.params.n_predict);
                    const int32_t committed = kv_committed(true);
                    const int32_t pool      = kv_pool_size();

                    if (committed + need > pool) {
                        SRV_INF("task %d could fork from slot %d at %lld tokens but needs %d more cells next to the %d spoken for: not forking%BS%n",
                                task.id, src->id, (long long) ckpt->n_tokens, need, committed);
                    } else {
                        // the destination's own conversation goes to the cache first, as the LRU path does
                        if (prompt_cache && dst->prompt.tokens.size() > 0) {
                            dst->prompt_save(*prompt_cache);
                        }

                        dst->fork_from(*src, *ckpt);

                        if (prompt_cache) {
                            prompt_cache->update();
                        }

                        n_forks++;

                        SRV_INF("task %d: fork of slot %d at %lld tokens (common prefix %zu of %zu), %zu tokens to process%BS%n",
                                task.id, src->id, (long long) ckpt->n_tokens, lcp_src, task.tokens.size(),
                                task.tokens.size() - (size_t) ckpt->n_tokens);

                        return dst;
                    }
                }
            }
        }

        // find the slot that has been least recently used
        if (ret == nullptr) {
''')

# 7. the turn-end checkpoint: method after create_checkpoint, callback wired at slot init
rep('''    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
''', '''    // customware (entry 012): the state at the end of a generation, taken when the generation ends. Before, it was
    // taken by the next request at the position where it started adding, if one came; a fork needs it now.
    void checkpoint_at_turn_end(server_slot & slot) {
        if (params_base.n_ctx_checkpoints <= 0) {
            return;
        }
        if (!slot.task || slot.task->type != SERVER_TASK_TYPE_COMPLETION) {
            return;
        }
        if (!(ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
              ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS ||
              n_swa > 0)) {
            return;
        }
        if (!slot.prompt.checkpoints.empty() && slot.prompt.checkpoints.back().n_tokens == slot.prompt.n_tokens()) {
            return;
        }

        const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
        const auto pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), slot.id);
        if (pos_min < 0 || pos_max < 0) {
            return;
        }

        create_checkpoint(slot, 0, pos_min, pos_max);
    }

    // returns false to decline the task, it is offered again after the decode is done
    bool process_single_task(server_task && task, bool is_yielding) {
''')
rep('''            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };
''', '''            slot.callback_on_reset = [this](const server_slot & slot) {
                // flush the generated token stats before reset()
                if (slot.stats.n_gen > 0) {
                    metrics_on_prediction(slot);
                }
            };

            slot.callback_on_turn_end = [this](server_slot & slot) {
                checkpoint_at_turn_end(slot);
            };
''')

# 8. no duplicate of the turn-end checkpoint when the next request starts there
rep('''                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);
''', '''                    do_checkpoint = do_checkpoint && (
                            slot.prompt.checkpoints.empty() ||
                            is_last_user_message || near_prompt_end ||
                            n_tokens_start > slot.prompt.checkpoints.back().n_tokens + params_base.checkpoint_min_step);

                    // customware (entry 012): the turn-end checkpoint already holds this state
                    do_checkpoint = do_checkpoint &&
                            !(!slot.prompt.checkpoints.empty() && slot.prompt.checkpoints.back().n_tokens == n_tokens_start);
''')

open(p, 'w', encoding='utf-8', newline='').write(s)
print('server-level edits applied')
