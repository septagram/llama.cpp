#!/usr/bin/env python
"""Review iteration 1 fixes (plan 26d-blade-august-ozone): the seven upheld findings of review
26d-battle-expel-living. Exact-string edits; every anchor must match exactly once."""
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

# ---------------------------------------------------------------- finding 1 + 3: the decision
patch('tools/server/server-sched.h', [
(
"""    int64_t now     = 0;       // us
    int64_t t_age   = 0;       // a request waiting longer than this (us) is promoted one class; 0 = off
    std::vector<sched_req>  reqs;
""",
"""    int64_t now     = 0;       // us
    int64_t t_age   = 0;       // a request waiting longer than this (us) is promoted one class; 0 = off
    // cells in the pool that no single running request holds alone and that purging every idle conversation
    // would not free: a prefix shared between a running request and anything else (a fork and its source, a tail
    // re-attached to its holder). Nobody's `held` or `owned`, so the walk is seeded with them
    int32_t shared  = 0;
    std::vector<sched_req>  reqs;
"""),
])

patch('tools/server/server-sched.cpp', [
(
"""    // the walk: idle cells count as free (they can be purged); the purge list is settled afterwards
    int32_t cells_used = 0;
""",
"""    // the walk: idle cells count as free (they can be purged); the purge list is settled afterwards. Cells shared
    // with a running request are taken whatever is purged, so they are charged up front
    int32_t cells_used = in.shared;
"""),
(
"""        // what the request costs the pool: its own cells, what it still needs, and what it keeps of the
        // conversation it continues (the rest of that conversation is freed by the takeover)
        const int32_t cost = r->held + r->need + (continues ? std::min(r->reuse_keep, idle_by_id[r->reuse]->owned) : 0);
""",
"""        // what the request costs the pool: its own cells, what it still needs, and what it keeps of the
        // conversation it continues (the rest of that conversation is freed by the takeover). Capped at the pool:
        // a request alone in the pool always fits, whatever its growth reserve (a continuation of a conversation
        // within one reserve of the pool size would otherwise never start, with nothing to wait for)
        const int32_t cost = std::min<int32_t>(r->held + r->need + (continues ? std::min(r->reuse_keep, idle_by_id[r->reuse]->owned) : 0), in.pool);
"""),
])

patch('tests/test-server-sched.cpp', [
(
"""    // 2. a fork pair: each holds only its tail; both fit although their token counts would not
    {
        auto in = base(16384, 2);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 12400 + 1500, 256, 0, 10)); // the parent owns the prefix and its tail
        in.reqs.push_back(req(2, SCHED_REQ_RUNNING, 1500, 256, 0, 20));         // the fork owns its tail
        auto out = sched_decide(in);
        CHECK(out.keep.size() == 2);
        CHECK(out.suspend.empty());
    }
""",
"""    // 2. a fork pair: each holds only its tail, the prefix is shared (nobody's alone); both fit although their
    //    token counts would not
    {
        auto in = base(16384, 2);
        in.shared = 12400;
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 1500, 256, 0, 10)); // the parent's tail
        in.reqs.push_back(req(2, SCHED_REQ_RUNNING, 1500, 256, 0, 20)); // the fork's tail
        auto out = sched_decide(in);
        CHECK(out.keep.size() == 2);
        CHECK(out.suspend.empty());
    }

    // 2b. the shared prefix is charged: a running fork (tail 1000) next to its idle sibling (owned 1000) share
    //     88000 of a 120064 pool; a cold 50000-token waiter must wait, and purging the sibling frees only its tail
    {
        auto in = base(120064, 3);
        in.shared = 88000;
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 1000, 256, 0, 10));
        in.idle.push_back(conv(1, 1000, 100));
        in.reqs.push_back(req(2, SCHED_REQ_WAITING, 0, 50000 + 256, 0, 20));
        auto out = sched_decide(in);
        CHECK(has(out.keep, 1));
        CHECK(!has(out.start, 2));
        CHECK(out.suspend.empty());
    }

    // 2c. the whole cost is capped at the pool: a conversation of 118000 idle in a 120064 pool, its next turn
    //     (2548 more with the reserve, keeping the 118000) starts with nothing running; before, 120548 never fit
    {
        auto in = base(120064, 2);
        in.idle.push_back(conv(0, 118000, 100));
        in.reqs.push_back(req(1, SCHED_REQ_WAITING, 0, 2548, 0, 10, /*reuse*/ 0, /*keeps*/ 118000));
        auto out = sched_decide(in);
        CHECK(has(out.start, 1));
        CHECK(out.purge.empty());
    }
"""),
])

# ---------------------------------------------------------------- the server
patch('tools/server/server-context.cpp', [
# finding 1: shared cells
(
"""        const server_prefix_holders holders = prefix_holders(nullptr);

        for (const auto & slot : slots) {
            if (slot.state == SLOT_STATE_SUSPENDED) {
                sched_req r;
                r.id        = slot.task->id;
                r.prio      = slot.task->params.priority;
                r.t_arrival = slot.task->t_arrival;
                r.state     = SCHED_REQ_SUSPENDED;
                r.need      = slot.prompt.n_tokens() + kv_resume_margin();
                // customware (entry 014): a parked tail behind a live holder comes back with its own cells only
""",
"""        const server_prefix_holders holders = prefix_holders(nullptr);

        // review fix: the cells shared between a running request and anything else are nobody's `held` or `owned`;
        // they are charged up front. The idle ids are subtracted together, so a prefix shared only between idle
        // conversations stays purgeable
        int32_t held_running = 0;
        std::vector<llama_seq_id> idle_ids;

        for (const auto & slot : slots) {
            if (slot.state == SLOT_STATE_SUSPENDED) {
                sched_req r;
                r.id        = slot.task->id;
                r.prio      = slot.task->params.priority;
                r.t_arrival = slot.task->t_arrival;
                r.state     = SCHED_REQ_SUSPENDED;
                // review fix: a request parked during its prefill comes back to finish it, so the whole prompt is
                // charged, as for a running one (the walk caps the cost at the pool)
                r.need      = std::max(slot.prompt.n_tokens(), slot.task->n_tokens()) + kv_resume_margin();
                // customware (entry 014): a parked tail behind a live holder comes back with its own cells only
"""),
(
"""                r.state     = SCHED_REQ_RUNNING;
                r.held      = kv_cells_owned({ slot.id });
                r.need      = std::max(0, slot.task->n_tokens() - (int32_t) slot.prompt.n_tokens()) + kv_margin();
                r.has_slot  = true;
                in.reqs.push_back(r);
                continue;
            }
            if (slot.prompt.n_tokens() > 0) {
                sched_conv c;
                c.id     = slot.id;
                c.owned  = kv_cells_owned({ slot.id });
                c.t_last = slot.t_last_used;
                in.idle.push_back(c);
            }
        }
""",
"""                r.state     = SCHED_REQ_RUNNING;
                r.held      = kv_cells_owned({ slot.id });
                r.need      = std::max(0, slot.task->n_tokens() - (int32_t) slot.prompt.n_tokens()) + kv_margin();
                r.has_slot  = true;
                in.reqs.push_back(r);
                held_running += r.held;
                continue;
            }
            if (slot.prompt.n_tokens() > 0) {
                sched_conv c;
                c.id     = slot.id;
                c.owned  = kv_cells_owned({ slot.id });
                c.t_last = slot.t_last_used;
                in.idle.push_back(c);
                idle_ids.push_back(slot.id);
            }
        }

        in.shared = std::max(0, kv_cells_used() - (idle_ids.empty() ? 0 : kv_cells_owned(idle_ids)) - held_running);
"""),
# finding 2: the two refusal checks
(
"""                    const int32_t need      = kv_need((int32_t) task.tokens.size() - (int32_t) ckpt->n_tokens, task.params.n_predict);
                    const int32_t committed = kv_committed(true);
                    const int32_t pool      = kv_pool_size();
""",
"""                    const int32_t need      = kv_need((int32_t) task.tokens.size() - (int32_t) ckpt->n_tokens, task.params.n_predict);
                    // review fix: not the suspended slots (kv_committed(true)): resumes belong to the decision, which
                    // admitted this task; a refusal here falls through to a cold start
                    const int32_t committed = kv_committed(false);
                    const int32_t pool      = kv_pool_size();
"""),
(
"""                    // what the active and suspended sequences will hold once their prompts are in. A restore that fits the
                    // cells free at this instant but not next to those is purged again by the first decode that runs out
                    // of room: seen in the field as a 94k state restored four seconds into another slot's 99k prefill,
                    // purged at 25k, and prefilled from zero on its next turn.
                    const int64_t committed = need > 0 ? (int64_t) kv_committed(true) : 0;
""",
"""                    // what the active sequences will hold once their prompts are in. A restore that fits the cells free
                    // at this instant but not next to those is purged again by the first decode that runs out of room:
                    // seen in the field as a 94k state restored four seconds into another slot's 99k prefill, purged
                    // at 25k, and prefilled from zero on its next turn. Review fix: not the suspended slots: resumes
                    // belong to the decision, which admitted this task, and a refusal here is a cold start
                    const int64_t committed = need > 0 ? (int64_t) kv_committed(false) : 0;
"""),
# finding 5: INFILL through the scheduler
(
"""        auto add_waiting = [&](const server_task & task) {
            if (task.type != SERVER_TASK_TYPE_COMPLETION) {
                return;
            }
""",
"""        auto add_waiting = [&](const server_task & task) {
            // review fix: an infill is a completion for the pool (priced as a cold start by plan_placement); the
            // other types cannot reach the queue on the shipped launchers
            if (task.type != SERVER_TASK_TYPE_COMPLETION && task.type != SERVER_TASK_TYPE_INFILL) {
                return;
            }
"""),
(
"""                    // customware (entry 013): under --kv-unified the scheduler decides whether this request starts now
                    if (params_base.kv_unified && task.type == SERVER_TASK_TYPE_COMPLETION) {
""",
"""                    // customware (entry 013): under --kv-unified the scheduler decides whether this request starts now
                    // (review fix: an infill too)
                    if (params_base.kv_unified && (task.type == SERVER_TASK_TYPE_COMPLETION || task.type == SERVER_TASK_TYPE_INFILL)) {
"""),
# finding 7: the misplaced comment
(
"""    // cells the given sequences own together (every label in the set): what removing all of them frees. In a
    // unified pool a fork shares its prefix with its source, so this is not the sum of token counts.
    // customware (entry 014): the sequences in the pool a tail can re-attach its prefix to: every slot with cells
""",
"""    // customware (entry 014): the sequences in the pool a tail can re-attach its prefix to: every slot with cells
"""),
(
"""    int32_t kv_cells_owned(const std::vector<llama_seq_id> & ids) const {
""",
"""    // cells the given sequences own together (every label in the set): what removing all of them frees. In a
    // unified pool a fork shares its prefix with its source, so this is not the sum of token counts.
    int32_t kv_cells_owned(const std::vector<llama_seq_id> & ids) const {
"""),
])

# ---------------------------------------------------------------- finding 6: no tails across media
patch('tools/server/server-task.cpp', [
(
"""    if (best < tail_min()) {
        *holder  = nullptr;
        *segment = nullptr;
        return 0;
    }

    if (*segment == nullptr) {
        *segment = find_segment(tokens, best);
    }

    return best;
}
""",
"""    if (best < tail_min()) {
        *holder  = nullptr;
        *segment = nullptr;
        return 0;
    }

    // review fix: the windows and the re-attach are in positions, and a media chunk before the split takes fewer
    // positions than tokens under M-RoPE: such a conversation is saved whole
    if (tokens.pos_next(best) != best) {
        *holder  = nullptr;
        *segment = nullptr;
        return 0;
    }

    if (*segment == nullptr) {
        *segment = find_segment(tokens, best);
    }

    return best;
}
"""),
])

# ---------------------------------------------------------------- finding 7: the suite header
patch('customware/test-regress.sh', [
(
"""#   entry 012 forking (test-fork.sh F1+F4), the small request next to a large one (test-third-request.sh).
""",
"""#   entries 012 and 014 forking and shared tails (test-fork.sh F1, F5 and F4: ONLY=154), the small request next
#   to a large one (test-third-request.sh).
"""),
])

print('patch-015a: done')
