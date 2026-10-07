#!/usr/bin/env python
"""Review iteration 2 fixes (review 26d-blade-engulf-upload): the prefix shared only between idle conversations is
charged to a takeover that keeps it (sched_input::idle_shared), and three comments stop describing dropped policy.
Exact-string edits; every anchor must match exactly once."""
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

patch('tools/server/server-sched.h', [
(
"""    int32_t shared  = 0;
    std::vector<sched_req>  reqs;
""",
"""    int32_t shared  = 0;
    // cells labelled only by idle conversations and by more than one of them (a finished fork next to its source):
    // purgeable, so nobody's `owned`, but a takeover that keeps such a prefix keeps these too, so they are charged
    // to it. With three or more idle conversations this lumps all idle-only sharing together and can over-charge a
    // takeover; that errs toward waiting, and the `min` with the kept prefix and the pool cap bound it
    int32_t idle_shared = 0;
    std::vector<sched_req>  reqs;
"""),
])

patch('tools/server/server-sched.cpp', [
(
"""        const int32_t cost = std::min<int32_t>(r->held + r->need + (continues ? std::min(r->reuse_keep, idle_by_id[r->reuse]->owned) : 0), in.pool);
""",
"""        // the kept part of a takeover: the conversation's own cells plus what it shares only with other idle ones
        // (the purge loop keeps using `owned`: purging the sibling still frees only its tail)
        const int32_t cost = std::min<int32_t>(r->held + r->need + (continues ? std::min(r->reuse_keep, idle_by_id[r->reuse]->owned + in.idle_shared) : 0), in.pool);
"""),
])

patch('tests/test-server-sched.cpp', [
(
"""    // 2c. the whole cost is capped at the pool: a conversation of 118000 idle in a 120064 pool, its next turn
""",
"""    // 2d. two idle siblings share a 70000-cell prefix (nobody's `owned`, purgeable); a turn continuing one of them
    //     keeps that prefix and is charged it: with 45000 running it does not start (45256 + 8048 + 71000 > 120064),
    //     and nothing is suspended for it (running is sticky among equals)
    {
        auto in = base(120064, 3);
        in.idle_shared = 70000;
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 45000, 256, 0, 10));
        in.idle.push_back(conv(10, 1000, 100));
        in.idle.push_back(conv(11, 1000, 200));
        in.reqs.push_back(req(2, SCHED_REQ_WAITING, 0, 8048, 0, 20, /*reuse*/ 10, /*keeps*/ 71000));
        auto out = sched_decide(in);
        CHECK(has(out.keep, 1));
        CHECK(!has(out.start, 2));
        CHECK(out.suspend.empty());
    }

    // 2c. the whole cost is capped at the pool: a conversation of 118000 idle in a 120064 pool, its next turn
"""),
])

patch('tools/server/server-context.cpp', [
(
"""        in.shared = std::max(0, kv_cells_used() - (idle_ids.empty() ? 0 : kv_cells_owned(idle_ids)) - held_running);
""",
"""        // what purging every idle conversation frees, minus what purging each alone frees: the cells shared only
        // between idle conversations. A takeover that keeps such a prefix is charged it (review iteration 2)
        const int32_t idle_owned_all = idle_ids.empty() ? 0 : kv_cells_owned(idle_ids);
        int32_t idle_owned_each = 0;
        for (const auto & c : in.idle) {
            idle_owned_each += c.owned;
        }
        in.shared      = std::max(0, kv_cells_used() - idle_owned_all - held_running);
        in.idle_shared = std::max(0, idle_owned_all - idle_owned_each);
"""),
(
"""    // cells spoken for: active slots (their prompt, or the full prompt still being processed, plus the growth
    // margin) and, optionally, suspended slots (they come back before anything new is admitted). Idle retained
    // slots are purgeable and not counted.
    int32_t kv_resident(bool include_suspended) const {
""",
"""    // cells spoken for: active slots (their prompt, or the full prompt still being processed, plus the growth
    // margin) and, with include_suspended, the suspended slots. Nobody passes true any more: the decision owns the
    // suspended slots (it resumes them when they fit, possibly behind a higher-ranked newcomer), and the two fit
    // checks in get_available_slot measure what the decision measures. Idle retained slots are purgeable and not
    // counted.
    int32_t kv_resident(bool include_suspended) const {
"""),
(
"""    // what the pool has to hold: the cells in use now, minus the purgeable idle ones, plus what the active sequences
    // will still take (the rest of a prompt being processed and a growth margin each) and, optionally, the suspended
    // ones (they come back before anything new is admitted). Falls back to the bookkeeping when the cache does not
    // report its occupancy.
    int32_t kv_committed(bool include_suspended, const server_slot * exclude_idle = nullptr) const {
""",
"""    // what the pool has to hold: the cells in use now, minus the purgeable idle ones, plus what the active sequences
    // will still take (the rest of a prompt being processed and a growth margin each) and, with include_suspended,
    // the suspended ones. Nobody passes true any more: the decision owns the suspended slots and may start a
    // higher-ranked newcomer ahead of them, so a fit check that counted them would refuse what the decision
    // admitted. Falls back to the bookkeeping when the cache does not report its occupancy.
    int32_t kv_committed(bool include_suspended, const server_slot * exclude_idle = nullptr) const {
"""),
(
"""                // customware: under a unified pool the restore needs free cells, and the other idle slots keep
                // theirs until the launch below clears them. Free them here first (saved, as the launch would),
                // and do not ask for a restore that cannot fit: the entry stays for the retry after admission
                // defers the task. Before this, a continuation of a cached 99k prompt behind a 70k prefill was
                // re-prefilled from zero with its state sitting in RAM.
                bool can_load = true;
""",
"""                // customware: under a unified pool the restore needs free cells, and the other idle slots keep
                // theirs until a decision purges them, or until the loop below frees them for this restore (saved,
                // as a purge would). A restore that cannot fit even then is not asked for and the task falls through
                // to a cold start; the decision that admitted it charges shared cells and the running requests'
                // reserves, so that fall-through should not be reached. Before this, a continuation of a cached 99k
                // prompt behind a 70k prefill was re-prefilled from zero with its state sitting in RAM.
                bool can_load = true;
"""),
])

print('patch-015b: done')
