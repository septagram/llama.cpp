#!/usr/bin/env python
"""Entry 013 follow-up: (1) a cap on the requests generating at once (slots are contexts, not parallelism);
(2) a request continuing an idle conversation keeps only the common prefix of it, the rest is freed by the
takeover, so the walk charges the kept cells, not the conversation's whole size."""
bs = chr(92)


def load(p):
    s = open(p, encoding='utf-8', newline='').read()
    nl = chr(13) + chr(10) if (chr(13) + chr(10)) in s else chr(10)
    return s, nl


def save(p, s):
    open(p, 'w', encoding='utf-8', newline='').write(s)


def rep(s, nl, old, new, count=1):
    old = old.replace('\n', nl).replace('%BS%', bs)
    new = new.replace('\n', nl).replace('%BS%', bs)
    assert s.count(old) == count, (s.count(old), old[:100])
    return s.replace(old, new)


# ---------------------------------------------------------------- server-sched.h
p = 'tools/server/server-sched.h'
s, nl = load(p)
assert 'n_active_max' not in s
s = rep(s, nl, '''    int     reuse     = -1;    // idle conversation (slot id) it would continue; those cells become its own
''', '''    int     reuse     = -1;    // idle conversation (slot id) it would continue
    int32_t reuse_keep = 0;    // cells of that conversation it keeps (the common prefix); the rest is freed by the takeover
''')
s = rep(s, nl, '''    int     n_slots = 0;
''', '''    int     n_slots = 0;
    int     n_active_max = 0;  // requests running at once (generating or prefilling); 0 = n_slots. Slots are contexts, not parallelism
''')
save(p, s)

# ---------------------------------------------------------------- server-sched.cpp
p = 'tools/server/server-sched.cpp'
s, nl = load(p)
s = rep(s, nl, '''    // suspended requests hold their slot whether or not they come back now
    int slots_used = 0;
    for (const auto & r : in.reqs) {
        if (r.state == SCHED_REQ_SUSPENDED) {
            slots_used++;
        }
    }
''', '''    // suspended requests hold their slot whether or not they come back now
    int slots_used = 0;
    for (const auto & r : in.reqs) {
        if (r.state == SCHED_REQ_SUSPENDED) {
            slots_used++;
        }
    }

    // how many may run at once: a third generation costs every slot a third of its speed for little in total
    const int active_max = in.n_active_max > 0 ? std::min(in.n_active_max, in.n_slots) : in.n_slots;
    int active = 0;
''')
s = rep(s, nl, '''        // what the request costs the pool: its own cells, what it still needs, and the conversation it continues
        const int32_t cost = r->held + r->need + (continues ? idle_by_id[r->reuse]->owned : 0);
''', '''        // what the request costs the pool: its own cells, what it still needs, and what it keeps of the
        // conversation it continues (the rest of that conversation is freed by the takeover)
        const int32_t cost = r->held + r->need + (continues ? std::min(r->reuse_keep, idle_by_id[r->reuse]->owned) : 0);
''')
s = rep(s, nl, '''            case SCHED_REQ_RUNNING:
                // it stays unless the cells it actually holds do not fit next to the higher-ranked ones; its
                // growth reserve is charged either way, so nothing lower-ranked starts into it
                fits_cells = cells_used + r->held <= in.pool;
                fits_slot  = true;
                break;
            case SCHED_REQ_SUSPENDED:
                fits_cells = cells_used + cost <= in.pool;
                fits_slot  = true; // it kept its slot
                break;
            case SCHED_REQ_WAITING:
                fits_cells = cells_used + cost <= in.pool;
                // an idle conversation's slot can be purged or taken over, so only slots of requests count
                fits_slot  = slots_used + 1 <= in.n_slots;
                break;
''', '''            case SCHED_REQ_RUNNING:
                // it stays unless the cells it actually holds do not fit next to the higher-ranked ones, or a
                // higher-ranked request took its place among the active ones; its growth reserve is charged
                // either way, so nothing lower-ranked starts into it
                fits_cells = cells_used + r->held <= in.pool;
                fits_slot  = active + 1 <= active_max;
                break;
            case SCHED_REQ_SUSPENDED:
                fits_cells = cells_used + cost <= in.pool;
                fits_slot  = active + 1 <= active_max; // it kept its slot
                break;
            case SCHED_REQ_WAITING:
                fits_cells = cells_used + cost <= in.pool;
                // an idle conversation's slot can be purged or taken over, so only slots of requests count
                fits_slot  = slots_used + 1 <= in.n_slots && active + 1 <= active_max;
                break;
''')
s = rep(s, nl, '''        if (fits_cells && fits_slot) {
            cells_used += cost;
            switch (r->state) {
                case SCHED_REQ_RUNNING:
                    out.keep.push_back(r->id);
                    slots_used++;
''', '''        if (fits_cells && fits_slot) {
            cells_used += cost;
            active++;
            switch (r->state) {
                case SCHED_REQ_RUNNING:
                    out.keep.push_back(r->id);
                    slots_used++;
''')
s = rep(s, nl, '''        } else if (r->state == SCHED_REQ_RUNNING) {
            // displaced: a higher class took the cells it holds
            out.suspend.push_back(r->id);
            slots_used++; // it keeps its slot while suspended
        }
''', '''        } else if (r->state == SCHED_REQ_RUNNING) {
            // displaced: a higher class took the cells it holds, or its place among the active ones
            out.suspend.push_back(r->id);
            slots_used++; // it keeps its slot while suspended
        }
''')
s = rep(s, nl, '''    snprintf(buf, sizeof(buf), "keep %zu, start %zu, suspend %zu, purge %zu; %d of %d cells committed, %d of %d slots",
             out.keep.size(), out.start.size(), out.suspend.size(), out.purge.size(),
             cells_used, in.pool, slots_used, in.n_slots);
''', '''    snprintf(buf, sizeof(buf), "keep %zu, start %zu, suspend %zu, purge %zu; %d of %d cells committed, %d of %d slots, %d of %d active",
             out.keep.size(), out.start.size(), out.suspend.size(), out.purge.size(),
             cells_used, in.pool, slots_used, in.n_slots, active, active_max);
''')
save(p, s)

# ---------------------------------------------------------------- the unit test
p = 'tests/test-server-sched.cpp'
s, nl = load(p)
s = rep(s, nl, '''static sched_req req(int id, sched_req_state st, int32_t held, int32_t need, int prio = 0, int64_t t = 0, int reuse = -1) {
    sched_req r;
    r.id = id; r.state = st; r.held = held; r.need = need; r.prio = prio; r.t_arrival = t; r.reuse = reuse;
''', '''static sched_req req(int id, sched_req_state st, int32_t held, int32_t need, int prio = 0, int64_t t = 0, int reuse = -1, int32_t reuse_keep = 0) {
    sched_req r;
    r.id = id; r.state = st; r.held = held; r.need = need; r.prio = prio; r.t_arrival = t; r.reuse = reuse; r.reuse_keep = reuse_keep;
''')
s = rep(s, nl, '''        in.reqs.push_back(req(1, SCHED_REQ_WAITING, 0, 20000, 0, 10, /*reuse*/ 2));
        auto out = sched_decide(in);
        CHECK(has(out.start, 1));
        // cost = 20000 + 5000 (taken over) = 25000; idle left: 12000 + 10000 = 22000 -> 47000 > 30000: purge 0 (-> 35000), then 1 (-> 25000)
        CHECK(out.purge.size() == 2);
''', '''        in.reqs.push_back(req(1, SCHED_REQ_WAITING, 0, 20000, 0, 10, /*reuse*/ 2, /*keeps*/ 5000));
        auto out = sched_decide(in);
        CHECK(has(out.start, 1));
        // cost = 20000 + 5000 (kept of the taken-over one) = 25000; idle left: 12000 + 10000 = 22000 -> 47000 > 30000: purge 0 (-> 35000), then 1 (-> 25000)
        CHECK(out.purge.size() == 2);
''')
s = rep(s, nl, '''    if (n_fail == 0) {
''', '''    // 12. a takeover keeps only the common prefix: a 42-token request continuing a 13k conversation by a 10-token
    //     prefix costs 10 + its need, not 13k, so the other idle conversation is not purged
    {
        auto in = base(24576, 3);
        in.idle.push_back(conv(0, 13000, 100));
        in.idle.push_back(conv(2, 12940, 200));
        in.reqs.push_back(req(1, SCHED_REQ_WAITING, 0, 42 - 10 + 16, 0, 10, /*reuse*/ 2, /*keeps*/ 10));
        auto out = sched_decide(in);
        CHECK(has(out.start, 1));
        CHECK(out.purge.empty());
    }

    // 13. at most n_active_max run at once: three slots, two generating, a third waiter waits although cells and a
    //     slot are free; a higher class still gets in by displacing the lowest-ranked running one
    {
        auto in = base(200000, 3);
        in.n_active_max = 2;
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 1000, 256, 0, 10));
        in.reqs.push_back(req(2, SCHED_REQ_RUNNING, 1000, 256, 0, 20));
        in.reqs.push_back(req(3, SCHED_REQ_WAITING, 0, 1000, 0, 30));
        auto out = sched_decide(in);
        CHECK(out.keep.size() == 2);
        CHECK(!has(out.start, 3));
        CHECK(out.suspend.empty());

        in.reqs.push_back(req(4, SCHED_REQ_WAITING, 0, 1000, 3, 40));
        auto out2 = sched_decide(in);
        CHECK(has(out2.start, 4));
        CHECK(has(out2.suspend, 2));
        CHECK(has(out2.keep, 1));
        CHECK(!has(out2.start, 3));
    }

    if (n_fail == 0) {
''')
save(p, s)

# ---------------------------------------------------------------- server-context.cpp: the cap and the kept cells
p = 'tools/server/server-context.cpp'
s, nl = load(p)
s = rep(s, nl, '''    static int64_t sched_age_us() {
        static const int64_t age = (getenv("LLAMA_SCHED_AGE_S") ? atoll(getenv("LLAMA_SCHED_AGE_S")) : 300) * 1000000;
        return age;
    }

    struct sched_placement {
        int32_t need  = 0;  // cells the request needs that are not in the pool yet
        int     reuse = -1; // the idle slot whose conversation it would continue
    };
''', '''    static int64_t sched_age_us() {
        static const int64_t age = (getenv("LLAMA_SCHED_AGE_S") ? atoll(getenv("LLAMA_SCHED_AGE_S")) : 300) * 1000000;
        return age;
    }

    // requests running at once (LLAMA_SCHED_MAX_ACTIVE, default 2): a third generation costs every slot a third of
    // its speed for little in total, so the other slots serve priority starts, suspended requests and idle
    // conversations that keep their cells
    static int sched_active_max() {
        static const int n = getenv("LLAMA_SCHED_MAX_ACTIVE") ? atoi(getenv("LLAMA_SCHED_MAX_ACTIVE")) : 2;
        return n;
    }

    struct sched_placement {
        int32_t need  = 0;  // cells the request needs that are not in the pool yet
        int     reuse = -1; // the idle slot whose conversation it would continue
        int32_t keep  = 0;  // cells of that conversation it keeps (the common prefix)
    };
''')
s = rep(s, nl, '''        if (idle != nullptr && lcp_idle >= lcp_busy) {
            pl.need  = kv_need(n - (int32_t) lcp_idle, task.params.n_predict);
            pl.reuse = idle->id;
            return pl;
        }
''', '''        if (idle != nullptr && lcp_idle >= lcp_busy) {
            pl.need  = kv_need(n - (int32_t) lcp_idle, task.params.n_predict);
            pl.reuse = idle->id;
            pl.keep  = (int32_t) lcp_idle;
            return pl;
        }
''')
s = rep(s, nl, '''        in.pool    = kv_pool_size();
        in.n_slots = (int) slots.size();
        in.now     = ggml_time_us();
        in.t_age   = sched_age_us();
''', '''        in.pool         = kv_pool_size();
        in.n_slots      = (int) slots.size();
        in.n_active_max = sched_active_max();
        in.now          = ggml_time_us();
        in.t_age        = sched_age_us();
''')
s = rep(s, nl, '''            r.need      = pl.need;
            r.reuse     = pl.reuse;
            in.reqs.push_back(r);
''', '''            r.need       = pl.need;
            r.reuse      = pl.reuse;
            r.reuse_keep = pl.keep;
            in.reqs.push_back(r);
''')
save(p, s)
print('cap and kept-cells edits applied')
