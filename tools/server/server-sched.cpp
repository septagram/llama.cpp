#include "server-sched.h"

#include <algorithm>
#include <cstdio>
#include <map>

int sched_eff_prio(const sched_req & r, const sched_input & in) {
    int prio = r.prio;
    if (r.state == SCHED_REQ_WAITING && in.t_age > 0 && in.now - r.t_arrival > in.t_age) {
        prio += 1;
    }
    return prio;
}

static int sched_state_rank(sched_req_state s) {
    switch (s) {
        case SCHED_REQ_RUNNING:   return 0;
        case SCHED_REQ_SUSPENDED: return 1;
        case SCHED_REQ_WAITING:   return 2;
    }
    return 3;
}

bool sched_before(const sched_req & a, const sched_req & b, const sched_input & in) {
    const int pa = sched_eff_prio(a, in);
    const int pb = sched_eff_prio(b, in);
    if (pa != pb) {
        return pa > pb;
    }
    const int ra = sched_state_rank(a.state);
    const int rb = sched_state_rank(b.state);
    if (ra != rb) {
        return ra < rb;
    }
    // among running requests of one class the one holding more cells ranks first: when one of them has to
    // leave, it is the cheapest to park and to bring back, as the overflow victim rule chooses
    if (a.state == SCHED_REQ_RUNNING && b.state == SCHED_REQ_RUNNING && a.held != b.held) {
        return a.held > b.held;
    }
    if (a.t_arrival != b.t_arrival) {
        return a.t_arrival < b.t_arrival;
    }
    return a.id < b.id;
}

sched_output sched_decide(const sched_input & in) {
    sched_output out;

    std::vector<const sched_req *> order;
    order.reserve(in.reqs.size());
    for (const auto & r : in.reqs) {
        order.push_back(&r);
    }
    std::stable_sort(order.begin(), order.end(), [&](const sched_req * a, const sched_req * b) {
        return sched_before(*a, *b, in);
    });

    // idle conversations: purgeable, least recent first
    std::vector<const sched_conv *> idle;
    idle.reserve(in.idle.size());
    for (const auto & c : in.idle) {
        idle.push_back(&c);
    }
    std::stable_sort(idle.begin(), idle.end(), [](const sched_conv * a, const sched_conv * b) {
        return a->t_last < b->t_last;
    });

    std::map<int, const sched_conv *> idle_by_id;
    for (const auto * c : idle) {
        idle_by_id[c->id] = c;
    }
    std::vector<int> taken_over; // idle conversations continued by a kept request: not purgeable

    // suspended requests hold their slot whether or not they come back now
    int slots_used = 0;
    for (const auto & r : in.reqs) {
        if (r.state == SCHED_REQ_SUSPENDED) {
            slots_used++;
        }
    }

    // how many may run at once: a third generation costs every slot a third of its speed for little in total
    const int active_max = in.n_active_max > 0 ? std::min(in.n_active_max, in.n_slots) : in.n_slots;
    int active = 0;

    // the walk: idle cells count as free (they can be purged); the purge list is settled afterwards
    int32_t cells_used = 0;
    const sched_req * victim = nullptr;

    for (const auto * r : order) {
        const bool continues = r->reuse >= 0 && idle_by_id.count(r->reuse) > 0 &&
            std::find(taken_over.begin(), taken_over.end(), r->reuse) == taken_over.end();

        // what the request costs the pool: its own cells, what it still needs, and what it keeps of the
        // conversation it continues (the rest of that conversation is freed by the takeover)
        const int32_t cost = r->held + r->need + (continues ? std::min(r->reuse_keep, idle_by_id[r->reuse]->owned) : 0);

        bool fits_cells = false;
        bool fits_slot  = false;

        switch (r->state) {
            case SCHED_REQ_RUNNING:
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
        }

        if (fits_cells && fits_slot) {
            cells_used += cost;
            active++;
            switch (r->state) {
                case SCHED_REQ_RUNNING:
                    out.keep.push_back(r->id);
                    slots_used++;
                    if (victim == nullptr ||
                        sched_eff_prio(*r, in) <  sched_eff_prio(*victim, in) ||
                        (sched_eff_prio(*r, in) == sched_eff_prio(*victim, in) &&
                            (r->held < victim->held || (r->held == victim->held && r->t_arrival > victim->t_arrival)))) {
                        victim = r;
                    }
                    break;
                case SCHED_REQ_SUSPENDED:
                    out.start.push_back(r->id);
                    break;
                case SCHED_REQ_WAITING:
                    out.start.push_back(r->id);
                    slots_used++;
                    break;
            }
            if (continues) {
                taken_over.push_back(r->reuse);
            }
        } else if (r->state == SCHED_REQ_RUNNING) {
            // displaced: a higher class took the cells it holds, or its place among the active ones
            out.suspend.push_back(r->id);
            slots_used++; // it keeps its slot while suspended
        }
        // suspended and waiting requests that do not fit simply wait
    }

    out.victim = victim ? victim->id : -1;

    // purges: least recent idle conversations until the kept set fits in cells and in slots
    int32_t idle_cells = 0;
    int     idle_slots = 0;
    for (const auto * c : idle) {
        if (std::find(taken_over.begin(), taken_over.end(), c->id) != taken_over.end()) {
            continue;
        }
        idle_cells += c->owned;
        idle_slots += 1;
    }
    for (const auto * c : idle) {
        if (std::find(taken_over.begin(), taken_over.end(), c->id) != taken_over.end()) {
            continue;
        }
        const bool cells_ok = cells_used + idle_cells <= in.pool;
        const bool slots_ok = slots_used + idle_slots <= in.n_slots;
        if (cells_ok && slots_ok) {
            break;
        }
        out.purge.push_back(c->id);
        idle_cells -= c->owned;
        idle_slots -= 1;
    }

    char buf[256];
    snprintf(buf, sizeof(buf), "keep %zu, start %zu, suspend %zu, purge %zu; %d of %d cells committed, %d of %d slots, %d of %d active",
             out.keep.size(), out.start.size(), out.suspend.size(), out.purge.size(),
             cells_used, in.pool, slots_used, in.n_slots, active, active_max);
    out.why = buf;

    return out;
}
