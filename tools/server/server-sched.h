#pragma once

// customware (entry 013): the scheduler of the unified KV pool as one pure decision.
//
// One ordered list of everything that wants to run (requests running, suspended or waiting), the idle
// conversations that hold reclaimable cells, and the two capacities (cells, slots). The decision walks the
// list in order and keeps what fits; what does not fit waits, a running request displaced by a higher class is
// suspended, and idle conversations are purged least-recently-used first when cells or slots are needed.
// Nothing is remembered between decisions except the list, so the server's separate rules (admission, the
// idle-slot clearing at launch, the overflow path, the resume pass) collapse into this function plus one
// reconcile step. No llama dependency: a unit test exercises it with the shapes seen in the logs.

#include <cstdint>
#include <string>
#include <vector>

enum sched_req_state {
    SCHED_REQ_RUNNING,   // in a slot, cells in the pool
    SCHED_REQ_SUSPENDED, // in a slot, state parked in RAM (entry 007), no cells
    SCHED_REQ_WAITING,   // deferred, no slot
};

struct sched_req {
    int     id        = -1;    // task id
    int     prio      = 0;     // explicit priority, higher first
    int64_t t_arrival = 0;     // when it arrived (us): order within a class, and the aging
    sched_req_state state = SCHED_REQ_WAITING;
    int32_t held      = 0;     // RUNNING: cells its sequence owns alone (what a suspension frees)
    int32_t need      = 0;     // cells it still needs beyond `held` and beyond what `reuse` provides
    int     reuse     = -1;    // idle conversation (slot id) it would continue
    int32_t reuse_keep = 0;    // cells of that conversation it keeps (the common prefix); the rest is freed by the takeover
    bool    has_slot  = false; // RUNNING and SUSPENDED hold a slot
};

struct sched_conv {
    int     id     = -1;       // slot id
    int32_t owned  = 0;        // cells purging it frees
    int64_t t_last = 0;        // last use (us); purged least recent first
};

struct sched_input {
    int32_t pool    = 0;       // cells
    int     n_slots = 0;
    int     n_active_max = 0;  // requests running at once (generating or prefilling); 0 = n_slots. Slots are contexts, not parallelism
    int64_t now     = 0;       // us
    int64_t t_age   = 0;       // a request waiting longer than this (us) is promoted one class; 0 = off
    std::vector<sched_req>  reqs;
    std::vector<sched_conv> idle;
};

struct sched_output {
    std::vector<int> keep;     // RUNNING ids that stay, in rank order
    std::vector<int> start;    // SUSPENDED (resume) and WAITING (launch) ids that fit, in rank order
    std::vector<int> suspend;  // RUNNING ids displaced by a higher class
    std::vector<int> purge;    // idle conversation ids to purge, least recent first, as many as needed
    int              victim = -1; // the RUNNING request to suspend if the pool overflows anyway (lowest rank; ties: fewest held, youngest)
    std::string      why;      // one line for the log
};

// the ordering key: effective priority (explicit, plus one class after t_age of waiting) descending, then
// running before suspended before waiting, then earlier arrival, then lower id
int  sched_eff_prio(const sched_req & r, const sched_input & in);
bool sched_before(const sched_req & a, const sched_req & b, const sched_input & in);

sched_output sched_decide(const sched_input & in);
