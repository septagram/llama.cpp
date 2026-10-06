// customware (entry 013): the scheduler's decision function, on the shapes seen in the launcher logs

#include "../tools/server/server-sched.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static bool has(const std::vector<int> & v, int id) {
    return std::find(v.begin(), v.end(), id) != v.end();
}

static sched_req req(int id, sched_req_state st, int32_t held, int32_t need, int prio = 0, int64_t t = 0, int reuse = -1) {
    sched_req r;
    r.id = id; r.state = st; r.held = held; r.need = need; r.prio = prio; r.t_arrival = t; r.reuse = reuse;
    r.has_slot = st != SCHED_REQ_WAITING;
    return r;
}

static sched_conv conv(int id, int32_t owned, int64_t t_last) {
    sched_conv c;
    c.id = id; c.owned = owned; c.t_last = t_last;
    return c;
}

static sched_input base(int32_t pool, int n_slots) {
    sched_input in;
    in.pool = pool; in.n_slots = n_slots; in.now = 1000; in.t_age = 0;
    return in;
}

int main() {
    // 1. the user's day: two ~93k agents alternate in a 120k pool, an 8k request arrives while one generates and the
    //    other waits: the small one starts next to the active one, the large one keeps waiting (no preemption among
    //    equals)
    {
        auto in = base(120064, 2);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 93000, 256, 0, 10));
        in.reqs.push_back(req(2, SCHED_REQ_WAITING, 0, 93000 + 256, 0, 20));
        in.reqs.push_back(req(3, SCHED_REQ_WAITING, 0, 8000 + 256, 0, 30));
        auto out = sched_decide(in);
        CHECK(has(out.keep, 1));
        CHECK(!has(out.start, 2));
        CHECK(has(out.start, 3));
        CHECK(out.suspend.empty());
    }

    // 2. a fork pair: each holds only its tail; both fit although their token counts would not
    {
        auto in = base(16384, 2);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 12400 + 1500, 256, 0, 10)); // the parent owns the prefix and its tail
        in.reqs.push_back(req(2, SCHED_REQ_RUNNING, 1500, 256, 0, 20));         // the fork owns its tail
        auto out = sched_decide(in);
        CHECK(out.keep.size() == 2);
        CHECK(out.suspend.empty());
    }

    // 3. priority: a higher class arrives with no room; the lowest-ranked running request is displaced
    {
        auto in = base(120064, 3);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 60000, 256, 0, 10));
        in.reqs.push_back(req(2, SCHED_REQ_RUNNING, 50000, 256, 0, 20));
        in.reqs.push_back(req(3, SCHED_REQ_WAITING, 0, 30000 + 256, 5, 30));
        auto out = sched_decide(in);
        CHECK(has(out.start, 3));
        CHECK(has(out.keep, 1));
        CHECK(has(out.suspend, 2));
    }

    // 4. equal class: running is sticky, the waiter waits even when it arrived earlier than the running one started
    {
        auto in = base(120064, 2);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 100000, 256, 0, 50));
        in.reqs.push_back(req(2, SCHED_REQ_WAITING, 0, 50000, 0, 10));
        auto out = sched_decide(in);
        CHECK(has(out.keep, 1));
        CHECK(out.start.empty());
        CHECK(out.suspend.empty());
    }

    // 5. aging: after t_age the waiter is promoted one class and displaces an equal-class running request
    {
        auto in = base(120064, 2);
        in.t_age = 100;
        in.now = 1000;
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 100000, 256, 0, 950));
        in.reqs.push_back(req(2, SCHED_REQ_WAITING, 0, 50000, 0, 10)); // waited 990 > 100
        auto out = sched_decide(in);
        CHECK(has(out.start, 2));
        CHECK(has(out.suspend, 1));
    }

    // 6. purges: idle conversations are purged least recent first, only as far as needed, and the one a request
    //    continues is never purged
    {
        auto in = base(30000, 3);
        in.idle.push_back(conv(0, 12000, 100)); // oldest
        in.idle.push_back(conv(1, 10000, 200));
        in.idle.push_back(conv(2, 5000, 300));  // the one being continued
        in.reqs.push_back(req(1, SCHED_REQ_WAITING, 0, 20000, 0, 10, /*reuse*/ 2));
        auto out = sched_decide(in);
        CHECK(has(out.start, 1));
        // cost = 20000 + 5000 (taken over) = 25000; idle left: 12000 + 10000 = 22000 -> 47000 > 30000: purge 0 (-> 35000), then 1 (-> 25000)
        CHECK(out.purge.size() == 2);
        CHECK(out.purge[0] == 0 && out.purge[1] == 1);
        CHECK(!has(out.purge, 2));
    }

    // 7. a waiting request needs a slot: with two slots held by running requests and one idle conversation, the
    //    idle one is purged for its slot; with three running, the waiter waits
    {
        auto in = base(200000, 3);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 1000, 256, 0, 10));
        in.reqs.push_back(req(2, SCHED_REQ_RUNNING, 1000, 256, 0, 20));
        in.idle.push_back(conv(2, 1000, 5));
        in.reqs.push_back(req(3, SCHED_REQ_WAITING, 0, 1000, 0, 30));
        auto out = sched_decide(in);
        CHECK(has(out.start, 3));
        CHECK(has(out.purge, 2));

        auto in2 = base(200000, 3);
        in2.reqs.push_back(req(1, SCHED_REQ_RUNNING, 1000, 256, 0, 10));
        in2.reqs.push_back(req(2, SCHED_REQ_RUNNING, 1000, 256, 0, 20));
        in2.reqs.push_back(req(4, SCHED_REQ_RUNNING, 1000, 256, 0, 25));
        in2.reqs.push_back(req(3, SCHED_REQ_WAITING, 0, 1000, 0, 30));
        auto out2 = sched_decide(in2);
        CHECK(!has(out2.start, 3));
        CHECK(out2.suspend.empty());
    }

    // 8. a suspended request keeps its slot and comes back before a waiter of its class, when its cells are free
    {
        auto in = base(120064, 2);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 20000, 256, 0, 10));
        in.reqs.push_back(req(2, SCHED_REQ_SUSPENDED, 0, 93000 + 1024, 0, 20));
        in.reqs.push_back(req(3, SCHED_REQ_WAITING, 0, 8000, 0, 30));
        auto out = sched_decide(in);
        CHECK(has(out.start, 2));
        CHECK(out.start.size() == 1 && out.start[0] == 2); // the suspended one first
        CHECK(!has(out.start, 3));                          // no slot left: 1 running + 2 suspended-holding = 2
    }

    // 9. a suspended request whose cells are not free yet stays suspended; nothing is displaced for it
    {
        auto in = base(120064, 2);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 93000, 256, 0, 10));
        in.reqs.push_back(req(2, SCHED_REQ_SUSPENDED, 0, 93000 + 1024, 0, 20));
        auto out = sched_decide(in);
        CHECK(has(out.keep, 1));
        CHECK(out.start.empty());
        CHECK(out.suspend.empty());
    }

    // 10. the overflow victim: among the kept running requests, the lowest class, then the fewest held cells
    {
        auto in = base(120064, 3);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 50000, 256, 1, 10));
        in.reqs.push_back(req(2, SCHED_REQ_RUNNING, 30000, 256, 0, 20));
        in.reqs.push_back(req(3, SCHED_REQ_RUNNING, 10000, 256, 0, 30));
        auto out = sched_decide(in);
        CHECK(out.victim == 3);
    }

    // 11. the reserve of a running request keeps a lower-ranked waiter out even when its held cells alone would fit
    {
        auto in = base(10000, 2);
        in.reqs.push_back(req(1, SCHED_REQ_RUNNING, 9000, 900, 0, 10));
        in.reqs.push_back(req(2, SCHED_REQ_WAITING, 0, 500, 0, 20));
        auto out = sched_decide(in);
        CHECK(has(out.keep, 1));
        CHECK(!has(out.start, 2));
    }

    if (n_fail == 0) {
        printf("test-server-sched: all cases passed\n");
        return 0;
    }
    printf("test-server-sched: %d failures\n", n_fail);
    return 1;
}
