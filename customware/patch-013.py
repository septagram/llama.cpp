#!/usr/bin/env python
"""Entry 013 server-level patch (kept for the record; the committed source is the result).
Run from the repo root once. %BS% stands for a backslash in the C source."""
import re

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


# ---------------------------------------------------------------- server-task.h: priority, t_arrival
p = 'tools/server/server-task.h'
s, nl = load(p)
assert 'priority' not in s, 'already patched'
s = rep(s, nl, '''    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables
''', '''    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t priority = 0; // customware (entry 013): scheduling class, higher first; a higher class may suspend a running lower one
''')
s = rep(s, nl, '''    int id_slot   = -1;
''', '''    int id_slot   = -1;

    int64_t t_arrival = 0; // customware (entry 013): when the scheduler first saw it (us); order within a class, aging
''')
save(p, s)

# ---------------------------------------------------------------- server-schema.cpp: the request field
p = 'tools/server/server-schema.cpp'
s, nl = load(p)
s = rep(s, nl, '''    add((new field_num("sse_ping_interval", params.sse_ping_interval))
''', '''    add((new field_num("priority", params.priority))
        ->set_desc("Scheduling priority (default 0, higher first). A request of a higher class may suspend a running request of a lower class when it needs its cells; among equals, running requests are never interrupted"));

    add((new field_num("sse_ping_interval", params.sse_ping_interval))
''')
save(p, s)

# ---------------------------------------------------------------- server-queue: peek and pull by id
p = 'tools/server/server-queue.h'
s, nl = load(p)
s = rep(s, nl, '''    void pop_deferred_task(int id_slot);
''', '''    void pop_deferred_task(int id_slot);

    // customware (entry 013): the scheduler reads the deferred tasks and pulls the ones it starts, in its order
    template <class F>
    void for_each_deferred(F && f) {
        std::unique_lock<std::mutex> lock(mutex_tasks);
        for (const auto & task : queue_tasks_deferred) {
            f(task);
        }
    }

    void pop_deferred_tasks(const std::vector<int> & ids);
''')
save(p, s)

p = 'tools/server/server-queue.cpp'
s, nl = load(p)
s = rep(s, nl, '''int server_queue::get_new_id() {
''', '''void server_queue::pop_deferred_tasks(const std::vector<int> & ids) {
    std::unique_lock<std::mutex> lock(mutex_tasks);

    // the named tasks go to the front of the queue in the given order; the others stay deferred in their order
    std::vector<server_task> picked;
    for (int id : ids) {
        for (auto it = queue_tasks_deferred.begin(); it != queue_tasks_deferred.end(); ++it) {
            if (it->id == id) {
                QUE_DBG("pop deferred task (scheduler), id_task = %d%BS%n", id);
                picked.push_back(std::move(*it));
                queue_tasks_deferred.erase(it);
                break;
            }
        }
    }
    for (auto it = picked.rbegin(); it != picked.rend(); ++it) {
        queue_tasks.emplace_front(std::move(*it));
    }

    time_last_task = ggml_time_ms();
    condition_tasks.notify_one();
}

int server_queue::get_new_id() {
''')
save(p, s)

# ---------------------------------------------------------------- server-context.cpp
p = 'tools/server/server-context.cpp'
s, nl = load(p)
assert 'sched_decide' not in s, 'already patched'

# include
m = re.search(r'#include "server-[a-z-]+\.h"' + re.escape(nl), s)
s = s[:m.end()] + '#include "server-sched.h"' + nl + s[m.end():]

# members
s = rep(s, nl, '''    int64_t n_forks        = 0; // customware (entry 012): requests that started as a fork of a busy slot's conversation
''', '''    int64_t n_forks        = 0; // customware (entry 012): requests that started as a fork of a busy slot's conversation

    // customware (entry 013): the scheduler runs on events and, in update_slots, when something changed or every 2 s
    bool    sched_dirty       = true;
    int64_t t_sched_last      = 0;
    int64_t n_sched_decisions = 0;
    int32_t sched_purged_last = 0;
''')

# resume: back to the prompt loop when the prompt was not fully processed
s = rep(s, nl, '''        suspended_state = nullptr;
        resume_failures = 0;
        state           = SLOT_STATE_GENERATING;
''', '''        suspended_state = nullptr;
        resume_failures = 0;

        // customware (entry 013): a request suspended during its prompt returns to the prompt loop, which finds the
        // processed prefix in place and continues; a generating one picks up with its pending sampled token
        state = task && prompt.n_tokens() < task->n_tokens() ? SLOT_STATE_STARTED : SLOT_STATE_GENERATING;
''')

# /slots: priority
s = rep(s, nl, '''            res["id_task"] = ptask->id;
''', '''            res["id_task"] = ptask->id;
            res["priority"] = ptask->params.priority;
''')

# callback_on_release: the scheduler, not the whole-queue re-post
s = rep(s, nl, '''            slot.callback_on_release = [this](int id_slot) {
                queue_tasks.pop_deferred_task(id_slot);
            };
''', '''            slot.callback_on_release = [this](int id_slot) {
                if (params_base.kv_unified) {
                    // customware (entry 013): a decision pulls what can start now; it may run from inside the slot
                    // loop, so it moves no state (LIGHT), the next full pass does
                    sched_dirty = true;
                    schedule("release", nullptr, SCHED_LIGHT);
                } else {
                    queue_tasks.pop_deferred_task(id_slot);
                }
            };
''')

# try_preempt: a named victim, prompt-processing slots allowed
s = rep(s, nl, '''    bool try_preempt(int32_t off) {
        if (!prompt_cache) {
            SRV_WRN("%s", "the KV pool overflowed and no prompt cache is available to park a sequence (--cache-ram 0)%BS%n");
            return false;
        }

        server_slot * victim = pick_victim();
        if (victim == nullptr) {
            return false;
        }
''', '''    bool try_preempt(int32_t off, int id_task_victim = -1) {
        if (!prompt_cache) {
            SRV_WRN("%s", "the KV pool overflowed and no prompt cache is available to park a sequence (--cache-ram 0)%BS%n");
            return false;
        }

        // customware (entry 013): the scheduler names the victim (its lowest-ranked running request, which may be
        // processing its prompt); without one, the entry 007 choice
        server_slot * victim = nullptr;
        if (id_task_victim >= 0) {
            for (auto & slot : slots) {
                if (slot.is_processing() && slot.state != SLOT_STATE_SUSPENDED && slot.state != SLOT_STATE_WAIT_OTHER &&
                        slot.task && slot.task->id == id_task_victim && !slot.task->is_child() && !slot.task->is_parent()) {
                    victim = &slot;
                }
            }
        }
        if (victim == nullptr) {
            victim = pick_victim();
        }
        if (victim == nullptr) {
            return false;
        }
        victim->spec_i_batch.clear();
''')

# try_clear_idle_slots -> purge_idle + the most-owned wrapper
s = rep(s, nl, '''        if (victim != nullptr) {
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
    }
''', '''        if (victim != nullptr) {
            purge_idle(*victim, victim_owned);
            res = true;
        }

        return res;
    }

    // an idle conversation leaves the pool: saved to the prompt cache first (a purge is not a cache miss), then cleared
    void purge_idle(server_slot & slot, int32_t owned) {
        SRV_WRN("purging slot %d with %zu tokens (%d cells of its own)%BS%n", slot.id, slot.prompt.tokens.size(), owned);

        if (prompt_cache && params_base.cache_idle_slots) {
            slot.prompt_save(*prompt_cache);
            prompt_cache->update();
        }

        slot.prompt_clear();
        sched_dirty = true;
    }
''')

# remove try_resume_suspended (the scheduler resumes)
i = s.index('    // bring suspended sequences back, oldest first, as soon as their cells are free; idle retained slots give way')
j = s.index('    // cells in use right now, from the cache itself')
s = s[:i] + s[j:]

# the scheduler itself, before the turn-end checkpoint method
s = rep(s, nl, '''    // customware (entry 012): the state at the end of a generation, taken when the generation ends. Before, it was
''', '''    //
    // customware (entry 013): one scheduler instead of five rules (customware/notes-scheduler.md, 013-scheduler.md)
    //

    enum sched_mode {
        SCHED_FULL,      // decide and reconcile: suspend, purge, resume, pull the waiting starts
        SCHED_LIGHT,     // decide and pull the waiting starts only (from inside the slot loop)
        SCHED_EMERGENCY, // a decode found no cells: purge what the decision purges; the victim goes through try_preempt
    };

    static int64_t sched_age_us() {
        static const int64_t age = (getenv("LLAMA_SCHED_AGE_S") ? atoll(getenv("LLAMA_SCHED_AGE_S")) : 300) * 1000000;
        return age;
    }

    struct sched_placement {
        int32_t need  = 0;  // cells the request needs that are not in the pool yet
        int     reuse = -1; // the idle slot whose conversation it would continue
    };

    // what starting the task would cost, without starting it: the same choice get_available_slot makes
    sched_placement plan_placement(const server_task & task) const {
        sched_placement pl;
        const int32_t n = task.n_tokens();

        if (!task.params.cache_prompt || task.type != SERVER_TASK_TYPE_COMPLETION || task.tokens.size() == 0) {
            pl.need = kv_need(n, task.params.n_predict);
            return pl;
        }

        // the idle slot with the best prefix, and the busy one
        const server_slot * idle = nullptr; size_t lcp_idle = 0;
        const server_slot * busy = nullptr; size_t lcp_busy = 0;
        for (const auto & slot : slots) {
            if (slot.prompt.tokens.size() == 0 || slot.state == SLOT_STATE_SUSPENDED || slot.state == SLOT_STATE_WAIT_OTHER) {
                continue;
            }
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }
            const size_t lcp = slot.prompt.tokens.get_common_prefix(task.tokens);
            if (float(lcp) / task.tokens.size() <= slot_prompt_similarity) {
                continue;
            }
            if (slot.is_processing()) {
                if (lcp > lcp_busy) { lcp_busy = lcp; busy = &slot; }
            } else {
                if (lcp > lcp_idle) { lcp_idle = lcp; idle = &slot; }
            }
        }

        if (idle != nullptr && lcp_idle >= lcp_busy) {
            pl.need  = kv_need(n - (int32_t) lcp_idle, task.params.n_predict);
            pl.reuse = idle->id;
            return pl;
        }

        if (busy != nullptr && task.id_slot == -1) {
            int64_t c = 0;
            for (const auto & cur : busy->prompt.checkpoints) {
                if (cur.n_tokens <= (int64_t) lcp_busy && cur.n_tokens < (int64_t) task.tokens.size() && cur.n_tokens > c) {
                    c = cur.n_tokens;
                }
            }
            if (c > 0 && float(c) / task.tokens.size() > slot_prompt_similarity) {
                pl.need = kv_need(n - (int32_t) c, task.params.n_predict);
                return pl;
            }
        }

        if (prompt_cache) {
            const int64_t entry = prompt_cache->peek(server_prompt{}, task.tokens);
            if (entry > 0) {
                // the entry is restored whole, then trimmed and extended
                pl.need = kv_need(std::max((int32_t) entry, n), task.params.n_predict);
                return pl;
            }
        }

        pl.need = kv_need(n, task.params.n_predict);
        return pl;
    }

    sched_input build_sched_input(const server_task * new_task) const {
        sched_input in;
        in.pool    = kv_pool_size();
        in.n_slots = (int) slots.size();
        in.now     = ggml_time_us();
        in.t_age   = sched_age_us();

        for (const auto & slot : slots) {
            if (slot.state == SLOT_STATE_SUSPENDED) {
                sched_req r;
                r.id        = slot.task->id;
                r.prio      = slot.task->params.priority;
                r.t_arrival = slot.task->t_arrival;
                r.state     = SCHED_REQ_SUSPENDED;
                r.need      = slot.prompt.n_tokens() + kv_resume_margin();
                r.has_slot  = true;
                in.reqs.push_back(r);
                continue;
            }
            if (slot.is_processing()) {
                if (!slot.task) {
                    continue;
                }
                sched_req r;
                r.id        = slot.task->id;
                r.prio      = slot.task->params.priority;
                r.t_arrival = slot.task->t_arrival;
                r.state     = SCHED_REQ_RUNNING;
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

        auto add_waiting = [&](const server_task & task) {
            if (task.type != SERVER_TASK_TYPE_COMPLETION) {
                return;
            }
            const sched_placement pl = plan_placement(task);
            sched_req r;
            r.id        = task.id;
            r.prio      = task.params.priority;
            r.t_arrival = task.t_arrival;
            r.state     = SCHED_REQ_WAITING;
            r.need      = pl.need;
            r.reuse     = pl.reuse;
            in.reqs.push_back(r);
        };

        const_cast<server_queue &>(queue_tasks).for_each_deferred([&](const server_task & task) { add_waiting(task); });
        if (new_task) {
            add_waiting(*new_task);
        }

        return in;
    }

    // one decision and its reconcile; returns it so the caller can read start/victim
    sched_output schedule(const char * reason, const server_task * new_task, sched_mode mode) {
        const sched_input in = build_sched_input(new_task);
        sched_output out = sched_decide(in);

        n_sched_decisions++;
        t_sched_last      = in.now;
        sched_dirty       = false;
        sched_purged_last = 0;

        if (!out.start.empty() || !out.suspend.empty() || !out.purge.empty()) {
            SRV_INF("scheduler (%s): %s%BS%n", reason, out.why.c_str());
        }

        auto slot_of_task = [&](int id_task) -> server_slot * {
            for (auto & slot : slots) {
                if (slot.is_processing() && slot.task && slot.task->id == id_task) {
                    return &slot;
                }
            }
            return nullptr;
        };

        if (mode == SCHED_FULL) {
            // displaced running requests: a higher class took their cells. Between decodes every token of a slot has
            // been decoded, so the state is saved as it stands and nothing in a batch has to be undone
            for (int id : out.suspend) {
                server_slot * slot = slot_of_task(id);
                if (slot == nullptr || slot->state == SLOT_STATE_SUSPENDED || slot->state == SLOT_STATE_WAIT_OTHER ||
                        slot->task->is_child() || slot->task->is_parent() || !prompt_cache) {
                    continue;
                }
                SLT_WRN(*slot, "displaced by a higher priority request: suspending (%d tokens)%BS%n", slot->prompt.n_tokens());
                slot->spec_i_batch.clear();
                slot->i_batch = -1;
                if (slot->suspend()) {
                    n_kv_preempted++;
                    sched_dirty = true;
                }
            }
        }

        if (mode != SCHED_LIGHT) {
            for (int id : out.purge) {
                for (auto & slot : slots) {
                    if (slot.id == id && !slot.is_processing() && slot.prompt.n_tokens() > 0) {
                        purge_idle(slot, kv_cells_owned({ slot.id }));
                        sched_purged_last++;
                    }
                }
            }
        }

        std::vector<int> waiting_starts;
        for (int id : out.start) {
            server_slot * slot = slot_of_task(id);
            if (slot != nullptr && slot->state == SLOT_STATE_SUSPENDED) {
                if (mode != SCHED_FULL) {
                    continue;
                }
                // a restore that failed is retried only once the occupancy has dropped below where it failed
                const int32_t cells_now = kv_cells_used();
                if (slot->resume_failures > 0 && cells_now >= slot->resume_last_used) {
                    continue;
                }
                slot->resume_last_used = cells_now;
                if (!slot->resume()) {
                    if (++slot->resume_failures >= 8) {
                        send_error(*slot, "failed to restore a suspended request", ERROR_TYPE_SERVER);
                        slot->release();
                    } else {
                        SLT_WRN(*slot, "restore attempt %d failed, retrying later%BS%n", slot->resume_failures);
                    }
                }
                sched_dirty = true;
                continue;
            }
            if (new_task != nullptr && id == new_task->id) {
                continue; // the caller launches it
            }
            waiting_starts.push_back(id);
        }

        if (!waiting_starts.empty()) {
            // they come back through process_single_task, where the decision is confirmed and the launch happens
            queue_tasks.pop_deferred_tasks(waiting_starts);
        }

        return out;
    }

    // customware (entry 012): the state at the end of a generation, taken when the generation ends. Before, it was
''')

# process_single_task: the decision before the slot choice; the admission block goes
s = rep(s, nl, '''                    const int id_task = task.id;

                    server_slot * slot = get_available_slot(task);
''', '''                    const int id_task = task.id;

                    // customware (entry 013): under --kv-unified the scheduler decides whether this request starts now
                    if (params_base.kv_unified && task.type == SERVER_TASK_TYPE_COMPLETION) {
                        if (task.t_arrival == 0) {
                            task.t_arrival = ggml_time_us();
                        }
                        const sched_output out = schedule("task", &task, SCHED_FULL);
                        if (std::find(out.start.begin(), out.start.end(), task.id) == out.start.end()) {
                            const sched_placement pl = plan_placement(task);
                            SRV_WRN("task %d needs %d cells (prompt %d, priority %d) but %s: waiting for cells%BS%n",
                                    id_task, pl.need, task.n_tokens(), task.params.priority, out.why.c_str());
                            n_kv_deferred++;
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                    }

                    server_slot * slot = get_available_slot(task);
''')

old_admission_start = '                    // customware (entry 007): under --kv-unified a request starts only when it fits next to the active'
old_admission_end = '''                            n_kv_deferred++;
                            queue_tasks.defer(std::move(task));
                            break;
                        }
                    }
'''.replace('\n', nl)
i = s.index(old_admission_start)
j = s.index(old_admission_end, i) + len(old_admission_end)
s = s[:i] + '                    // customware (entry 013): admission is the scheduler\'s decision above (entry 007\'s check was here)' + nl + s[j:]

# the idle-slot save-and-clear at launch: not under the scheduler
s = rep(s, nl, '''                    if (params_base.cache_idle_slots) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache%BS%n");
''', '''                    // customware (entry 013): under --kv-unified idle conversations keep their cells until a decision
                    // purges them (least recent first, when cells or a slot are needed); a launch no longer saves and
                    // clears every idle slot
                    if (params_base.cache_idle_slots && !params_base.kv_unified) {
                        for (auto & slot : slots) {
                            if (!slot.is_processing()) {
                                SLT_TRC(slot, "%s", "saving idle slot to prompt cache%BS%n");
''')

# update_slots: the tick replaces the resume pass
s = rep(s, nl, '''        // customware (entry 007): suspended sequences come back as soon as their cells are free
        if (params_base.kv_unified) {
            try_resume_suspended();
        }
''', '''        // customware (entry 013): a full decision when something changed, and every 2 s for the aging
        if (params_base.kv_unified && (sched_dirty || ggml_time_us() - t_sched_last > 2000000)) {
            schedule("tick", nullptr, SCHED_FULL);
        }
''')

# decode failure: the decision purges, then its victim
s = rep(s, nl, '''            if (ret == 1 && n_batch == 1 && params_base.kv_unified) {
                if (try_clear_idle_slots()) {
                    return false; // retry with the same batch
                }
                if (try_preempt(off)) {
                    n_batch = llama_n_batch(ctx_tgt);
                    return false; // retry the (now smaller) batch at full size
                }
            }
''', '''            if (ret == 1 && n_batch == 1 && params_base.kv_unified) {
                // customware (entry 013): the decision purges what it would purge; if that frees nothing, its victim
                // (the lowest-ranked running request, or one it displaced) leaves the batch and is suspended
                const sched_output out = schedule("overflow", nullptr, SCHED_EMERGENCY);
                if (sched_purged_last > 0 || try_clear_idle_slots()) {
                    return false; // retry with the same batch
                }
                const int victim = !out.suspend.empty() ? out.suspend.front() : out.victim;
                if (try_preempt(off, victim)) {
                    n_batch = llama_n_batch(ctx_tgt);
                    sched_dirty = true;
                    return false; // retry the (now smaller) batch at full size
                }
            }
''')

save(p, s)
print('server-level edits applied')
