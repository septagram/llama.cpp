#!/usr/bin/env python
"""Entry 013 follow-up: a prompt the pool cannot hold even alone is refused at once instead of waiting for cells that
never come (the regression suite's kvu scenario 4 and keep-alive scenario E: a 44k prompt on a 32k / 24k pool waited
until the client gave up); and a request whose growth reserve exceeds the pool is charged the pool, so it runs alone
as the pre-013 admission let it (the prompt fits; the generation may still overflow and then fails as before)."""
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

patch('tools/server/server-context.cpp', [
(
"""                    // customware (entry 013): under --kv-unified the scheduler decides whether this request starts now
                    if (params_base.kv_unified && task.type == SERVER_TASK_TYPE_COMPLETION) {
                        if (task.t_arrival == 0) {
                            task.t_arrival = ggml_time_us();
                        }
                        const sched_output out = schedule("task", &task, SCHED_FULL);
""",
"""                    // customware (entry 013): under --kv-unified the scheduler decides whether this request starts now
                    if (params_base.kv_unified && task.type == SERVER_TASK_TYPE_COMPLETION) {
                        if (task.t_arrival == 0) {
                            task.t_arrival = ggml_time_us();
                        }
                        // a prompt the pool cannot hold even alone is refused here, as the launch refuses it; waiting
                        // for cells that never come is not an answer (a 44k prompt on a 32k pool waited until the
                        // client gave up)
                        if (task.n_tokens() >= kv_pool_size()) {
                            send_error(task, string_format("request (%d tokens) exceeds the available context size (%d tokens), try increasing it",
                                    task.n_tokens(), kv_pool_size()), ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                            break;
                        }
                        const sched_output out = schedule("task", &task, SCHED_FULL);
"""),
(
"""            r.state     = SCHED_REQ_WAITING;
            r.need       = pl.need;
            r.reuse      = pl.reuse;
            r.reuse_keep = pl.keep;
            in.reqs.push_back(r);
""",
"""            r.state     = SCHED_REQ_WAITING;
            // a request whose growth reserve exceeds the pool is charged the pool: it runs alone, as entry 007's
            // admission let it (the prompt fits; a generation that outgrows the pool fails as before)
            r.need       = std::min(pl.need, in.pool);
            r.reuse      = pl.reuse;
            r.reuse_keep = pl.keep;
            in.reqs.push_back(r);
"""),
])

print('patch-013c: done')
