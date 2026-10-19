#!/usr/bin/env python
"""Review iteration 4 (26d-blade-shore-essay), the upheld finding: a request displaced before its prompt loop ran
came back GENERATING the conversation its slot held before (resume() judged the state from token counts). The slot
records at suspension whether it was generating and resumes to GENERATING only then; everything else (a prefill in
progress, a launch whose prompt loop had not run) goes back to STARTED, whose block recomputes the common prefix.
Exact-string edits; every anchor must match exactly once."""
import sys

def patch(path, edits):
    src = open(path, encoding='utf-8').read()
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            print('FAIL %s: anchor found %d times:\n%s' % (path, n, old[:300]))
            sys.exit(1)
        src = src.replace(old, new)
    open(path, 'w', encoding='utf-8', newline='\n').write(src)
    print('patched', path, len(edits), 'edits')

OLD1 = """    server_prompt_cache       * cache_for_suspend = nullptr;
    server_prompt_cache_state * suspended_state   = nullptr;
    int64_t                     t_suspended       = 0;
"""
NEW1 = """    server_prompt_cache       * cache_for_suspend = nullptr;
    server_prompt_cache_state * suspended_state   = nullptr;
    int64_t                     t_suspended       = 0;
    // review fix (iteration 4): was the slot generating when it was parked? Only then does it resume as generating;
    // a prefill in progress, or a launch whose prompt loop had not run yet (the slot still holding its previous
    // conversation's tokens, possibly more of them than the new prompt has), goes back to STARTED, whose block
    // recomputes the common prefix. Judged from token counts, a longer old conversation came back as "generating"
    // and streamed its continuation to the new client.
    bool                        suspended_generating = false;
"""

OLD2 = """        mem.seq_rm(id, -1, -1);

        suspended_state = st;
        t_suspended     = ggml_time_us();
        i_batch         = -1;
        state           = SLOT_STATE_SUSPENDED;
"""
NEW2 = """        mem.seq_rm(id, -1, -1);

        suspended_state      = st;
        suspended_generating = state == SLOT_STATE_GENERATING;
        t_suspended          = ggml_time_us();
        i_batch              = -1;
        state                = SLOT_STATE_SUSPENDED;
"""

OLD3 = """        // customware (entry 013): a request suspended during its prompt returns to the prompt loop, which finds the
        // processed prefix in place and continues; a generating one picks up with its pending sampled token
        state = task && prompt.n_tokens() < task->n_tokens() ? SLOT_STATE_STARTED : SLOT_STATE_GENERATING;
"""
NEW3 = """        // customware (entry 013): a request suspended during its prompt returns to the prompt loop, which finds the
        // processed prefix in place and continues; a generating one picks up with its pending sampled token.
        // Review fix (iteration 4): decided by what the slot was doing when parked, not by token counts
        state = suspended_generating ? SLOT_STATE_GENERATING : SLOT_STATE_STARTED;
"""

patch('tools/server/server-context.cpp', [(OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3)])

print('patch-015c: done')
