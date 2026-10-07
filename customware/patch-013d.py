#!/usr/bin/env python
"""Two edits in server-context.cpp: (1) the SLT_INF of the entry 014 tail restore in get_available_slot got a literal
newline inside its C string (patch-014b.py had a raw backslash-n there): join it back; (2) entry 013 follow-up: a
pool-overflow victim that could never come back (the cells it holds alone plus the resume margin exceed the pool) is
not parked, it fails alone with the context error, and the others go on with its cells."""
import sys

BS = chr(92)
NL = chr(10)

def patch(path, edits):
    src = open(path, encoding='utf-8').read()
    for old, new in edits:
        old = old.replace('%BS%', BS).replace('%NL%', NL)
        new = new.replace('%BS%', BS).replace('%NL%', NL)
        n = src.count(old)
        if n != 1:
            print('FAIL %s: anchor found %d times:\n%s' % (path, n, old[:300]))
            sys.exit(1)
        src = src.replace(old, new)
    open(path, 'w', encoding='utf-8', newline='\n').write(src)
    print('patched', path, len(edits), 'edits')

patch('tools/server/server-context.cpp', [
(
"""restoring the tail here, next to it%NL%",
""",
"""restoring the tail here, next to it%BS%n",
"""),
(
"""        // the sampled token was appended to the prompt when it entered the batch; it was not decoded
        if (!removed.empty()) {
            victim->prompt.tokens.keep_first(victim->prompt.tokens.size() - removed.size());
        }

        if (!victim->suspend()) {
            return false;
        }
""",
"""        // the sampled token was appended to the prompt when it entered the batch; it was not decoded
        if (!removed.empty()) {
            victim->prompt.tokens.keep_first(victim->prompt.tokens.size() - removed.size());
        }

        // customware (entry 013, follow-up): a victim that could never come back (the cells it holds alone plus the
        // resume margin exceed the pool) is not parked: it fails alone with the context error, its conversation goes
        // to the cache like any purged one, and the others go on with its cells. Parked, nothing could have brought
        // it back and its client would have waited forever.
        {
            const int32_t owned = kv_cells_owned({ victim->id });
            if (owned + kv_resume_margin() > kv_pool_size()) {
                SRV_WRN("KV pool overflow: slot %d (task %d, %d tokens, %d cells of its own) outgrew the pool of %d cells and could not come back: failing it, the others continue%BS%n",
                        victim->id, victim->task->id, n_before, owned, kv_pool_size());
                send_error(*victim, string_format("context size exceeded: the request (%d tokens) outgrew the KV pool (%d cells)", n_before, kv_pool_size()),
                        ERROR_TYPE_EXCEED_CONTEXT_SIZE);
                victim->release();
                purge_idle(*victim, owned);
                return true;
            }
        }

        if (!victim->suspend()) {
            return false;
        }
"""),
])

print('patch-013d: done')
