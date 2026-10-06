#!/usr/bin/env python
"""Entry 012 follow-up: the prompt cache must never erase a pinned (suspended-slot) entry through its prompt
matching. alloc() treated the pinned entry's empty token list as 'fully contained in the new prompt' and erased
it while a slot was suspended; the resume then read freed memory. select() skips them too (its f_keep would be
0/0 for an empty list)."""
p = 'tools/server/server-task.cpp'
s = open(p, encoding='utf-8', newline='').read()
nl = chr(13) + chr(10) if (chr(13) + chr(10)) in s else chr(10)
bs = chr(92)


def rep(old, new, count=1):
    global s
    old = old.replace('\n', nl).replace('%BS%', bs)
    new = new.replace('\n', nl).replace('%BS%', bs)
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);
''', '''    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        // customware (entry 012): a suspended slot's parked state has no token list and takes no part in matching
        if (it->pinned) {
            continue;
        }

        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);
''')

rep('''    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);
''', '''    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        // customware (entry 012): never a pinned entry: its empty token list is "contained" in every prompt, and a
        // suspended slot holds a pointer to it (found by a fork that was suspended, then a purge that saved first)
        if (it->pinned) {
            ++it;
            continue;
        }

        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);
''')

rep('''    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.cbegin(); it != states.cend(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
''', '''    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.cbegin(); it != states.cend(); ++it) {
        // customware (entry 012): pinned entries belong to suspended slots and are not prompts to load
        if (it->pinned) {
            continue;
        }

        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);
''')

open(p, 'w', encoding='utf-8', newline='').write(s)
print('prompt cache pinned-entry fix applied')
