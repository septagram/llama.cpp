#!/usr/bin/env python
"""Build the requests of a fork test from a base request and the answer the server gave to it.

usage: fork-requests.py <base.json> <answer.json> <out-prefix> <n_predict> <tag1> <tag2> [...]
Each output <out-prefix>-<tag>.json is the base conversation plus the assistant turn from the answer (its content,
with the reasoning text in front when the template preserves it) plus a user message "[<tag>] <follow-up>", greedy,
thinking off, ignore_eos, n_predict tokens. All of them share the same prefix up to the end of the assistant turn,
which is the fork point."""
import json
import sys

base, answer, prefix, n_predict = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tags = sys.argv[5:]
d = json.load(open(base))
a = json.load(open(answer))
msg = a['choices'][0]['message']
assistant = {'role': 'assistant', 'content': msg.get('content') or ''}
if msg.get('reasoning_content'):
    assistant['reasoning_content'] = msg['reasoning_content']
for tag in tags:
    r = dict(d)
    r['messages'] = list(d['messages']) + [assistant, {'role': 'user', 'content': '[%s] Continue with two more paragraphs about %s.' % (tag, tag)}]
    r['max_tokens'] = n_predict
    r['temperature'] = 0
    r['top_k'] = 1
    r['seed'] = 7
    r['ignore_eos'] = True
    r['chat_template_kwargs'] = {'enable_thinking': False}
    r['reasoning_budget'] = 0
    json.dump(r, open('%s-%s.json' % (prefix, tag), 'w'))
print('wrote', ', '.join('%s-%s.json' % (prefix, t) for t in tags))
