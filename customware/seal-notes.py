#!/usr/bin/env python
"""Write each customware entry's "Final commit list" section from customware/entry-commits.txt: the patch it is
sealed into, its commits (code or notes), and where its code travels when history interleaved two entries. An
existing "## Final commit list" section is replaced; otherwise the section is appended. Entries without an
"Implementation notes" heading get a pointer to where their results and deviations are recorded."""
import os
import re
import subprocess
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

UPSTREAM = '6e60f3560'
SEALED = '2026-10-20'

NOTES = {
    '004': "This patch also carries entry 005's code commit `1fdd1bd9e` (hot experts parked in the compute buffer "
           "tail), at its head: it landed before 004's code, and 004's adaptive experts edit the same lines, so the "
           "two cannot be separated (a three-way merge conflicts).",
    '005': "No patch of its own: its code commit `1fdd1bd9e` is carried at the head of `004-adaptive-hot-experts.patch` "
           "(it landed before 004's code, and 004 edits the same lines). Its other commits are notes and launchers.",
    '006': "Its server-side batch-cut change landed in entry 007's commit `df79272fd` and travels in "
           "`007-kvu-queueing-preemption.patch`.",
    '007': "Its commit `df79272fd` also carries entry 006's server-side batch cut. Its admission follow-up (a request "
           "never waits for cells nobody holds; the reserve capped at `LLAMA_KV_RESERVE`) landed in `f6a59d2e8` "
           "together with entry 008's profiling groundwork, and travels in `008-depth-profile.patch`.",
    '008': "Its first commit `f6a59d2e8` also carries entry 007's admission follow-up and the launchers' vision projector.",
    '013': "`e49f3c06a` and `cdda90e0b` are the follow-ups from the regression suite; they landed after entry 014's "
           "engine commit and apply here in sequence.",
    '014': "This patch also carries the fixes of the review cadence over entries 010-014 (`46b6c069d`, `6e708180e`, "
           "`3d68d48f6`, `b3fe31f24`, `5c6a27f06`): they touch the code of 010-014 together and were written on top of "
           "014. The findings, verdicts and what each fix is for: `customware/REPORT-2026-10-07.md`.",
}


def git(*args):
    return subprocess.run(['git', *args], capture_output=True, text=True, encoding='utf-8', check=True).stdout


def touches_code(c):
    return git('diff', '--name-only', c + '^', c, '--', ':(exclude)customware').strip() != ''


entries = []
for line in open('customware/entry-commits.txt', encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    parts = line.split()
    entries.append((parts[0], parts[1:]))

for entry, commits in entries:
    md = next(f for f in sorted(os.listdir('customware')) if f.startswith(entry + '-') and f.endswith('.md'))
    path = os.path.join('customware', md)
    patch = md[:-3] + '.patch'
    text = open(path, encoding='utf-8').read()

    lines = ['## Final commit list', '']
    if os.path.exists(os.path.join('customware', patch)):
        lines.append('Sealed %s into `%s`: this entry\'s own diff (customware/ excluded), applied after the entries '
                     'before it onto upstream `%s` (tag `v0.5.0-customware.0`); `customware/seal-entries.sh` rebuilds '
                     'and checks it from `customware/entry-commits.txt`.' % (SEALED, patch, UPSTREAM))
    else:
        lines.append('Sealed %s with no patch file of its own (see below).' % SEALED)
    lines.append('')
    for c in commits:
        subj = git('log', '-1', '--format=%h %s', c).strip()
        h, s = subj.split(' ', 1)
        kind = '' if touches_code(c) else ' (notes only)'
        lines.append('- `%s` %s%s' % (h, s, kind))
    if entry in NOTES:
        lines += ['', NOTES[entry]]
    if not re.search(r'^## Implementation', text, re.M):
        lines += ['', 'What shipped, measured results and deviations from the plan are recorded in the sections above '
                      '(this entry predates the "Implementation notes" heading).']
    section = '\n'.join(lines) + '\n'

    m = re.search(r'^## Final commit list\n.*?(?=^## |\Z)', text, re.M | re.S)
    if m:
        text = text[:m.start()] + section + ('\n' if text[m.end():].strip() else '') + text[m.end():]
    else:
        text = text.rstrip('\n') + '\n\n' + section
    open(path, 'w', encoding='utf-8', newline='\n').write(text)
    print('%s: %d commits, %s' % (md, len(commits), 'replaced' if m else 'appended'))
