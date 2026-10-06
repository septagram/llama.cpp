#!/usr/bin/env python
"""Post a streaming chat request and timestamp what comes back: the headers, every SSE comment (ping) and
every data event. Entry 011 uses it to show that a waiting request is kept alive.

usage: sse-client.py <port> <request.json> <name> [--detail]
prints one summary line; with --detail also one line per ping / first data events."""
import http.client
import json
import sys
import time

port, path, name = int(sys.argv[1]), sys.argv[2], sys.argv[3]
detail = '--detail' in sys.argv
body = json.load(open(path))
body['stream'] = True
payload = json.dumps(body)

t0 = time.time()
conn = http.client.HTTPConnection('127.0.0.1', port, timeout=3600)
conn.request('POST', '/v1/chat/completions', payload, {'Content-Type': 'application/json'})
resp = conn.getresponse()
t_headers = time.time() - t0
ctype = resp.getheader('Content-Type', '')

if resp.status != 200 or 'event-stream' not in ctype:
    text = resp.read().decode('utf-8', 'replace')
    try:
        err = json.loads(text).get('error', {})
        what = '%s: %s' % (err.get('type'), (err.get('message') or '')[:80])
    except Exception:
        what = text[:120]
    print('  %s: HTTP %d after %.1f s, %s, not a stream: %s' % (name, resp.status, t_headers, ctype, what))
    sys.exit(0)

pings = []          # times of comments before the first data event
pings_late = 0      # comments after it
t_first_data = None
n_data = 0
done = False
error = None
last = t_headers
max_gap = 0.0
text_out = ''
while True:
    line = resp.readline()
    if not line:
        break
    line = line.rstrip(b'\r\n')
    if not line:
        continue
    now = time.time() - t0
    max_gap = max(max_gap, now - last)
    last = now
    if line.startswith(b':'):
        if t_first_data is None:
            pings.append(now)
            if detail:
                print('    +%6.1f s  ping' % now)
        else:
            pings_late += 1
        continue
    if line.startswith(b'data:'):
        data = line[5:].strip()
        if data == b'[DONE]':
            done = True
            continue
        try:
            obj = json.loads(data)
        except Exception:
            continue
        if 'error' in obj:
            error = obj['error']
            continue
        if t_first_data is None:
            t_first_data = now
            if detail:
                print('    +%6.1f s  first data event' % now)
        n_data += 1
        for ch in obj.get('choices', []):
            text_out += (ch.get('delta', {}).get('content') or '')
t_end = time.time() - t0
print('  %s: HTTP %d, headers after %.1f s, %d pings before the first data event%s, first data at %s, %d data events, '
      'longest silence %.1f s, %s, %.0f s total, %d chars'
      % (name, resp.status, t_headers, len(pings),
         (' (at ' + ', '.join('%.0f' % p for p in pings[:4]) + (', ...' if len(pings) > 4 else '') + ' s)') if pings else '',
         ('+%.1f s' % t_first_data) if t_first_data is not None else 'never',
         n_data, max_gap, 'ended with [DONE]' if done else ('in-stream error %s' % error if error else 'no [DONE]'), t_end, len(text_out)))
