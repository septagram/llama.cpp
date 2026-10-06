# customware/011: a waiting request stays alive, and the queue is retried as a whole

## Goal

A streaming request that has to wait (queued for KV cells, queued for a slot, its state being restored, a long
prefill) must not be dropped by its client. Contention costs latency, never a failed request: the promise of
entry 007, which today holds only as long as the client's own timeout allows.

## What happens today (read in the code, measured in the launcher logs)

- The server already pings a silent stream: `--sse-ping-interval` (30 s by default) sends the SSE comment
  `:\n\n` between results. But only after the stream has started.
- A stream starts with the first result. `handle_completions_impl` waits for it before it sends anything, the
  headers included, so that an early error can still be a plain HTTP error. A request that is queued or
  prefilling therefore sends nothing at all. Node clients give up after 300 s without headers (undici's
  `headersTimeout`).
- In the logs of 2026-09-30 .. 10-06: 34 requests were cancelled by the client while still queued (25 of 28 in
  one session at 284-305 s), and 4 prefills longer than 295 s ended with the client gone (405,923 tokens
  prefilled for nobody). A cold prompt above ~69k tokens takes more than 300 s on this box.
- The deferred queue is retried one request per slot release (a request pinned to the released slot, else the
  first). A request that still does not fit goes to the back, and the others wait for the next release even when
  a slot is free and they would fit.

## Design

1. **Commit to the stream after one ping interval.** In the streaming branch, wait for the first result at most
   `sse_ping_interval` seconds. If it comes in time, nothing changes (an error is still an HTTP error). If not,
   send the headers (200, `text/event-stream`) with one ping as the first chunk and hand over to the existing
   streaming loop, which already pings on silence, delivers results as they come, and formats an error that
   arrives after the first chunk as an in-stream error event. No new ping format, no new option: the interval
   and its per-request override (`sse_ping_interval`) keep their meaning, and -1 keeps today's behaviour.
2. **Retry the whole deferred queue on a release**, in deferral order, requests pinned to the released slot
   first. Each is admitted or deferred again, so the order is preserved.
3. Non-streaming requests are out of scope: nothing can be sent before the JSON body without committing the
   status code. Recorded as a limitation; leading-whitespace chunks would be the opt-in if a client needs it.

## Steps

1. `tools/server/server-context.cpp`, `handle_completions_impl`: the bounded wait for the first result and the
   early commit.
2. `tools/server/server-queue.cpp`: `pop_deferred_task` moves every deferred task back.
3. `customware/test-keepalive.sh` (real model, port 5011, pool 24576, `--sse-ping-interval 5`), with a small
   Python client that timestamps the headers and every chunk:
   - Q: a streaming request queued behind a generating one for ~40 s: headers within ~6 s, a ping every ~5 s
     until the first data chunk, then the normal stream ending in `[DONE]`.
   - P: a cold 22k streaming request: pings during the prefill.
   - E: a prompt larger than the pool still answers with an HTTP error status and a JSON error, not a stream.
   - R: both slots busy, a large request (cannot fit) and a small one queued behind it; when the small slot is
     released the queued small one starts at once, before the large active one finishes.
4. Release, launcher check, note `customware/011-keepalive.md`, NEXT.md, dashboard, memory.

## Verification

- Q and P show the headers and the pings with timestamps; E shows the status code; R shows the launch order in
  the server log.
- With `--sse-ping-interval -1` the first-result wait is unbounded as before.

## Risks

- A client that treats an SSE comment before the first event as an error. The same comment is already sent
  mid-stream by default, and OpenRouter sends comments before the first event, so parsers that survive those
  survive this.
- An error after the early commit arrives as an in-stream error event with status 200. Only requests that
  waited longer than the interval are affected; before this entry those requests got no answer at all once the
  client timed out.

## Implementation notes and results (2026-10-06)

- Correction to "What happens today": a stream does not start with the first token but with a begin marker that
  the server emits as soon as a task starts processing (`is_begin`, a result whose JSON is null; upstream uses it
  to send the headers). So a launched request already had its headers and its pings during a long prefill. The
  silent window is only before the launch: queued for a slot or for cells, and the state restore. The four long
  prefills that ended with the client gone had headers and pings; that client's own five-minute limit ended them.
- `handle_completions_impl`: the first wait is bounded by `sse_ping_interval`; on timeout the response commits
  with the SSE comment `:\n\n` as its first chunk. The streaming loop turns a null-JSON result into a ping: after an early
  commit the begin marker reaches the loop, and the first test run showed it going out as `data: null` (the test
  client crashed on it, as a strict client would).
- `server_queue::pop_deferred_task` moves every deferred task back, pinned ones first, in deferral order.
- No launcher change: the default interval is 30 s, so a queued request gets its headers after 30 s and a ping
  every 30 s.

`customware/test-keepalive.sh ka2` (pool 24576, `--sse-ping-interval 5`, `sse-client.py`):

| case | result |
|------|--------|
| F, answers at once | headers at 0.0 s, no ping before the first data event, `[DONE]` |
| E, prompt larger than the pool | HTTP 400, JSON `exceed_context_size_error`, not a stream |
| Q, queued behind a generating request, then a cold 12.3k prefill | headers after 5.1 s, 17 pings before the first data event (82.2 s), longest silence 5.0 s, 50 data events, `[DONE]`, no malformed event |
| R, both slots busy, large Z and small D queued in that order | D answered at +36 s, before the active large X (+67 s); Z at +70 s from the cache (`cache_n` 12337) |

Known and left alone: a deferred request that is retried and still cannot start may make the slot picker save
the least recently used idle slot to the prompt cache without launching anything (once per distinct state);
with the whole queue retried on each release this happens earlier, not more often. The scheduler rewrite
(notes-scheduler.md) removes the side effect.
