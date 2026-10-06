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
