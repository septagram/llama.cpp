You are working in the llmsniff repository (Rust, a streaming sniffer for OpenAI-compatible LLM traffic). Its `--headless` mode prints plain tagged lines instead of the TUI, but it was never curated and is noisy.

Task, in this order:

1. Read `src/` and understand how the headless output is produced and how requests, streamed deltas, tool calls and completions flow through the program.
2. Make the headless output concise and useful for watching a coding agent drive a local `llama-server` from a terminal:
   - one line when a request starts: request id, model, and the number of messages;
   - the streamed assistant text folded into lines of at most 160 characters, each prefixed with the request id;
   - reasoning/thinking content shown the same way but with a distinct prefix, and skipped entirely when `--strip-reasoning` is given;
   - each tool call on one line as `[tool] <name>(<compact arguments>)`;
   - one line when a request ends: request id, finish reason, and the `timings` from the upstream response if present (prompt tokens, prompt t/s, generated tokens, generation t/s);
   - no heartbeat, keep-alive or empty-delta noise.
3. Do not change the TUI behaviour.
4. Run `cargo build` and fix every error and warning you introduced. Run `cargo test` if tests exist.
5. Commit your changes to the current branch with a descriptive commit message.

Work autonomously. Do not ask questions; make reasonable decisions and note them in the commit message.
