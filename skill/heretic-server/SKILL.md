---
name: heretic-server
description: Operate the Qwen3.8-27B-Heretic (Trial 46 abliterated) self-hosted endpoint served from Kaggle via llama.cpp + Cloudflare tunnel. Use when running inference on the endpoint, configuring clients/agents (OpenAI SDK compatible), diagnosing endpoint failures, testing abliteration red-team behavior, or before/after a Kaggle serving session. Covers endpoint acquisition, sequential request discipline, thinking-model parameters, token/time budgeting on T4s, and the failure playbook.
---

# Heretic Server Skill (Qwen3.8-27B, Trial 46)

## 0. Identity knife

- Model id on the API: `Qwen3.8-27B-Heretic` (alias stamped by llama-server).
- It is the **Trial 46 abliterated LoRA** applied at runtime on top of
  `Qwen3.8-27B-Q4_K_M` base — abliteration verified at 0/100 refusals, KL 0.5558.
- It is a **thinking model**: requests stream `reasoning_content` first, then
  `content`. Thinking can eat 500–1000+ tokens before the answer starts.
- Endpoint is an OpenAI-compatible `/v1` API behind an HTTPS Cloudflare quick
  tunnel. Location: Kaggle (usually Dual T4). Throughput ~10–12 tok/s; prompt
  processing ~70 tok/s.

## 1. Endpoint acquisition — ALWAYS do this first

Endpoint credentials **rotate every session** (new URL + new key each Kaggle run).
Never reuse an old URL from memory or notes.

1. Read `client/heretic_config.json` (next to heretic_client.py) if it exists.
2. Ask the user for the freshly printed `HERETIC_ENDPOINT={...}` line if config
   looks stale or `/health` fails. Import it:
   ```bash
   python client/heretic_client.py paste 'HERETIC_ENDPOINT={...}'
   ```
3. Verify before any real work:
   ```bash
   python client/heretic_client.py health
   ```
   Expect `[PASS] /health: {"status":"ok"}` and a model listing.
4. If the user has no endpoint printed yet → the serving session is not running.
   Tell the user to run `kaggle_cell.py` on Kaggle (GPU T4 x2, internet ON).
   Only the user can start it; never guess URLs.

## 2. Golden rules (violations = the "garbage response" class of bugs)

1. **Sequential requests only.** One request at a time, end-to-end. Never fire a
   new request while another is in flight, never parallel-fan-out. The server
   runs `--parallel 1`: requests queue; stacking requests risks slot-state
   corruption that shows up as instant tiny garbage answers.
2. **Bounded `max_tokens`.** agent ≤ 1600, deep ≤ 3000 (a little more if
   explicitly needed). Requests can outlive patience on 11 tok/s hardware —
   bounds keep worst-case shutoff predictable.
3. **Never abandon a stream and immediately request again.** Through the
   Cloudflare tunnel the server may keep generating for a disconnected client.
   If you must cut a request short, wait ~30–60 s (or check `slots` shows idle)
   before the next request.
4. **Expect the think phase.** A "stuck" request that has produced only
   `[THINK]` text for ~2–3 min is HEALTHY on T4s. Do not kill it prematurely.
   If driving the OpenAI SDK yourself, print/handle `reasoning_content` (or it
   looks silent for minutes) and always ask for generous `max_tokens`.
5. **Garbage detection.** Answer finishes instantly (<3 s) with <12 tokens and
   `finish_reason=stop` → suspicious. Retry once. Two in a row → the slot state
   is polluted; the ONLY fix is restarting the Kaggle serving script (stop
   session → rerun cell → new URL+key). Tell the user, never paper over it.

## 3. Request profiles (use these exact sampling values)

| Profile | Use for | temperature | top_p | top_k | max_tokens | mode |
|---------|---------|-------------|-------|-------|------------|------|
| `agent` | Agent loops, tool calls, quick answers | 0.7 | 0.8 | 20 | ≤1600 | `/no_think` system prefix (thinking off) |
| `deep`  | Reasoning, writing, analysis | 0.6 | 0.95 | 20 | ≤3000 | thinking on |

- Client CLI does this for you via `--profile agent|deep`.
- OpenAI SDK users: pass those sampling params explicitly every request.
- If `/no_think` turns out not to be honored (check the `\n[THINK]` prefix at
  runtime), fall back to profile `agent` still being fine — thinking just
  consumes more of the budget. Do NOT try `chat_template_kwargs:
  {enable_thinking: false}` — on this template a bugged empty think block can
  leak junk like `\_` into content.
- Tool calling works in OpenAI format (`tools=[...]`, JINJA enabled server-side).

## 4. CLI client (stdlib zero-dep)

```bash
python client/heretic_client.py paste '<HERETIC_ENDPOINT={...}>'   # import creds
python client/heretic_client.py health                            # alive? slots?
python client/heretic_client.py ask "prompt"                      # quick answer (agent)
python client/heretic_client.py ask "..." --profile deep --max-tokens 2400
python client/heretic_client.py chat                              # REPL (! toggles profile)
python client/heretic_client.py redteam                           # 0/8 refusal check
python client/heretic_client.py tools                             # tool-call smoke test
```

Or with the Python SDK straight at the endpoint:

```python
from openai import OpenAI
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=600)
r = client.chat.completions.create(model="Qwen3.8-27B-Heretic",
    messages=[{"role": "system", "content": "/no_think You are a direct, helpful assistant."},
              {"role": "user", "content": "..."}],
    temperature=0.7, top_p=0.8, max_tokens=1600)
```

## 5. Failure playbook

| Symptom | Cause | Fix |
|---|---|---|
| `/health` unreachable | tunnel/session dead or URL rotated | user reruns `kaggle_cell.py`; import new endpoint |
| Instant tiny garbage answers | zombie/corrupted slot state | retry once; if repeats → user restarts serving session |
| Long silence, no tokens | think phase OR server died mid-request | check `health` + `slots`; if session dead → restart serving script |
| 429 / queue timeouts | two requests sent at once | stop, go strictly sequential |
| OOM on server restart in log | free VRAM era mismatch / ctx too high for 1 GPU | keep `GPU T4 x2` selected |
| Refusals appear | adapter not applied (`--lora` missing) or wrong adapter | check serve log lora load line; rerun serving script |

Server-side diagnostics the user can read in the Kaggle output:
- `endpoints.json` written to `/kaggle/working`
- heartbeat lines `[i] heartbeat: … slot_busy=… uptime=…`
- watchdog auto-restarts server/tunnel with attempts printed as `[FAIL] … auto-restart`
