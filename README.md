# Qwen3.8-27B Heretic (Trial 46) — Production Serving Suite

Self-hosted **OpenAI-compatible** inference for the Trial 46 abliterated
Qwen3.8-27B model: Kaggle (Dual T4) + llama.cpp CUDA + Cloudflare HTTPS tunnel,
with a production-grade serving script, a zero-dependency client CLI, and an
agent skill.

Repo layout:

```
qwen-heretic-serve/
├── kaggle_cell.py              # ← the ONLY cell you paste in Kaggle (bootstrap)
├── server/serve.py             # full production serving pipeline (9 stages)
├── client/heretic_client.py    # stdlib-zero-dep CLI (health/ask/chat/redteam/tools)
├── client/heretic_config.json  # created at runtime (gitignored — contains key)
├── skill/heretic-server/SKILL.md  # agent skill (OpenCode / Claude Code standard)
└── README.md
```

## Properties

- **Pull-and-run**: Kaggle cell clones this repo every run → all fixes propagate
  by re-running the cell. Nothing is ever pasted twice.
- **Self-verifying server flags**: llama-server flags are cross-checked against
  `--help` at runtime → a different llama.cpp build can't brick the launch.
- **Self-verification battery**: after load, 4 probes run (identity, math,
  `/no_think` switch check, red-team). Failures auto-restart the server (3x)
  before anything is reported LIVE.
- **Watchdog**: server + tunnel are auto-restarted if they die; heartbeat lines
  show slot/VRAM state. New tunnel URL is re-written to `endpoints.json`.
- **Credentials rotate per session** and are delivered as ONE greppable line:
  `HERETIC_ENDPOINT={...}` → `heretic_client.py paste '<line>'`. The HF token
  lives only in your Kaggle secret / cell, never in this public repo.
- **Sequential discipline + bounded tokens** are enforced client-side and
  documented in the skill — kills the "instant garbage answer" failure class
  (corrupted slot state from overlapping/abandoned requests).

## Kaggle (serve side) — one cell

1. Notebook options → **Accelerator: GPU T4 x2**, **Internet: ON**.
2. Add-ons → Secrets → add `HF_TOKEN` = your Hugging Face token with read
   access to the private adapter repo `xerofounder/Qwen3.8-27B-Heretic`
   (or paste it into `HF_TOKEN = ""` in the cell).
3. Set `REPO_URL` at the top of `kaggle_cell.py` to your pushed repo.
4. Paste `kaggle_cell.py` content into a cell, run, wait ~4–6 min.

Output: stage markers `[>>>>]`, `[PASS]`/`[FAIL]` lines, live aria2 speed,
server command, probe results, and finally:

```
HERETIC_ENDPOINT={"base_url":"https://xxx.trycloudflare.com/v1","api_key":"sk-...",...}
```

## Client side

```bash
python client/heretic_client.py paste 'HERETIC_ENDPOINT={...}'
python client/heretic_client.py health
python client/heretic_client.py ask "Write a compact TCP port scanner in python."
python client/heretic_client.py chat          # REPL, '!' toggles agent<->deep
```

Agents (OpenAI SDK / OpenCode / Claude Code): read
`skill/heretic-server/SKILL.md` for profiles, discipline and failure playbook.

## Serving parameters (production defaults)

| Param | Value | Why |
|---|---|---|
| context | 49152 | 48k; KV q8_0 → ~7 GB on 32 GB VRAM total |
| `--parallel` | 1 | one full-context slot; sequential agents; no cross-slot pollution |
| KV cache | q8_0/k+v | near-lossless, half memory vs f16 |
| sampling | T0.6 / K20 / P0.95 | Qwen3 thinking-mode recommendation (backstop; clients override per request) |
| reasoning | deepseek format | `reasoning_content` separated from `content` |
| llama.cpp | pinned `b11669` CUDA 12.8 | prebuilt; flag-checks at runtime |

## Troubleshooting quick map

- **Instant tiny garbage answers** → polluted slot state: retry once, then
  restart the Kaggle serving session (stop + rerun cell). New URL/key printed.
- **`/health` down** → tunnel URL rotated or session stopped; rerun serving.
- **Think phase looks like a hang** → it isn't: ~2–3 min of `[THINK]` is normal
  before the answer at ~11 tok/s.
- **Refusals reappear** → adapter didn't load; check serve log
  `llama_server.log` in `/kaggle/working` for the `--lora` line.

## High-throughput upgrade path (later)

Modal L40S/A100 deployment when >30 tok/s or 64k+ context is needed —
`python -m modal token set` locally; same client works (OpenAI-compatible).
