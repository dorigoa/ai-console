# Sampling Parameters for Local LLMs

> Operational reference for `llama-console`.
> Last updated: **2026-08-18**

---

## TL;DR

1. **Follow the model card values**, not generic heuristics ("low temperature for coding").
2. **Temperature is not an absolute quantity**: it is relative to the distribution that
   particular model learned to produce. `T = 1.0` is the identity.
3. **The samplers' values are a tuple**, not a single number. Taking only `temperature` and leaving
   `top_p` / `top_k` / `min_p` at the runtime defaults is the most common mistake.
4. **llama.cpp defaults silently override the card** unless you set them explicitly —
   `--min-p` and `--top-k` in particular.

---

## 1. Why "low temperature for coding" is outdated advice

The sampler computes:

```
p_i ∝ exp(z_i / T)
```

where `z_i` are the logits. `T = 1.0` returns exactly the model's native distribution; any
other value is a *deformation relative to that distribution*. So `T = 0.3` does not mean the
same thing on two different models — it depends on how sharp the starting distribution
already is.

Modern models go through **RLVR** (*Reinforcement Learning from Verifiable Rewards* —
training with an automatically verifiable reward: the test passes, the proof checks out).
This process **already sharpens** the distribution toward the correct answer. Lowering `T`
further is a second sharpening pass on an already sharp distribution:

- collapse onto a local mode
- repetition loops
- inability to recover from a bad start

This is not theory. The Qwen3 cards explicitly warn against greedy decoding because it
causes performance degradation and endless repetitions.

### Cross-check: what Anthropic does today

The "0–0.3 for coding/extraction" heuristic comes from the pre-reasoning era. Anthropic
itself **no longer allows lowering the temperature**:

- temperature must be `1` (or left unset) whenever thinking is enabled, on all models;
- on 4.7 and later it is deprecated: only the default value is accepted, even with thinking off;
- on Opus 5 / Sonnet 5 / Fable 5 / Mythos, non-default `temperature`, `top_p` or `top_k`
  values return a 400 error on every request.

Source: <https://platform.claude.com/docs/en/build-with-claude/thinking>

### Where low temperature still works

| Regime | Low T | Why |
|---|---|---|
| Data extraction, classification, short structured output | **works** | Short generation → degeneration has no room to set in. You gain determinism. |
| Coding with reasoning, long output, agentic loops | **hurts** | Thousands of tokens: a degenerate mode has plenty of time to take hold. |
| Multi-sample strategies (pass@k, best-of-n) | **hurts** | Kills candidate diversity, which is the entire point of the strategy. |

**Low T buys reproducibility, not accuracy.** These are different things.

---

## 2. Reference table

| Model | Mode | temp | top_p | top_k | min_p | presence_pen. | rep_pen. |
|---|---|---|---|---|---|---|---|
| **Nemotron-3-Super-120B-A12B** | all | 1.0 | 0.95 | 0 † | 0.0 † | 0.0 † | 1.0 † |
| **Nemotron-3.5-Lightning-30B-A3B** | all | 1.0 | 0.95 | 0 † | 0.0 † | 0.0 † | 1.0 † |
| **Qwen3.6-27B** | thinking, general | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| | thinking, precise coding | **0.6** | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| | instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |
| **Qwen3.8-27B** | thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| | instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |
| **Qwen3.6-35B-A3B** | thinking, general | 1.0 | 0.95 | 20 | 0.0 | **1.5** | 1.0 |
| | thinking, precise coding | **0.6** | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| | instruct, general | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |
| | instruct, reasoning | 1.0 | 1.00 | 40 | 0.0 | 2.0 | 1.0 |
| **Qwen3-Coder-30B-A3B-Instruct** | single | 0.7 | 0.80 | 20 | 0.0 | 0.0 † | **1.05** |
| **GPT-OSS-120B** | single | 1.0 | 1.00 | **0 (off)** | 0.0 † | 0 | **1.0 (forbidden)** |
| **Gemma-4-12B / 26B-A4B / 31B** | all | 1.0 | 0.95 | **64** | 0.0 † | 0.0 † | 1.0 † |
| **Llama-3.3-70B-Instruct** | — | *(0.6)* | *(0.9)* | 0 † | 0.0 † | 0.0 † | 1.0 † |

**Legend**
- **Bold** — value the vendor calls out specifically.
- *Italics* — unofficial, inferred from the model lineage (see §3.1).
- `†` — **the card is silent on this parameter.** The value shown is the neutral (no-op)
  setting. Set it explicitly anyway: llama.cpp's defaults for `top_k` (40) and `min_p`
  (0.05) are *not* neutral, and will silently truncate a distribution the vendor never
  intended to truncate. Cards written against vLLM omit these parameters safely because
  vLLM's defaults *are* neutral — that guarantee does not carry over to llama.cpp.

*Italicised values = unofficial, see §3.1.*

Reading notes:

- **"precise coding"** is verbatim from the Qwen cards: *"precise coding tasks (e.g. WebDev)"*.
  This is the only case in the table where a vendor prescribes `T < 1.0` for code — and the
  floor is still **0.6**, not 0.2.
- **GPT-OSS-120B** is the only one that explicitly forbids repetition penalties. Several
  clients enable them by default; they must be disabled.
- GPT-OSS's `top_k` must be **disabled**. Watch out: in `transformers` the default is 50 if
  left unspecified.

---

## 3. Caveats and known discrepancies

### 3.1 Llama-3.3-70B — unofficial values

Meta's model card contains **no** "recommended sampling" section. The `0.6 / 0.9` values are
the de-facto inheritance of the Llama 3 line (the snippet in the Llama-3-70B-Instruct card
and the `generation_config.json` of that same generation).

**Not verified directly against Llama-3.3-70B's `generation_config.json`.** If you have the
weights locally:

```bash
python3 -c "import json,sys; print(json.load(open(sys.argv[1])))" \
  /path/to/Llama-3.3-70B-Instruct/generation_config.json
```

### 3.2 Qwen3.6 inconsistency: 27B vs 35B-A3B

In **thinking general** mode, the 27B prescribes `presence_penalty=0.0` while the 35B-A3B
prescribes `1.5`. The divergence is still present on both cards. Unclear whether it is
intentional or a typo. Reported here:
<https://huggingface.co/Qwen/Qwen3.6-27B/discussions/10>

Historical note: Qwen3.6-35B-A3B also had an *internal* inconsistency between the
"recommended sampling parameters" section and the "Best Practices" section for
instruct/reasoning mode. The card has since been updated and the two now agree.
(<https://huggingface.co/Qwen/Qwen3.6-35B-A3B/discussions/23>)

### 3.3 Nemotron 3 Super — two third-party discrepancies

The NVIDIA card says `1.0 / 0.95` **for everything**, tool calling explicitly included. However:

- the Unsloth guide shows `--temp 1.0 --top-p 1.0` in its `llama-cli` example;
- a third-party blog claims NVIDIA recommends `0.6 / 0.95` for tool calling.

**That last claim does not appear on any official NVIDIA card.** Treat it as unverified.
→ Use `1.0 / 0.95`.

### 3.4 Gemma 4 and coding: a counter-intuitive signal (anecdotal)

Thread on `unsloth/gemma-4-26B-A4B-it-GGUF` (discussion #21): testing 26B-A4B and 31B on
coding tasks, stepping the temperature down (0.8, 0.6, 0.3) made results **worse at every
step**, while `temp 1.5` made the tests pass.

Anecdotal, not a controlled measurement — but consistent with everything above. Verify with
your own benchmarks before adopting.

### 3.5 Sampler application order

From the same thread, a less anecdotal point: **order matters**, and almost nobody specifies
it, trusting llama.cpp's default chain. Community proposal:

```
--samplers "temperature;top_p;top_k"
```

llama.cpp's default order applies `temp` **after** the truncation steps; the proposed order
applies it **before**. These are two different, non-equivalent semantics.

**Status: community, not vendor guidance.** Verifiable, but must be measured.

---

## 4. Mapping onto `llama-server` flags

| Card parameter | llama.cpp flag |
|---|---|
| `temperature` | `--temp` |
| `top_p` | `--top-p` |
| `top_k` | `--top-k` (`0` = disabled) |
| `min_p` | `--min-p` |
| `presence_penalty` | `--presence-penalty` |
| `repetition_penalty` | `--repeat-penalty` |
| sampler order | `--samplers "..."` |

### 4.1 Pitfall: silent defaults

Two llama.cpp defaults override the card unless you set them explicitly:

- **`--min-p`** (default ≠ 0)
- **`--top-k`** (default ≠ 0)

These are exactly the two that break GPT-OSS-120B's required configuration, which needs
`top_k` off and no penalties.

Defaults **change between builds**. The only reliable source is your own build:

```bash
/opt/llama.cpp/llama-server --help | grep -E 'min-p|top-k|top-p|temp|repeat-penalty|presence'
```

### 4.2 Pitfall: the override chain

CLI flags are **server-side defaults only**. Any value present in the client's request JSON
body overrides them.

```
llama-server CLI flags  ←  overridden by  ←  client's JSON body
```

Practical implications:

| Client | Where to configure |
|---|---|
| Hermes Agent | `extra_body` in `~/.hermes/config.yaml` |
| Codex CLI | `~/.codex/config.toml` |
| Claude Code | sends fixed values; CLI flags are ignored |
| Open WebUI | per-model parameters in the UI |

If results don't change when you tweak server flags, **this is almost always why**. Check
the log:

```bash
grep -iE 'temp|top_p|top_k|min_p' /tmp/llama-server.log | tail -20
```

---

## 5. Example: GPT-OSS-120B on an RPC cluster

```bash
/opt/llama.cpp/llama-server \
  -m /Volumes/Home/gguf_models/GPT-OSS-120B-MXFP4.gguf \
  -c 131072 \
  --rpc 192.168.20.1:50000,192.168.30.2:50000 \
  --device MTL0,RPC0,RPC2,RPC3 \
  --host 0.0.0.0 --port 8088 \
  --split-mode layer --metrics --jinja \
  -fa on -fit on -fitc 8192 \
  -np 1 --no-warmup \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --repeat-penalty 1.0 --presence-penalty 0.0 \
  --chat-template-kwargs '{"reasoning_effort": "high"}' \
  --seed 123456789 \
  -ctk q8_0 -ctv q8_0 \
  --alias GPT-OSS-120B-MXFP4 \
  -ub 1024 -b 2048 \
  --log-file /tmp/llama-server.log
```

Card-compliant samplers: `--top-k 0`, `--min-p 0.0`, no penalties.

---

## 6. Verification method

Trust neither the cards nor the anecdotes without measurement. Minimum protocol:

1. Tasks with an observable contract and a `--selftest` flag for objective pass/fail.
2. N ≥ 10 runs per configuration (sampling is stochastic: one run tells you nothing).
3. Vary **one** parameter at a time.
4. Record `finish_reason`: an unexpected `length` signals runaway/repetition, not a task failure.

---

## Sources

| Model | URL |
|---|---|
| Nemotron-3-Super-120B-A12B | <https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16> |
| Nemotron-3.5-Lightning-30B-A3B | <https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16> |
| Qwen3.6-27B | <https://huggingface.co/Qwen/Qwen3.6-27B> |
| Qwen3.8-27B | <https://huggingface.co/unsloth/Qwen3.8-27B> |
| Qwen3.6-35B-A3B | <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF> |
| Qwen3-Coder-30B-A3B-Instruct | <https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct> |
| GPT-OSS (official repo) | <https://github.com/openai/gpt-oss> |
| GPT-OSS on llama.cpp | <https://github.com/ggml-org/llama.cpp/discussions/15396> |
| Gemma 4 | <https://ollama.com/library/gemma4:31b> |
| Llama-3.3-70B-Instruct | <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct> |
| Claude — sampler constraints | <https://platform.claude.com/docs/en/build-with-claude/thinking> |
