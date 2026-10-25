#!/usr/bin/env python3
"""
kv-size.py - stima la memoria della KV cache di un modello (come in llama.cpp,
`llama-server -c N`) leggendo il config.json da Hugging Face o da una cartella locale.

Uso:
  python3 kv-size.py <fornitore>/<modello> [altri ...]        -> GiB per 128k token
  python3 kv-size.py qwen/qwen3.8-27b --kv-size 16k           -> GiB per 16384 token
  python3 kv-size.py qwen/qwen3.8-27b -c 32k -ctk q8_0 -ctv q8_0
  python3 kv-size.py qwen/qwen3.8-27b --verbose               -> parametri letti + calcolo
  python3 kv-size.py /percorso/cartella_modello               (cartella con config.json)

Modelli gated o privati: --token, oppure variabile HF_TOKEN, oppure il token
salvato da `hf auth login`. Se huggingface.co non e' raggiungibile usa la copia
nella cache locale di Hugging Face (~/.cache/huggingface/hub), se esiste.
Solo libreria standard, Python >= 3.8.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
TIMEOUT_S = 20
GIB = 2 ** 30
MIB = 2 ** 20

# byte per elemento dei tipi di cache di llama.cpp (blocchi da 32 valori + scale fp16)
CACHE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32,
               "q5_0": 22 / 32, "q4_1": 20 / 32, "q4_0": 18 / 32, "iq4_nl": 18 / 32}
STATE_BYTES = 4  # stati ricorrenti (layer lineari/SSM) tenuti in f32

# Nomi alternativi della stessa grandezza nei vari formati di config
ALIASES = {
    "n_layers": ("num_hidden_layers", "n_layer", "n_layers", "num_layers"),
    "n_heads": ("num_attention_heads", "n_head", "n_heads", "num_heads"),
    "n_kv": ("num_key_value_heads", "num_kv_heads", "n_kv_heads", "n_head_kv",
             "multi_query_group_num", "num_query_groups"),
    "hidden": ("hidden_size", "n_embd", "d_model", "dim"),
    "head_dim": ("head_dim", "kv_channels", "attention_head_dim"),
    "max_ctx": ("max_position_embeddings", "max_sequence_length", "seq_length",
                "n_positions", "max_seq_len"),
}
# Sotto-configurazioni dei modelli multimodali che contengono la parte LLM
NESTED = ("text_config", "llm_config", "language_config", "language_model",
          "text_model", "thinker_config")
# Chiavi interpretate dallo script (non vanno elencate tra quelle "da controllare")
USED = {k for v in ALIASES.values() for k in v} | {
    "layer_types", "layers_block_type", "hybrid_override_pattern", "attn_type_list",
    "full_attn_idxs", "attn_layer_period", "attn_layer_offset", "full_attention_interval",
    "sliding_window", "use_sliding_window", "sliding_window_pattern",
    "_sliding_window_pattern", "max_window_layers", "attention_chunk_size",
    "no_rope_layers", "no_rope_layer_interval", "num_kv_shared_layers", "kv_lora_rank",
    "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim", "multi_query", "model_type",
    "num_nextn_predict_layers",
}
RELEVANT_RE = re.compile(
    r"kv|key_value|head|window|attn|attention|layer_type|shared|chunk|lora|global|"
    r"local|linear|mamba|ssm|conv|state|nextn|mtp|index|swa|cache", re.I)
IRRELEVANT_RE = re.compile(
    r"dropout|bias|softcap|scal|multiplier|pdrop|implementation|initializer|eps$|norm|"
    r"_act$|activation|token|vocab|rope|temperature|dtype|q_lora_rank|expert|router|moe",
    re.I)

OTHER_LABEL = {
    "recurrent": "lineari/SSM: stato fisso, non dipende da -c (cresce con -np)",
    "none": "senza attenzione (solo MLP/MoE)",
    "shared": "riusano la KV di layer precedenti",
    "unknown": "tipo non riconosciuto: controlla il config",
}


class ConfigError(Exception):
    """Errore con messaggio leggibile per l'utente."""


# ----------------------------------------------------------------- lettura

def first(cfg, keys):
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            return k, v
    return None, None


def hf_home():
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface").expanduser()


def get_token(cli_token):
    if cli_token:
        return cli_token.strip()
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    path = Path(os.environ.get("HF_TOKEN_PATH") or hf_home() / "token").expanduser()
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def read_json_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigError(f"impossibile leggere {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{path} non contiene un oggetto JSON")
    return data


def fetch_json(repo, revision, fname, token):
    rev = urllib.parse.quote(revision, safe="")
    url = f"{ENDPOINT}/{repo}/resolve/{rev}/{fname}"
    req = urllib.request.Request(url, headers={"User-Agent": "kv-size/1.0"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        raw = resp.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigError(f"{url} non e' JSON valido: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"{url} non contiene un oggetto JSON")
    return data, url


def from_cache(repo, revision, fname):
    cache = Path(os.environ.get("HF_HUB_CACHE") or hf_home() / "hub").expanduser()
    base = cache / ("models--" + repo.replace("/", "--"))
    snaps = base / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [snaps / revision / fname]
    ref = base / "refs" / revision
    if ref.is_file():
        try:
            candidates.insert(0, snaps / ref.read_text().strip() / fname)
        except OSError:
            pass
    for c in candidates:
        if c.is_file():
            return c
    others = sorted((c for c in snaps.glob(f"*/{fname}") if c.is_file()),
                    key=lambda c: c.stat().st_mtime, reverse=True)
    return others[0] if others else None


def load_config(spec, revision, token):
    """Ritorna (config, descrizione_sorgente)."""
    p = Path(spec).expanduser()
    if p.exists():
        if p.is_dir():
            for fname in ("config.json", "params.json"):
                if (p / fname).is_file():
                    return read_json_file(p / fname), str(p / fname)
            raise ConfigError(f"nessun config.json/params.json in {p}")
        return read_json_file(p), str(p)

    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", spec):
        raise ConfigError(f"'{spec}' non e' un percorso esistente ne' un id <fornitore>/<modello>")

    net_err = None
    for fname in ("config.json", "params.json"):   # params.json: formato nativo Mistral
        try:
            return fetch_json(spec, revision, fname, token)
        except urllib.error.HTTPError as e:
            code = (e.headers.get("X-Error-Code") or "") if e.headers else ""
            if code == "RevisionNotFound":
                raise ConfigError(f"revisione '{revision}' inesistente in {spec}") from e
            if code == "RepoNotFound":
                raise ConfigError(f"repo '{spec}' inesistente, oppure privato e senza token valido") from e
            if code == "GatedRepo" or e.code in (401, 403):
                hint = ("token presente ma senza accesso: accetta la licenza sulla pagina del modello"
                        if token else "serve un token: --token, HF_TOKEN o `hf auth login`")
                raise ConfigError(f"accesso negato a '{spec}' (HTTP {e.code}"
                                  f"{', ' + code if code else ''}); {hint}") from e
            if e.code == 404:
                continue
            net_err = f"HTTP {e.code}"
            break
        except OSError as e:   # URLError, timeout, DNS, connessione rifiutata
            net_err = getattr(e, "reason", None) or e
            break

    if net_err is not None:
        for fname in ("config.json", "params.json"):
            cached = from_cache(spec, revision, fname)
            if cached:
                return read_json_file(cached), f"{cached} (cache locale; rete: {net_err})"
        raise ConfigError(f"{ENDPOINT} non raggiungibile ({net_err}) e nessuna copia in cache locale")
    raise ConfigError("config.json/params.json assenti nel repo: forse e' un repo solo-GGUF; "
                      "usa l'id del modello base, oppure leggi i metadati del .gguf con gguf-dump")


# ----------------------------------------------------------------- analisi del config

def pick_text_config(cfg):
    """Scende nelle sotto-config multimodali fino alla parte LLM."""
    cur, path = cfg, []
    for _ in range(4):
        nxt = None
        for k in NESTED:
            sub = cur.get(k)
            if isinstance(sub, dict) and (first(sub, ALIASES["n_layers"])[1] is not None
                                          or any(isinstance(sub.get(n), dict) for n in NESTED)):
                nxt = k
                break
        if nxt is None:
            break
        cur = cur[nxt]
        path.append(nxt)
    return cur, path


def norm_kind(s):
    s = str(s).lower()
    if any(t in s for t in ("linear", "mamba", "conv", "ssm", "recurrent", "delta",
                            "lightning", "rwkv")):
        return "recurrent"
    if "chunk" in s:
        return "chunked"
    if any(t in s for t in ("sliding", "local", "swa")):
        return "sliding"
    if any(t in s for t in ("full", "global", "attention", "attn", "hybrid")):
        return "full"
    if any(t in s for t in ("mlp", "moe", "ffn")):
        return "none"
    return "unknown"


def layer_kinds(tc, L, model_type, notes, warns):
    """Tipo di ogni layer: full / sliding / chunked / recurrent / none / unknown."""
    def mismatch(key, n):
        warns.append(f"'{key}' ha {n} elementi ma i layer sono {L}: ignorato")

    for key in ("layer_types", "layers_block_type"):
        lt = tc.get(key)
        if isinstance(lt, list) and lt:
            if len(lt) == L:
                return [norm_kind(x) for x in lt], key
            mismatch(key, len(lt))

    pat = tc.get("hybrid_override_pattern")          # Nemotron-H
    if isinstance(pat, str) and pat:
        if len(pat) == L:
            m = {"*": "full", "M": "recurrent", "-": "none", "E": "none"}
            return [m.get(c, "unknown") for c in pat], "hybrid_override_pattern"
        mismatch("hybrid_override_pattern", len(pat))

    atl = tc.get("attn_type_list")                   # MiniMax
    if isinstance(atl, list) and atl:
        if len(atl) == L:
            return ["full" if x else "recurrent" for x in atl], "attn_type_list"
        mismatch("attn_type_list", len(atl))

    idx = tc.get("full_attn_idxs")                   # LFM2
    if isinstance(idx, list) and idx:
        s = set(idx)
        return ["full" if i in s else "recurrent" for i in range(L)], "full_attn_idxs"

    per, off = tc.get("attn_layer_period"), tc.get("attn_layer_offset")   # Jamba
    if isinstance(per, int) and per > 0 and isinstance(off, int):
        return (["full" if i % per == off else "recurrent" for i in range(L)],
                "attn_layer_period/offset")

    fai = tc.get("full_attention_interval")          # Qwen3-Next e successori
    if isinstance(fai, int) and fai > 0:
        return (["full" if (i + 1) % fai == 0 else "recurrent" for i in range(L)],
                f"full_attention_interval={fai}")

    if tc.get("attention_chunk_size"):               # Llama 4
        nrl = tc.get("no_rope_layers")
        if isinstance(nrl, list) and len(nrl) == L:
            return ["chunked" if x else "full" for x in nrl], "no_rope_layers + attention_chunk_size"
        iv = tc.get("no_rope_layer_interval") or 4
        warns.append(f"chunked attention: assunto 1 layer globale ogni {iv}")
        return (["full" if (i + 1) % iv == 0 else "chunked" for i in range(L)],
                f"attention_chunk_size, intervallo {iv}")

    sw, use_sw = tc.get("sliding_window"), tc.get("use_sliding_window")
    if sw and use_sw is False:
        notes.append(f"sliding_window={sw} ignorata: use_sliding_window=false")
    elif sw:
        p = tc.get("sliding_window_pattern") or tc.get("_sliding_window_pattern")
        if not p and str(model_type or "").startswith("gemma2"):
            p = 2
        if isinstance(p, int) and p > 0:
            return (["full" if (i + 1) % p == 0 else "sliding" for i in range(L)],
                    f"sliding_window_pattern={p}")
        mwl = tc.get("max_window_layers")
        if use_sw is True and isinstance(mwl, int):
            return (["sliding" if i >= mwl else "full" for i in range(L)],
                    "use_sliding_window + max_window_layers")
        warns.append("sliding_window senza schema per-layer: assunta su tutti i layer "
                     "(il motore puo' ignorarla: verifica n_swa nel log di llama.cpp)")
        return ["sliding"] * L, "sliding_window su tutti i layer"
    return ["full"] * L, "tutti full attention (nessuno schema ibrido/SWA nel config)"


def all_int(*xs):
    return all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in xs)


def recurrent_state(tc):
    """Elementi dello stato fisso per layer lineare/SSM, se il tipo e' riconosciuto."""
    def g(*keys):
        return first(tc, keys)

    # Gated DeltaNet (Qwen3-Next e successori): stato n_v_heads x d_k x d_v + buffer conv
    keys = ("linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
            "linear_value_head_dim", "linear_conv_kernel_dim")
    nvh, nkh, dk, dv, kc = (tc.get(k) for k in keys)
    if all_int(nvh, nkh, dk, dv, kc):
        s = nvh * dk * dv
        conv = (kc - 1) * (2 * nkh * dk + nvh * dv)
        return {"type": "Gated DeltaNet", "elements_per_layer": s + conv,
                "bytes_per_layer": STATE_BYTES * (s + conv)}, set(keys)

    # Mamba2 (Nemotron-H, Granite 4, ...): stato d_inner x d_state + buffer conv
    k_nh, nh = g("mamba_num_heads", "mamba_n_heads")
    k_hd, hd = g("mamba_head_dim", "mamba_d_head")
    k_ds, ds = g("ssm_state_size", "mamba_d_state", "state_size")
    k_kc, kc = g("conv_kernel", "mamba_d_conv")
    k_ng, ng = g("n_groups", "mamba_n_groups")
    if all_int(nh, hd, ds, kc):
        ng = ng if all_int(ng) else 1
        d_inner = nh * hd
        s = d_inner * ds
        conv = (kc - 1) * (d_inner + 2 * ng * ds)
        used = {k for k in (k_nh, k_hd, k_ds, k_kc, k_ng) if k}
        return {"type": "Mamba2", "elements_per_layer": s + conv,
                "bytes_per_layer": STATE_BYTES * (s + conv)}, used
    return None, set()


def analyze(cfg):
    tc, path = pick_text_config(cfg)
    notes, warns = [], []
    model_type = tc.get("model_type") or cfg.get("model_type")

    _, L = first(tc, ALIASES["n_layers"])
    if not all_int(L):
        raise ConfigError("numero di layer non trovato (num_hidden_layers o equivalenti)")
    _, n_heads = first(tc, ALIASES["n_heads"])
    k_max, max_ctx = first(tc, ALIASES["max_ctx"])

    k_kv, n_kv = first(tc, ALIASES["n_kv"])
    if n_kv is None:
        if tc.get("multi_query"):
            n_kv, k_kv = 1, "multi_query"
        elif n_heads is not None:
            n_kv, k_kv = n_heads, "= teste Q: MHA, num_key_value_heads assente"
            notes.append("num_key_value_heads assente: assunta MHA (n_kv = num_attention_heads)")
    if isinstance(n_kv, list):
        if len(n_kv) != L:
            raise ConfigError(f"{k_kv} e' una lista di {len(n_kv)} elementi ma i layer sono {L}")
        kv_per_layer = list(n_kv)
    else:
        kv_per_layer = [n_kv] * L

    _, hidden = first(tc, ALIASES["hidden"])
    k_hd, head_dim = first(tc, ALIASES["head_dim"])
    if head_dim is None and all_int(hidden, n_heads):
        head_dim, k_hd = hidden // n_heads, f"hidden_size/num_attention_heads = {hidden}/{n_heads}"
        if hidden % n_heads:
            warns.append("hidden_size non divisibile per num_attention_heads: head_dim incerto")

    mla = None
    if tc.get("kv_lora_rank"):
        rank, rope = tc["kv_lora_rank"], tc.get("qk_rope_head_dim")
        if not all_int(rope):
            warns.append("MLA: qk_rope_head_dim assente, assunto 0")
            rope = 0
        mla = {"kv_lora_rank": rank, "qk_rope_head_dim": rope,
               "qk_nope_head_dim": tc.get("qk_nope_head_dim"),
               "v_head_dim": tc.get("v_head_dim"), "latent": rank + rope}
        notes.append("MLA: llama.cpp salva in K il latente (kv_lora_rank + qk_rope_head_dim); "
                     "la V occupa altri kv_lora_rank elementi oppure e' una vista della K, "
                     "secondo versione e -fa: il totale riporta entrambi i casi")
        head_dim_v = mla["v_head_dim"]
    else:
        head_dim_v = tc.get("v_head_dim") or head_dim

    kinds, scheme = layer_kinds(tc, L, model_type, notes, warns)
    shared = tc.get("num_kv_shared_layers") or 0
    if all_int(shared) and shared < L:
        kinds[L - shared:] = ["shared"] * shared

    windows = {"full": None, "sliding": tc.get("sliding_window"),
               "chunked": tc.get("attention_chunk_size")}
    groups, others, missing_win = {}, {}, set()
    for i, kind in enumerate(kinds):
        if kind not in windows:
            others[kind] = others.get(kind, 0) + 1
            continue
        win = windows[kind]
        if kind != "full" and not win:
            missing_win.add(kind)
        if mla:
            key = (kind, "mla", win)
            grp = groups.setdefault(key, {
                "kind": kind, "attention": "MLA", "layers": 0, "window": win,
                "kv_lora_rank": mla["kv_lora_rank"], "qk_rope_head_dim": mla["qk_rope_head_dim"],
                "elements_per_token_per_layer": mla["latent"]})
        else:
            nk = kv_per_layer[i]
            ok = all_int(nk, head_dim, head_dim_v)
            key = (kind, nk, head_dim, head_dim_v, win)
            grp = groups.setdefault(key, {
                "kind": kind, "attention": "GQA/MHA", "layers": 0, "window": win,
                "n_kv": nk, "d_k": head_dim, "d_v": head_dim_v,
                "elements_per_token_per_layer": nk * (head_dim + head_dim_v) if ok else None})
        grp["layers"] += 1
    for kind in sorted(missing_win):
        warns.append(f"layer '{kind}' senza dimensione della finestra nel config: "
                     f"conteggiati come full")
    if any(gr["elements_per_token_per_layer"] is None for gr in groups.values()):
        warns.append("n_kv o head_dim mancanti: elementi per token non calcolabili")
    if others.get("unknown"):
        warns.append(f"{others['unknown']} layer di tipo non riconosciuto, esclusi dal calcolo")
    nextn = tc.get("num_nextn_predict_layers")
    if nextn:
        notes.append(f"{nextn} layer MTP (num_nextn_predict_layers) non inclusi nel conteggio")

    rec, rec_keys = (None, set())
    if others.get("recurrent"):
        rec, rec_keys = recurrent_state(tc)
        if rec is None:
            warns.append(f"stato fisso dei {others['recurrent']} layer lineari/SSM non stimato "
                         f"(tipo non riconosciuto): escluso dal totale")

    related = {k: v for k, v in tc.items()
               if k not in USED and k not in rec_keys and k not in NESTED
               and not k.endswith("_config")
               and RELEVANT_RE.search(k) and not IRRELEVANT_RE.search(k)}

    return {
        "model_type": model_type,
        "architectures": cfg.get("architectures"),
        "subconfig": ".".join(path) or None,
        "max_context": max_ctx, "max_context_key": k_max,
        "n_layers": L,
        "n_heads": n_heads,
        "n_kv_heads": n_kv, "n_kv_heads_key": k_kv,
        "head_dim_k": head_dim, "head_dim_v": head_dim_v, "head_dim_key": k_hd,
        "sliding_window": tc.get("sliding_window"),
        "attention_chunk_size": tc.get("attention_chunk_size"),
        "mla": mla,
        "layer_scheme": scheme,
        "kv_groups": list(groups.values()),
        "layers_without_ctx_kv": others,
        "recurrent_state": rec,
        "notes": notes,
        "warnings": warns,
        "related_keys": related,
    }


# ----------------------------------------------------------------- calcolo

def pad256(x):
    """llama.cpp alloca le celle della KV a multipli di 256."""
    return -(-x // 256) * 256


def estimate(r, kv_size, type_k, type_v, ubatch):
    bk, bv = CACHE_BYTES[type_k], CACHE_BYTES[type_v]
    n_ctx = pad256(kv_size)
    rows, total, total_view = [], 0.0, 0.0
    for g in r["kv_groups"]:
        win = g["window"]
        cells = n_ctx if (g["kind"] == "full" or not win) else min(n_ctx, pad256(win + ubatch))
        if g["attention"] == "MLA":
            k_el, v_el = g["kv_lora_rank"] + g["qk_rope_head_dim"], g["kv_lora_rank"]
        else:
            if g["elements_per_token_per_layer"] is None:
                raise ConfigError("n_kv o head_dim mancanti nel config: KV non calcolabile "
                                  "(usa --verbose per i dettagli)")
            k_el, v_el = g["n_kv"] * g["d_k"], g["n_kv"] * g["d_v"]
        b = g["layers"] * cells * (k_el * bk + v_el * bv)
        b_view = g["layers"] * cells * k_el * bk if g["attention"] == "MLA" else b
        rows.append({"kind": g["kind"], "attention": g["attention"], "layers": g["layers"],
                     "cells": cells, "k_elements": k_el, "v_elements": v_el, "bytes": b})
        total += b
        total_view += b_view

    n_rec = r["layers_without_ctx_kv"].get("recurrent", 0)
    rec_bytes = None
    if n_rec and r["recurrent_state"]:
        rec_bytes = n_rec * r["recurrent_state"]["bytes_per_layer"]
        total += rec_bytes
        total_view += rec_bytes
    return {
        "kv_size": kv_size, "cells": n_ctx, "type_k": type_k, "type_v": type_v,
        "ubatch": ubatch, "groups": rows,
        "recurrent_layers": n_rec, "recurrent_bytes": rec_bytes,
        "total_bytes": total,
        "total_bytes_if_v_view_of_k": total_view if r["mla"] else None,
        "total_gib": total / GIB,
    }


# ----------------------------------------------------------------- output

def human(b):
    return f"{b / GIB:.2f} GiB" if b >= GIB else f"{b / MIB:.1f} MiB"


def fmt_val(v):
    if isinstance(v, list):
        return f"[lista di {len(v)}]" if len(v) > 8 else json.dumps(v)
    if isinstance(v, dict):
        keys = list(v)
        return "{" + ", ".join(keys[:6]) + (", ..." if len(keys) > 6 else "") + "}"
    return json.dumps(v)


def short_line(name, r, e):
    line = f"{name}: {e['total_bytes'] / GIB:.2f} GiB"
    extra = []
    if e["total_bytes_if_v_view_of_k"] is not None:
        extra.append(f"{e['total_bytes_if_v_view_of_k'] / GIB:.2f} GiB se la V e' una vista della K")
    if r["warnings"]:
        extra.append("stima incerta, vedi --verbose")
    return line + (f"  ({'; '.join(extra)})" if extra else "")


def print_verbose(name, src, r, e):
    out = [f"=== {name} ==="]
    add = out.append
    sub = f"  [sotto-config: {r['subconfig']}]" if r["subconfig"] else ""
    add(f"sorgente      : {src}{sub}")
    arch = ", ".join(r["architectures"] or [])
    add(f"model_type    : {r['model_type'] or 'n.d.'}" + (f"  ({arch})" if arch else ""))
    if r["max_context"]:
        add(f"contesto max  : {r['max_context']}  ({r['max_context_key']})")
    else:
        add("contesto max  : n.d.")
    add(f"layer         : {r['n_layers']}")
    kv = r["n_kv_heads"]
    kv_s = f"per-layer {min(kv)}..{max(kv)}" if isinstance(kv, list) else str(kv)
    add(f"teste Q / KV  : {r['n_heads']} / {kv_s}  ({r['n_kv_heads_key']})")
    if r["mla"]:
        m = r["mla"]
        add(f"MLA           : kv_lora_rank={m['kv_lora_rank']}  qk_rope_head_dim="
            f"{m['qk_rope_head_dim']}  -> latente {m['latent']} elem/token/layer")
    else:
        add(f"head_dim K/V  : {r['head_dim_k']} / {r['head_dim_v']}  ({r['head_dim_key']})")
    add(f"schema layer  : {r['layer_scheme']}")
    add("gruppi di layer con KV:")
    if not r["kv_groups"]:
        add("  (nessuno)")
    for g in r["kv_groups"]:
        desc = "MLA" if g["attention"] == "MLA" else f"n_kv={g['n_kv']}  d_k={g['d_k']}  d_v={g['d_v']}"
        win = f"  finestra={g['window']}" if g["window"] else ""
        el = g["elements_per_token_per_layer"]
        el_s = f"{el} elem/token/layer" if el is not None else "elem/token n.d."
        add(f"  {g['kind']:<8}{g['layers']:>4} layer  {desc}{win}  ->  {el_s}")
    if r["layers_without_ctx_kv"]:
        add("layer senza KV che cresce con -c:")
        for kind, n in r["layers_without_ctx_kv"].items():
            add(f"  {n:>4}  {OTHER_LABEL.get(kind, kind)}")
        rs = r["recurrent_state"]
        if rs:
            add(f"        stato {rs['type']}: {rs['elements_per_layer']} elem/layer in f32 "
                f"= {human(rs['bytes_per_layer'])}/layer")
    if r["warnings"] or r["notes"]:
        add("note:")
        out.extend(f"  [!] {w}" for w in r["warnings"])
        out.extend(f"  - {n}" for n in r["notes"])
    if r["related_keys"]:
        add("chiavi correlate non interpretate (controllale a mano):")
        out.extend(f"  {k} = {fmt_val(v)}" for k, v in r["related_keys"].items())

    bk, bv = CACHE_BYTES[e["type_k"]], CACHE_BYTES[e["type_v"]]
    add("")
    add(f"calcolo per --kv-size {e['kv_size']} (celle allocate {e['cells']}), "
        f"cache K={e['type_k']} V={e['type_v']}, -ub {e['ubatch']}:")
    for row in e["groups"]:
        add(f"  {row['kind']:<8}{row['layers']:>4} layer x {row['cells']:>7} token x "
            f"(K {row['k_elements']} x {bk:g} B + V {row['v_elements']} x {bv:g} B)"
            f"  = {human(row['bytes'])}")
    if e["recurrent_bytes"] is not None:
        add(f"  stato fisso {e['recurrent_layers']:>4} layer lineari/SSM (f32)"
            f"  = {human(e['recurrent_bytes'])}")
    add(f"  TOTALE = {human(e['total_bytes'])}")
    if e["total_bytes_if_v_view_of_k"] is not None:
        add(f"  TOTALE se la V e' una vista della K = {human(e['total_bytes_if_v_view_of_k'])}")
    add("  non inclusi: pesi del modello e compute buffer")
    print("\n".join(out) + "\n")


def parse_tokens(s):
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*", s)
    if not m:
        raise argparse.ArgumentTypeError(f"'{s}' non valido: usa un intero o un suffisso k/M (es. 16k)")
    n = float(m.group(1)) * {"": 1, "k": 1024, "m": 1024 ** 2}[m.group(2).lower()]
    if n < 1 or n != int(n):
        raise argparse.ArgumentTypeError(f"'{s}' non e' un numero intero di token positivo")
    return int(n)


def positive_int(s):
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{s}' non e' un intero") from None
    if v < 1:
        raise argparse.ArgumentTypeError("deve essere >= 1")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stima la RAM/VRAM della KV cache (llama.cpp) dal config.json di un modello.")
    ap.add_argument("models", nargs="+", metavar="MODELLO",
                    help="id <fornitore>/<modello>, cartella locale o file config.json")
    ap.add_argument("-c", "--kv-size", type=parse_tokens, default=128 * 1024,
                    help="token di contesto, come -c di llama-server; accetta 16k, 1M "
                         "(default: 128k = 131072)")
    ap.add_argument("-ctk", "--cache-type-k", default="f16", choices=sorted(CACHE_BYTES),
                    help="tipo della cache K (default: f16)")
    ap.add_argument("-ctv", "--cache-type-v", default="f16", choices=sorted(CACHE_BYTES),
                    help="tipo della cache V (default: f16)")
    ap.add_argument("-ub", "--ubatch", type=positive_int, default=512,
                    help="-ub di llama.cpp, incide solo sui layer sliding (default: 512)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="mostra i parametri letti dal config e il calcolo")
    ap.add_argument("--json", action="store_true", help="output JSON completo")
    ap.add_argument("--revision", default="main", help="branch, tag o commit (default: main)")
    ap.add_argument("--token", help="token Hugging Face (default: HF_TOKEN o token salvato)")
    args = ap.parse_args(argv)

    token = get_token(args.token)
    results, failed = [], 0
    for spec in args.models:
        try:
            cfg, src = load_config(spec, args.revision, token)
            try:
                r = analyze(cfg)
                e = estimate(r, args.kv_size, args.cache_type_k, args.cache_type_v, args.ubatch)
            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                raise ConfigError(f"struttura del config non prevista "
                                  f"({type(exc).__name__}: {exc})") from exc
        except ConfigError as exc:
            failed += 1
            if args.json:
                results.append({"model": spec, "error": str(exc)})
            else:
                print(f"{spec}: ERRORE: {exc}", file=sys.stderr)
            continue

        if args.json:
            results.append({"model": spec, "source": src, **r, "estimate": e})
            continue
        if isinstance(r["max_context"], int) and args.kv_size > r["max_context"]:
            print(f"{spec}: attenzione, --kv-size {args.kv_size} supera il contesto massimo "
                  f"del modello ({r['max_context']})", file=sys.stderr)
        if args.verbose:
            print_verbose(spec, src, r, e)
        else:
            print(short_line(spec, r, e))

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
