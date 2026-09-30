"""Reference implementation (v4, job D: changes from v1 = bos_id(), identical for Qwen3; span_keys(); an 8th row field first_sub; tokenizer_checks()) for the tree-recovery runner jobs (T4 frame-only re-scoring, T1 endpoint distances).

Conventions reproduce the 1b span-cost cache exactly:
  text(words) = nltk TreebankWordDetokenizer().detokenize(words with '\\/' -> '/' and '\\*' -> '*')
  a substituted sentence = text(words[:i] + [filler] + words[j+1:]); '<del>' inserts nothing; when the span starts
  the sentence (i = 0) its first character is upper-cased (sub_text)
  ids = tokenizer(text, add_special_tokens=False); the model reads [BOS] + ids, BOS = <|endoftext|>
  every text token is scored by its log-probability; log-softmax in float32 over the full vocabulary.
New in T4: the sentence's end mark (end_string: '.', '?' or '!') is tokenised ON ITS OWN and its ids are appended,
so the sentence's own tokens never change and a span that ends the sentence still has a frame token after it.
The frame of a substitution is defined on tokens: the suffix shared with the original, capped at the tokens that lie
wholly after the span, plus the end mark. The shared prefix (capped at the tokens wholly before the span) is
recorded only to check causality: it scores identically in both sentences, so it cannot enter a cost.
"""
import re
import numpy as np, torch
from nltk.tokenize.treebank import TreebankWordDetokenizer

FILLERS = ["it", "there", "did", "then", "do so", "does so", "did so", "done so", "doing so",
           "is", "was", "be", "been", "happens", "happened", "blorp", "<del>"]
_D = TreebankWordDetokenizer()


def unesc(w): return w.replace("\\/", "/").replace("\\*", "*")


def text_of(words): return _D.detokenize([unesc(w) for w in words])


def end_string(tree):
    """The sentence's end mark: its last leaf, skipping closing quotes and brackets, if tagged '.' and one of
    '.', '?', '!'; otherwise '.'."""
    for tag, w in reversed(re.findall(r"\(([^\s()]+) ([^\s()]+)\)", tree)):
        if tag in ("''", "-RRB-", "-NONE-"): continue
        return w if tag == "." and w in (".", "?", "!") else "."
    return "."


def load(model_dir, revision=None, dtype=torch.bfloat16, device="cpu"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_dir, revision=revision)
    mdl = AutoModelForCausalLM.from_pretrained(model_dir, revision=revision, dtype=dtype).to(device).eval()
    return tok, mdl, bos_id(tok)


def bos_id(tok):
    """v2 (job D): the start token the model reads first. The tokenizer's own BOS if it defines one (Llama-3:
    <|begin_of_text|>, GPT-2: <|endoftext|>); otherwise <|endoftext|> (Qwen3 base defines no BOS, and v1 used this id)."""
    b = tok.bos_token_id
    if b is None: b = tok.convert_tokens_to_ids("<|endoftext|>")
    assert isinstance(b, int) and 0 <= b < len(tok) and b in tok.all_special_ids, ("no usable start token", tok.bos_token, b)
    return b


def sub_text(words, i, j, filler):
    """The substituted sentence exactly as the 1b cache scored it."""
    t = text_of(words[:i] + ([] if filler == "<del>" else [filler]) + words[j + 1:])
    return t[:1].upper() + t[1:] if i == 0 else t


def word_chars(words, text):
    """Character span of every (unescaped) word in the detokenised text; asserts only spaces are skipped."""
    out, pos = [], 0
    for w in words:
        u = unesc(w); k = text.index(u, pos); assert text[pos:k].strip() == "", (w, text[pos:k]); out.append((k, k + len(u))); pos = k + len(u)
    return out


def encode(tok, text):
    e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(e.input_ids), [tuple(o) for o in e.offset_mapping]


@torch.no_grad()
def forward(mdl, bos, id_lists, device="cpu", want_states=False):
    """One batched, right-padded pass over [BOS] + ids. Per sequence: log-prob of every token (float32) and,
    if asked, the log-softmax state after every position (row 0 = after BOS, row t+1 = after token t)."""
    L = max(len(x) for x in id_lists) + 1
    inp = torch.zeros(len(id_lists), L, dtype=torch.long); att = torch.zeros_like(inp)
    for r, x in enumerate(id_lists):
        inp[r, 0] = bos; inp[r, 1:len(x) + 1] = torch.tensor(x); att[r, :len(x) + 1] = 1
    ls = torch.log_softmax(mdl(input_ids=inp.to(device), attention_mask=att.to(device)).logits.float(), -1)
    out = []
    for r, x in enumerate(id_lists):
        tl = ls[r, torch.arange(len(x)), torch.tensor(x)].cpu().numpy()
        out.append((tl, ls[r, :len(x) + 1].cpu().numpy() if want_states else None))
    return out


def _common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]: n += 1
    return n


def t4_rows(tok, mdl, bos, words, span_keys, end, fillers=FILLERS, batch=64, device="cpu"):
    """T4 for one sentence. Returns (orig, rows):
      orig = {"text", "ids", "lp" (per-token log-probs of the text tokens), "end_ids", "end_lp" (per end-mark token)}
      rows[filler]["i,j"] = [total, n_tok, n_pre, n_suf, suf_sub, end_sub, pre_check, first_sub]
        total, n_tok  the substituted sentence's summed log-prob and token count WITHOUT the end mark (= the 1b cache)
        n_pre, n_suf  frame tokens before / after the substitution (see module docstring)
        suf_sub       summed log-prob of the n_suf suffix tokens in the substituted sentence
        end_sub       summed log-prob of the end-mark tokens after the substituted sentence
        pre_check     sum over the n_pre prefix tokens of (log-prob substituted - log-prob original); ~0 by causality
        first_sub     (v3) log-prob of the FIRST frame token after the span in the substituted sentence: the first suffix
                      token if n_suf > 0, else the first end-mark token. Its original counterpart is orig.lp[-n_suf] or
                      orig.end_lp[0]. Lets the analysis drop the one frame token whose leading space also encodes
                      where the replacement word ends (a word-boundary check).
    Frame-only cost of a substitution (computed at analysis, not here):
      (sum(orig.lp[-n_suf:]) - suf_sub + sum(orig.end_lp) - end_sub) / (n_suf + len(end_ids))
    i.e. the mean log-prob drop over the tokens AFTER the span plus the end mark; the prefix is left out of the
    mean because a causal model scores it identically in both sentences (pre_check)."""
    text = text_of(words); ids0, off0 = encode(tok, text); end_ids = tok(end, add_special_tokens=False).input_ids
    (lp0e, _), = forward(mdl, bos, [ids0 + end_ids], device); lp0 = lp0e[:len(ids0)]
    wc = word_chars(words, text)
    orig = dict(text=text, ids=ids0, lp=[float(v) for v in lp0], end_ids=end_ids, end_lp=[float(v) for v in lp0e[len(ids0):]])
    jobs = []
    for f in fillers:
        for key in span_keys:
            i, j = map(int, key.split(","))
            ids, _ = encode(tok, sub_text(words, i, j, f))
            cap_pre = sum(1 for (a, b) in off0 if b <= wc[i][0])                 # original tokens wholly before the span
            cap_suf = sum(1 for (a, b) in off0 if a >= wc[j][1])                 # original tokens wholly after the span
            n_pre = min(_common_prefix(ids0, ids), cap_pre)
            n_suf = min(_common_prefix(ids0[::-1], ids[::-1]), cap_suf)
            if n_pre + n_suf > len(ids): n_suf = len(ids) - n_pre              # never let the two overlap in a short substitution
            jobs.append((f, key, ids, n_pre, n_suf))
    rows = {f: {} for f in fillers}
    for k in range(0, len(jobs), batch):
        chunk = jobs[k:k + batch]
        for (f, key, ids, n_pre, n_suf), (lpe, _) in zip(chunk, forward(mdl, bos, [c[2] + end_ids for c in chunk], device)):
            lp = lpe[:len(ids)]
            rows[f][key] = [float(lp.sum()), len(ids), n_pre, n_suf, float(lp[len(ids) - n_suf:].sum()) if n_suf else 0.0,
                            float(lpe[len(ids):].sum()), float((lp[:n_pre] - lp0[:n_pre]).sum()), float(lpe[len(ids) - n_suf])]
    return orig, rows


def tokenizer_checks(tok, sentences, fillers=FILLERS):
    """v4 (job D): run once per tokenizer BEFORE the full run; model not needed. Deterministic, so the counts must equal
    the expected counts in the runner note exactly. sentences = rows of sentences_3161.jsonl (dicts with words, text, end).
      is_fast             offsets are required by t4_rows (frame boundaries)
      special_in_text     tokens of any text that are special tokens (must be 0)
      roundtrip_fail      texts whose decode(encode(text)) != text, originals and substituted (must be 0)
      offsets_bad         texts whose offsets are not increasing and inside the text (must be 0)
      frame_pre_short     substitutions where the shared prefix is shorter than the tokens wholly before the span
      frame_suf_short     same for the suffix: a token merged across the span's right edge (report; expected small)
      end_merge           originals where tokenizing text + end differs from ids + end_ids (report; t4_rows always
                          appends end_ids separately, as job A did)"""
    c = dict(is_fast=bool(tok.is_fast), n_sentences=0, n_texts=0, special_in_text=0, roundtrip_fail=0, offsets_bad=0,
             frame_pre_short=0, frame_suf_short=0, end_merge=0, n_substitutions=0)
    sp = set(tok.all_special_ids)
    def one(text):
        ids, off = encode(tok, text); c["n_texts"] += 1
        c["special_in_text"] += sum(1 for t in ids if t in sp)
        c["roundtrip_fail"] += tok.decode(ids, clean_up_tokenization_spaces=False) != text
        c["offsets_bad"] += any(b < a or a < 0 or b > len(text) for a, b in off) or any(off[k][0] < off[k - 1][0] for k in range(1, len(off)))
        return ids, off
    for r in sentences:
        words, end = r["words"], r["end"]; text = text_of(words); assert text == r["text"]
        ids0, off0 = one(text); wc = word_chars(words, text); c["n_sentences"] += 1
        c["end_merge"] += tok(text + end, add_special_tokens=False).input_ids != ids0 + tok(end, add_special_tokens=False).input_ids
        for f in fillers:
            for key in span_keys(len(words)):
                i, j = map(int, key.split(","))
                ids, _ = one(sub_text(words, i, j, f)); c["n_substitutions"] += 1
                cap_pre = sum(1 for (a, b) in off0 if b <= wc[i][0]); cap_suf = sum(1 for (a, b) in off0 if a >= wc[j][1])
                c["frame_pre_short"] += _common_prefix(ids0, ids) < cap_pre
                c["frame_suf_short"] += _common_prefix(ids0[::-1], ids[::-1]) < cap_suf
    return c


def span_keys(n):
    """v2 (job D): every span of an n-word sentence as 'i,j' (0-based, inclusive): j > i, the whole sentence excluded.
    Equal to the key set of the 1b cache for its 1,000 sentences."""
    return ["%d,%d" % (i, j) for i in range(n) for j in range(i + 1, n) if (i, j) != (0, n - 1)]


def jsd_distance(la, lb):
    """Jensen-Shannon distance, sqrt of the divergence in nats, between two log-prob vectors (float64)."""
    la = la.astype(np.float64); lb = lb.astype(np.float64); pa, pb = np.exp(la), np.exp(lb)
    lm = np.logaddexp(la, lb) - np.log(2.0)
    return float(np.sqrt(max(0.5 * np.sum(pa * (la - lm)) + 0.5 * np.sum(pb * (lb - lm)), 0.0)))


def t1_rows(tok, mdl, bos, words, span_keys, device="cpu"):
    """T1 for one sentence, one pass of the ORIGINAL text (no end mark). For span (i, j): L2 and Jensen-Shannon
    distance between the state after word i-1 (after BOS when i = 0) and the state after word j, where 'after word k'
    = after the last token that starts inside word k."""
    text = text_of(words); ids, off = encode(tok, text); wc = word_chars(words, text)
    (_, st), = forward(mdl, bos, [ids], device, want_states=True)
    last = [max(t for t, (a, b) in enumerate(off) if a < e and b > s) for (s, e) in wc]
    after = lambda k: st[0] if k < 0 else st[last[k] + 1]
    out = {}
    for key in span_keys:
        i, j = map(int, key.split(",")); a, b = after(i - 1), after(j)
        out[key] = [float(np.linalg.norm(a - b)), jsd_distance(a, b)]
    return out
