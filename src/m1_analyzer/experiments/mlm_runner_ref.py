"""Reference implementation for runner job E: the same substitutions as job D, scored by MASKED language models.

Imports tree_runner_ref (v4, unchanged) for the sentence text, the substituted texts, the span keys and the frame rule,
so a masked model scores exactly the substitutions a causal model scored in job D.

Score: PLL-word-l2r (Kauf & Ivanova, ACL 2023, doi 10.18653/v1/2023.acl-short.80; minicons PLL_metric='within_word_l2r').
For every token t of a sequence, the log-probability of t when t AND every later token of the same word (tokenizer word ids)
are replaced by <mask>; the model sees
<s> text-tokens end-mark-tokens </s>. Log-softmax in float32 at the target position only.

Why: a causal model cannot use the words BEFORE a span to judge what replaced it, because its predictions of those
words never see the replacement. For a span that ends the sentence, its only evidence is the end mark. A masked model
predicts every word from both sides, so the words before a sentence-final span do react to the replacement."""
import numpy as np, torch
import tree_runner_ref as TR
from tree_runner_ref import FILLERS, text_of, sub_text, word_chars, encode, span_keys, _common_prefix


def load_mlm(model_dir, revision=None, dtype=torch.float32, device="cpu"):
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained(model_dir, revision=revision)
    mdl = AutoModelForMaskedLM.from_pretrained(model_dir, revision=revision, dtype=dtype).to(device).eval()
    assert tok.is_fast and None not in (tok.mask_token_id, tok.cls_token_id, tok.sep_token_id, tok.pad_token_id)
    return tok, mdl


def encode_words(tok, text, n_end):
    """Token ids, offsets and word index of every token of text (no special tokens), then n_end end-mark tokens as
    one extra word. Words are the tokenizer's own pre-tokens (BatchEncoding.word_ids), as in minicons'
    PLL-word-l2r, so a contraction or a digit run is its own word and '.' is never merged with the word before it."""
    e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    w = e.word_ids(); assert None not in w and all(w[k] <= w[k + 1] for k in range(len(w) - 1)), ("word ids", text)
    nxt = (w[-1] + 1) if w else 0
    return list(e.input_ids), [tuple(o) for o in e.offset_mapping], list(w) + [nxt] * n_end


@torch.no_grad()
def masked_logits(mdl, inp, att, rr, pos, full=False):
    """Logits at the one masked position of each row. For RoBERTa the output head runs on that position only
    (identical numbers, a fraction of the cost); any other masked LM, or full=True, takes the full logits."""
    if not full and hasattr(mdl, "roberta") and hasattr(mdl, "lm_head"):
        h = mdl.roberta(input_ids=inp, attention_mask=att).last_hidden_state
        return mdl.lm_head(h[rr, pos])
    return mdl(input_ids=inp, attention_mask=att).logits[rr, pos]


@torch.no_grad()
def pll(tok, mdl, seqs, wids, device="cpu", batch=256):
    """PLL-word-l2r of every token of every sequence. seqs: token-id lists (text + end mark, no special tokens);
    wids: their word indices. One masked copy per token; copies are batched across sequences and right-padded."""
    jobs = []
    for s, (ids, wid) in enumerate(zip(seqs, wids)):
        assert len(ids) == len(wid)
        for t in range(len(ids)):
            u = t
            while u + 1 < len(ids) and wid[u + 1] == wid[t]: u += 1
            jobs.append((s, t, u))                                      # mask positions t..u (t and later pieces of its word)
    out = [np.zeros(len(x), dtype=np.float64) for x in seqs]
    for k in range(0, len(jobs), batch):
        ch = jobs[k:k + batch]; L = max(len(seqs[s]) for s, _, _ in ch) + 2
        inp = torch.full((len(ch), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros_like(inp)
        for r, (s, t, u) in enumerate(ch):
            x = [tok.cls_token_id] + list(seqs[s]) + [tok.sep_token_id]
            for v in range(t, u + 1): x[v + 1] = tok.mask_token_id
            inp[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
        rr = torch.arange(len(ch), device=device); pos = torch.tensor([t + 1 for _, t, _ in ch], device=device)
        tgt = torch.tensor([seqs[s][t] for s, t, _ in ch], device=device)
        lg = masked_logits(mdl, inp.to(device), att.to(device), rr, pos)
        lp = torch.log_softmax(lg.float(), -1)[rr, tgt].cpu().numpy()
        for (s, t, _), v in zip(ch, lp): out[s][t] = float(v)
    return out


def mlm_rows(tok, mdl, words, span_keys, end, fillers=FILLERS, batch=256, chunk=2048, device="cpu"):
    """Job E for one sentence. Returns (orig, rows), the same layout as tree_runner_ref.t4_rows:
      orig = {"text", "ids", "lp" (PLL-word-l2r of each text token), "end_ids", "end_lp"}
      rows[filler]["i,j"] = [total, n_tok, n_pre, n_suf, pre_sub, suf_sub, end_sub, first_sub, last_pre_sub]
        total, n_tok  summed PLL and count of the substituted sentence's text tokens (end mark excluded)
        n_pre, n_suf  frame tokens before / after the substitution, by exactly t4_rows' rule
        pre_sub       summed PLL of the n_pre prefix tokens in the substituted sentence (NOT ~0 here: a masked model's
                      prediction of a word before the span sees the replacement)
        suf_sub       summed PLL of the n_suf suffix tokens
        end_sub       summed PLL of the end-mark tokens
        first_sub     PLL of the first token after the span (first end-mark token if n_suf == 0), as in job D
        last_pre_sub  PLL of the last prefix token, the one just before the replacement (0.0 if n_pre == 0)
    Costs at analysis, each a mean of (orig - substituted) over its tokens:
      context = prefix + suffix + end mark;  continuation = suffix + end mark (the causal frame);  left = prefix only."""
    text = text_of(words); end_ids = tok(end, add_special_tokens=False).input_ids; ne = len(end_ids)
    ids0, off0, w0 = encode_words(tok, text, ne); wc = word_chars(words, text)
    (lp0e,) = pll(tok, mdl, [ids0 + end_ids], [w0], device, batch)
    orig = dict(text=text, ids=ids0, lp=[float(v) for v in lp0e[:len(ids0)]], end_ids=end_ids, end_lp=[float(v) for v in lp0e[len(ids0):]])
    jobs = []
    for f in fillers:
        for key in span_keys:
            i, j = map(int, key.split(","))
            st = sub_text(words, i, j, f); ids, off, wi = encode_words(tok, st, ne)
            cap_pre = sum(1 for (a, b) in off0 if b <= wc[i][0]); cap_suf = sum(1 for (a, b) in off0 if a >= wc[j][1])
            n_pre = min(_common_prefix(ids0, ids), cap_pre); n_suf = min(_common_prefix(ids0[::-1], ids[::-1]), cap_suf)
            if n_pre + n_suf > len(ids): n_suf = len(ids) - n_pre                  # t4_rows' rule, verbatim
            jobs.append((f, key, ids, n_pre, n_suf, wi))
    rows = {f: {} for f in fillers}
    for k in range(0, len(jobs), chunk):
        ch = jobs[k:k + chunk]
        for (f, key, ids, n_pre, n_suf, _), lpe in zip(ch, pll(tok, mdl, [c[2] + end_ids for c in ch], [c[5] for c in ch], device, batch)):
            n = len(ids); lp = lpe[:n]
            rows[f][key] = [float(lp.sum()), n, n_pre, n_suf, float(lp[:n_pre].sum()), float(lp[n - n_suf:].sum()) if n_suf else 0.0,
                            float(lpe[n:].sum()), float(lpe[n - n_suf]), float(lp[n_pre - 1]) if n_pre else 0.0]
    return orig, rows


def mlm_checks(tok, sentences, fillers=FILLERS, causal_tok=None):
    """Tokenizer-only checks for job E, once per tokenizer and set, before the full run. Deterministic, so the counts
    must equal the runner note's. word ids are asserted monotone inside encode_words.
      n_forwards          masked copies the set needs (the job's cost): one per text and end-mark token
      text_split_differs  texts that the causal model's tokenizer (GPT-2 for RoBERTa) splits into different token strings
      frame_differs       substitutions whose (n_pre, n_suf) differ from the causal tokenizer's (among equal splits)"""
    c = dict(n_sentences=0, n_substitutions=0, n_forwards=0, text_split_differs=0, frame_differs=0)
    for r in sentences:
        words, end = r["words"], r["end"]; text = text_of(words); assert text == r["text"]
        ne = len(tok(end, add_special_tokens=False).input_ids)
        ids0, off0, _ = encode_words(tok, text, ne); wc = word_chars(words, text); c["n_sentences"] += 1; c["n_forwards"] += len(ids0) + ne
        if causal_tok is not None:
            g0, _ = encode(causal_tok, text); c["text_split_differs"] += tok.convert_ids_to_tokens(ids0) != causal_tok.convert_ids_to_tokens(g0)
        for f in fillers:
            for key in span_keys(len(words)):
                i, j = map(int, key.split(","))
                st = sub_text(words, i, j, f); ids, _, _ = encode_words(tok, st, ne)
                c["n_substitutions"] += 1; c["n_forwards"] += len(ids) + ne
                if causal_tok is not None:
                    g, _ = encode(causal_tok, st)
                    same = tok.convert_ids_to_tokens(ids) == causal_tok.convert_ids_to_tokens(g); c["text_split_differs"] += not same
                    if same:
                        cap_pre = sum(1 for (a, b) in off0 if b <= wc[i][0]); cap_suf = sum(1 for (a, b) in off0 if a >= wc[j][1])
                        fr = lambda x0, x: (min(_common_prefix(x0, x), cap_pre), min(_common_prefix(x0[::-1], x[::-1]), cap_suf))
                        c["frame_differs"] += fr(ids0, ids) != fr(g0, g)
    return c
