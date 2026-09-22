"""Task 8a frames: the clause-return item generator and its tokenizer filter.

Act 8 of the teaching document measures whether, after an embedded clause
closes (``The soldier that the singer trusted |``), the model's next-token
state sits closer to the bare subject's (``The soldier |``) than a control of
the same token length that ends on the same word but has no clause to close
(``The singer heard the soldier trusted |``). Task 8a scores that test on
several models with **one frame list shared by all of them**, so that whatever
differs between models is the model, not the items.

The generator between the ``BEGIN`` / ``END`` markers below is the one the
request specifies, copied **unmodified** (the record's ``meta.generator``
says so). A frame is kept only if it passes every assertion in every
tokenizer it is filtered on: each pool word is a single token with a leading
space (``tk(" " + w, add_special_tokens=False)``), every (embedded, control)
pair has the same token count under the tokenizer's *default*
``add_special_tokens`` (a BOS is counted on both sides and cancels) and, for
the token-matched ``_B`` controls, the same last token. ``ORC_A`` is the legacy
control: exempt from the last-token assertion by design, held to the count.

Everything outside the markers is glue: the pass codes in template order,
the fixed verb pool the verb-mass readout sums over, the ``"prefix |suffix"``
sentence format, the frame-file round trip, and the per-tokenizer token table
that lets the assertions be re-checked from a record without the tokenizers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

# --- BEGIN verbatim generator (the request's code block, seed 8) --------------
import random
NOUNS = ("author student engineer singer lawyer teacher doctor farmer pilot poet judge nurse actor banker soldier painter "
         "dancer driver baker coach mayor priest sailor tailor waiter writer editor critic guard clerk monk chef king queen "
         "senator general professor officer bishop merchant hunter scholar sheriff butler gardener miner knight wizard "
         "captain colonel prince princess duke widow stranger neighbor cousin uncle aunt boy girl man woman child").split()
VERBS = ("praised hired fired trusted admired ignored followed helped blamed taught feared loved hated met called visited "
         "thanked warned watched married attacked defended questioned rescued criticized mocked greeted invited betrayed "
         "chased kissed pushed punished paid protected remembered noticed described mentioned recognized supported").split()
MAINS = [("won", "the prize"), ("left", "the room"), ("passed", "the exam"), ("signed", "the contract"), ("fixed", "the engine"),
         ("filed", "the appeal"), ("wrote", "a letter"), ("opened", "the door"), ("missed", "the train"), ("sold", "the house"),
         ("bought", "a car"), ("lost", "the game"), ("kept", "the secret"), ("read", "the report"), ("found", "the key"),
         ("broke", "the window"), ("paid", "the bill"), ("answered", "the phone"), ("closed", "the shop"), ("built", "a fence")]
CVS = "said thought knew claimed believed heard felt".split()
FUNC = ["that", "whom", "quickly", "and", "the", "The"]           # must also be single tokens everywhere

def single(tk, w): return len(tk(" " + w, add_special_tokens=False).input_ids) == 1
def n_tok(tk, p): return len(tk(p).input_ids)                    # default add_special_tokens: BOS counted on both sides
def last_tok(tk, p): return tk.decode([tk(p).input_ids[-1]])

def filter_pools(tokenizers):
    keep = lambda ws: [w for w in ws if all(single(tk, w) for tk in tokenizers)]
    return keep(NOUNS), keep(VERBS), [(v, o) for v, o in MAINS if all(single(tk, v) for tk in tokenizers)], keep(CVS)

def make_frame(rng, NOUNS, VERBS, MAINS, CVS):
    n1, n2, n3, n4 = rng.sample(NOUNS, 4); v2, v3, v4 = rng.sample(VERBS, 3); V, OBJ = rng.choice(MAINS); cv1, cv2, cv3 = rng.sample(CVS, 3)
    fr = dict(N1=n1, N2=n2, N3=n3, N4=n4, V2=v2, V3=v3, V4=v4, V=V, OBJ=OBJ, CV1=cv1, CV2=cv2, CV3=cv3)
    S = {  # code: (prefix up to the stop, suffix after it) - the prefix is what the model sees
        "REF":    ("The {N1}", " {V} {OBJ}."),
        "ORC":    ("The {N1} that the {N2} {V2}", " {V} {OBJ}."),
        "ORC_A":  ("The {N2} {V2} the {N1} that", " {V} {OBJ}."),                      # Act 8's legacy control
        "ORC_B":  ("The {N2} {CV1} the {N1} {V2}", " {OBJ}."),                          # token-matched control
        "WHO":    ("The {N1} whom the {N2} {V2}", " {V} {OBJ}."),
        "WHO_B":  ("The {N2} {CV1} the {N1} {V2}", " {OBJ}."),
        "RED":    ("The {N1} the {N2} {V2}", " {V} {OBJ}."),
        "RED_B":  ("The {N2} and {N3} {V2}", " {OBJ}."),
        "SRC":    ("The {N1} that {V2} the {N2}", " {V} {OBJ}."),
        "SRC_B":  ("The {N1} quickly {V2} the {N2}", "."),
        "EMB2":   ("The {N1} that the {N2} that the {N3} {V3} {V2}", " {V} {OBJ}."),
        "EMB2_B": ("The {N3} {CV2} that the {N2} {CV1} the {N1} {V2}", " {OBJ}."),
        "EMB3":   ("The {N1} that the {N2} that the {N3} that the {N4} {V4} {V3} {V2}", " {V} {OBJ}."),
        "EMB3_B": ("The {N4} {CV3} that the {N3} {CV2} that the {N2} {CV1} the {N1} {V2}", " {OBJ}."),
    }
    return fr, {k: (p.format(**fr), s.format(**fr)) for k, (p, s) in S.items()}

PAIRS = [("ORC", "ORC_B"), ("ORC", "ORC_A"), ("WHO", "WHO_B"), ("RED", "RED_B"), ("SRC", "SRC_B"), ("EMB2", "EMB2_B"), ("EMB3", "EMB3_B")]

def passes(S, tokenizers):
    for tk in tokenizers:
        for e, c in PAIRS:
            if n_tok(tk, S[e][0]) != n_tok(tk, S[c][0]): return False
            if c.endswith("_B") and last_tok(tk, S[e][0]) != last_tok(tk, S[c][0]): return False
    return True

def generate(tokenizers, n=200, seed=8, max_draws=3000):
    N, Vb, Mn, C = filter_pools(tokenizers); assert all(all(single(tk, w) for tk in tokenizers) for w in FUNC), "a function word is not a single token somewhere"
    rng = random.Random(seed); frames = []; draws = 0
    while len(frames) < n and draws < max_draws:
        draws += 1; fr, S = make_frame(rng, N, Vb, Mn, C)
        if any(f["N1"] == fr["N1"] and f["V"] == fr["V"] for f, _ in frames): continue
        if passes(S, tokenizers): frames.append((fr, S))
    return frames, dict(pool_nouns=len(N), pool_verbs=len(Vb), pool_mains=len(Mn), pool_cvs=len(C), draws=draws)
# --- END verbatim generator ----------------------------------------------------

GENERATOR_NOTE = "the code block above, unmodified"

#: The pass codes, in the template order of `make_frame`; ``REF`` is the bare subject.
CODES: tuple[str, ...] = (
    "REF", "ORC", "ORC_A", "ORC_B", "WHO", "WHO_B", "RED", "RED_B", "SRC", "SRC_B",
    "EMB2", "EMB2_B", "EMB3", "EMB3_B",
)
REF = "REF"
#: The frame's slot names, in the order `make_frame` fills them.
SLOTS: tuple[str, ...] = ("N1", "N2", "N3", "N4", "V2", "V3", "V4", "V", "OBJ", "CV1", "CV2", "CV3")

#: The verb pool the verb-mass readout sums over (70 past-tense / finite
#: verbs, fixed by the request). Per model only the words whose leading-space
#: form is a single token are kept (`verb_pool_kept`); the pool is never extended.
VERB_POOL_8A: tuple[str, ...] = tuple(
    "admired answered attacked became began betrayed blamed bought broke built called came closed criticized "
    "defended did died feared filed fired fixed followed found gave got greeted had has hated helped hired ignored "
    "invited is kept knew left lived lost loved made married met missed mocked opened paid passed played praised "
    "questioned read rescued said saw signed sold taught thanked took trusted visited wanted warned was watched "
    "went won worked wrote".split()
)

#: Every word a frame can contain, so a word-level test tokenizer can spell any
#: prefix with one token per word (see `testing.build_tiny_local_model(extra_vocab=...)`).
FRAME_WORDS_8A: tuple[str, ...] = tuple(sorted(
    set(NOUNS) | set(VERBS) | {v for v, _ in MAINS} | {w for _, o in MAINS for w in o.split()}
    | set(CVS) | set(FUNC) | set(VERB_POOL_8A) | {"."}
))


# ------------------------------------------------------------- sentences


def format_sentence(prefix: str, suffix: str) -> str:
    """``"prefix |suffix"``: the stop ``|`` marks where the prefix ends (no trailing space)."""
    return f"{prefix} |{suffix}"


def split_sentence(text: str) -> tuple[str, str]:
    """Inverse of `format_sentence`: ``(prefix, suffix)``. Accepts ``"prefix | suffix"`` too.

    The prefix is everything before the first ``" |"``; nothing else is stripped,
    so a prefix never gains or loses a trailing space on the way through a file.
    """
    prefix, sep, suffix = text.partition(" |")
    if not sep:
        raise ValueError(f"no ' |' stop in sentence {text!r}")
    return prefix, suffix


# ------------------------------------------------------------- generation


def generate_frames_8a(
    tokenizers: Mapping[str, Any],
    *,
    n: int = 200,
    seed: int = 8,
    max_draws: int = 3000,
) -> dict[str, Any]:
    """Draw the frame list once, filtered on every tokenizer in `tokenizers`.

    Returns the record's ``meta`` / ``verb_pool`` / ``frames`` blocks: each frame
    is its twelve slots plus ``sentences`` (``code -> "prefix |suffix"``). The
    frame index in the list is the frame id everywhere else. Pool sizes are the
    post-filter sizes; ``draws`` is how many draws the generator used.
    """
    if not tokenizers:
        raise ValueError("generate_frames_8a needs at least one tokenizer to filter on.")
    keys = list(tokenizers.keys())
    frames, pools = generate([tokenizers[k] for k in keys], n=n, seed=seed, max_draws=max_draws)
    out = []
    for fr, S in frames:
        entry = {slot: fr[slot] for slot in SLOTS}
        entry["sentences"] = {code: format_sentence(*S[code]) for code in CODES}
        out.append(entry)
    return {
        "meta": {
            "seed": int(seed), "n_frames_requested": int(n), "n_frames": len(out),
            "pools": dict(pools), "tokenizers_filtered_on": keys, "generator": GENERATOR_NOTE,
        },
        "verb_pool": list(VERB_POOL_8A),
        "frames": out,
    }


def frame_prefixes(frame: Mapping[str, Any]) -> dict[str, str]:
    """``code -> prefix`` (what the model sees) from a frame's ``sentences``."""
    return {code: split_sentence(frame["sentences"][code])[0] for code in CODES}


# --------------------------------------------------------------- tokens


def token_table(tokenizer: Any, frames: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``[{frame, code, n_tok, last_tok}]`` for every pass of every frame, under this tokenizer.

    Counts use the tokenizer's default ``add_special_tokens`` (so ``bos_added``
    models count one more on every pass); ``last_tok`` is the decoded last id.
    """
    rows = []
    for index, frame in enumerate(frames):
        for code, prefix in frame_prefixes(frame).items():
            rows.append({"frame": index, "code": code, "n_tok": n_tok(tokenizer, prefix),
                         "last_tok": last_tok(tokenizer, prefix)})
    return rows


def token_assertions(n_tok_by_code: Mapping[str, int], last_tok_by_code: Mapping[str, str]) -> list[str]:
    """The `passes` assertions on recorded counts and last tokens; the failures, as text.

    Empty means the frame passes under the tokenizer these values came from.
    """
    failures = []
    for e, c in PAIRS:
        if n_tok_by_code[e] != n_tok_by_code[c]:
            failures.append(f"{e}/{c}: n_tok {n_tok_by_code[e]} != {n_tok_by_code[c]}")
        if c.endswith("_B") and last_tok_by_code[e] != last_tok_by_code[c]:
            failures.append(f"{e}/{c}: last_tok {last_tok_by_code[e]!r} != {last_tok_by_code[c]!r}")
    return failures


def verb_pool_kept(tokenizer: Any, pool: Sequence[str] = VERB_POOL_8A) -> list[str]:
    """The pool words whose leading-space form is one token in this tokenizer, in pool order."""
    return [w for w in pool if single(tokenizer, w)]


# ---------------------------------------------------------------- files


def save_frames_8a(payload: Mapping[str, Any], path: str | os.PathLike) -> Path:
    """Write the generated frames (atomically) so every model scores the same list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_frames_8a(path: str | os.PathLike) -> dict[str, Any]:
    """Read a frames file back and check its shape (14 sentences per frame, the meta block)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("meta", "verb_pool", "frames"):
        if key not in payload:
            raise ValueError(f"{path} is not a Task 8a frames file: missing {key!r}.")
    for index, frame in enumerate(payload["frames"]):
        missing = [code for code in CODES if code not in frame.get("sentences", {})]
        if missing:
            raise ValueError(f"{path}: frame {index} lacks sentences for {missing}.")
        for code in CODES:
            split_sentence(frame["sentences"][code])
    return payload


def frames_sha256(path: str | os.PathLike) -> str:
    """Identity of a frames file, stamped into every cache header and checked on merge."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "NOUNS", "VERBS", "MAINS", "CVS", "FUNC", "PAIRS", "single", "n_tok", "last_tok", "filter_pools",
    "make_frame", "passes", "generate", "GENERATOR_NOTE", "CODES", "REF", "SLOTS", "VERB_POOL_8A",
    "FRAME_WORDS_8A", "format_sentence", "split_sentence", "generate_frames_8a", "frame_prefixes",
    "token_table", "token_assertions", "verb_pool_kept", "save_frames_8a", "load_frames_8a", "frames_sha256",
]
