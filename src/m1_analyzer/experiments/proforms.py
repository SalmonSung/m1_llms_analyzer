"""Replacement policies: which word stands in for a span, chosen blind.

The substitution test replaces a run of words with a proform and measures the
fluency drop. In the teaching doc the proform was chosen by hand, knowing what
the span was. Task 1b is not allowed that knowledge, so a *policy* picks the
replacement from the span's position and length alone. Two policies:

* `MinOverSet`: score every proform in a fixed set and take the cheapest
  (the theory doc's policy; costs 4 passes per span but keeps all four).
* `ByLengthClass`: one fixed proform per span-length class (the runbook's
  wording; what Task 1a is meant to settle).

Both satisfy `ReplacementPolicy`, and the cost cache stores every proform that
was scored, so a run made with `MinOverSet` can be re-read under any
`ByLengthClass` whose proforms are a subset -- no GPU needed.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

DEFAULT_PROFORMS: tuple[str, ...] = ("it", "there", "did", "then")

#: The reserved "proform" that deletes the span instead of replacing it. A control:
#: it asks whether a cheap span is cheap because *any* shortening helps the mean.
DELETION = "<del>"


def substitute(words: Sequence[str], i: int, j: int, proform: str) -> list[str]:
    """Words with ``words[i..j]`` replaced by `proform` (capitalised at i == 0).

    `proform` may be several words (``"do so"``); it is inserted as one entry and
    detokenised later. ``DELETION`` drops the span and, at i == 0, capitalises the
    word that becomes sentence-initial, so the variant still reads as a sentence.
    """
    if not (0 <= i <= j < len(words)):
        raise ValueError(f"span ({i}, {j}) is outside 0..{len(words) - 1}")
    if proform == DELETION:
        rest = list(words[:i]) + list(words[j + 1 :])
        if i == 0 and rest:
            rest[0] = rest[0][:1].upper() + rest[0][1:]
        return rest
    replacement = proform[:1].upper() + proform[1:] if i == 0 else proform
    return list(words[:i]) + [replacement] + list(words[j + 1 :])


def _clean(items: Sequence[str]) -> tuple[str, ...]:
    """Strip, drop blanks, keep first occurrences in order."""
    return tuple(dict.fromkeys(p.strip() for p in items if p.strip()))


@runtime_checkable
class ReplacementPolicy(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def proforms(self) -> tuple[str, ...]:
        """Every proform this policy can ever ask for (what a cache must hold)."""

    def candidates(self, words: Sequence[str], i: int, j: int) -> list[str]:
        """The proforms to score for span (i, j)."""

    def choose(self, costs: Mapping[str, float]) -> float:
        """Reduce per-proform costs of one span to the span's cost."""


class MinOverSet:
    """Score each proform in the set; the span's cost is the cheapest.

    `controls` (``"blorp"``, ``DELETION`` ...) are scored and cached for every span
    exactly like the proforms, so phase B can compare against them, but they never
    take part in the min: the induction cost stays "does any real proform fit?".
    """

    def __init__(self, proforms: Sequence[str] = DEFAULT_PROFORMS, controls: Sequence[str] = ()):
        self._proforms = _clean(proforms)
        if not self._proforms:
            raise ValueError("MinOverSet needs at least one proform.")
        self._controls = tuple(c for c in _clean(controls) if c not in self._proforms)

    @property
    def name(self) -> str:
        name = "min over {%s}" % ", ".join(self._proforms)
        if self._controls:
            name += " + controls {%s}" % ", ".join(self._controls)
        return name

    @property
    def proforms(self) -> tuple[str, ...]:
        """Proforms then controls: everything phase A scores and a cache must hold."""
        return self._proforms + self._controls

    @property
    def controls(self) -> tuple[str, ...]:
        return self._controls

    def candidates(self, words: Sequence[str], i: int, j: int) -> list[str]:
        return list(self.proforms)

    def choose(self, costs: Mapping[str, float]) -> float:
        available = [costs[p] for p in self._proforms if p in costs]
        if not available:
            raise KeyError("no cost for any proform in %r" % (self._proforms,))
        return min(available)


class ByLengthClass:
    """One fixed proform per span-length class, e.g. ``{(2, 3): "it", (4, None): "that"}``.

    Keys are ``(min_words, max_words)`` with ``None`` for no upper bound.
    """

    def __init__(self, mapping: Mapping[tuple[int, int | None], str]):
        if not mapping:
            raise ValueError("ByLengthClass needs at least one length class.")
        self._classes = sorted(((lo, hi, p) for (lo, hi), p in mapping.items()), key=lambda c: c[0])

    @property
    def name(self) -> str:
        return "by length: " + ", ".join(
            f"{lo}-{hi if hi is not None else ''} -> {p!r}" for lo, hi, p in self._classes
        )

    @property
    def proforms(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p for _, _, p in self._classes))

    def proform_for(self, length: int) -> str:
        for lo, hi, p in self._classes:
            if length >= lo and (hi is None or length <= hi):
                return p
        raise ValueError(f"no length class covers a span of {length} words")

    def candidates(self, words: Sequence[str], i: int, j: int) -> list[str]:
        return [self.proform_for(j - i + 1)]

    def choose(self, costs: Mapping[str, float]) -> float:
        if len(costs) != 1:
            # A cache may hold more proforms than the class needs; caller passes only
            # the relevant one, but be tolerant and take the cheapest of what is there.
            return min(costs.values())
        return next(iter(costs.values()))


def parse_policy(spec: str, controls: str = "") -> ReplacementPolicy:
    """Build a policy from a short string, for notebook config fields.

    ``"min:it,there,did,then"`` -> MinOverSet; ``"length:2-3=it,4-6=that,7+=this"``
    -> ByLengthClass. A bare comma list means MinOverSet. `controls` is a comma
    list (``"blorp,<del>"``) that MinOverSet scores but never chooses; ``<del>`` is
    the deletion control in either list.
    """
    spec = spec.strip()
    if controls.strip() and spec.startswith("length:"):
        raise ValueError("controls are only supported with a min-over-set policy")
    if spec.startswith("length:"):
        mapping: dict[tuple[int, int | None], str] = {}
        for part in spec[len("length:"):].split(","):
            rng, _, proform = part.partition("=")
            rng, proform = rng.strip(), proform.strip()
            if not proform:
                raise ValueError(f"length class {part!r} has no proform")
            if rng.endswith("+"):
                mapping[(int(rng[:-1]), None)] = proform
            else:
                lo, _, hi = rng.partition("-")
                mapping[(int(lo), int(hi or lo))] = proform
        return ByLengthClass(mapping)
    if spec.startswith("min:"):
        spec = spec[len("min:"):]
    return MinOverSet(spec.split(","), controls=controls.split(","))
