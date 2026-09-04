"""Sentence-structure experiments built on the scoring service.

Phase A (GPU): `compute_span_costs` scores every span of every sentence and
caches the raw numbers. Phase B (CPU): `analyse_1b` turns the cache into the
record that `experiment_figures.fig_1b` draws. See notebooks/experiment_1b.ipynb.
"""

from . import experiment_figures
from .proforms import DEFAULT_PROFORMS, ByLengthClass, MinOverSet, ReplacementPolicy, parse_policy, substitute
from .span_costs import SpanCostTable, compute_span_costs, load_span_costs, variant_count
from .spans import (
    INDUCERS,
    bracket_prf,
    cky_induce,
    crosses,
    enumerate_spans,
    greedy_induce,
    left_branching,
    random_binary,
    rank_curve,
    right_branching,
    spans_to_brackets,
)
from .stats import bootstrap_ci, paired_bootstrap_diff, trapezoid_area
from .task_1b import METHODS, analyse_1b, evaluate_sentence, run_task_1b, validate_record, verdict
from .treebank import (
    CONVENTIONS,
    PTB_NLTK_NAME,
    PUNCT_TAGS,
    TreebankSentence,
    detokenize_ptb,
    gold_spans_from_tree,
    hand_examples,
    load_gold_jsonl,
    load_ptb_nltk,
    ptb_tree_strings,
    save_gold_jsonl,
    theory_example,
)

__all__ = [
    "experiment_figures",
    "DEFAULT_PROFORMS", "ByLengthClass", "MinOverSet", "ReplacementPolicy", "parse_policy", "substitute",
    "SpanCostTable", "compute_span_costs", "load_span_costs", "variant_count",
    "INDUCERS", "bracket_prf", "cky_induce", "crosses", "enumerate_spans", "greedy_induce",
    "left_branching", "random_binary", "rank_curve", "right_branching", "spans_to_brackets",
    "bootstrap_ci", "paired_bootstrap_diff", "trapezoid_area",
    "METHODS", "analyse_1b", "evaluate_sentence", "run_task_1b", "validate_record", "verdict",
    "CONVENTIONS", "PTB_NLTK_NAME", "PUNCT_TAGS", "TreebankSentence", "detokenize_ptb",
    "gold_spans_from_tree", "hand_examples", "load_gold_jsonl", "load_ptb_nltk",
    "ptb_tree_strings", "save_gold_jsonl", "theory_example",
]
