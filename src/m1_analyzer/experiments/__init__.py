"""Sentence-structure experiments built on the scoring and state services.

Task 1b -- phase A (GPU): `compute_span_costs` scores every span of every
sentence and caches the raw numbers; phase B (CPU): `analyse_1b` turns the
cache into the record that `experiment_figures.fig_1b` draws.
Task 9a -- phase A: `compute_splices` scores every admissible cut of every
paragraph; phase B: `analyse_9a` builds the `fig_9a` record after the hand
audit. See notebooks/experiment_1b.ipynb and notebooks/experiment_9a.ipynb.
"""

from . import experiment_figures
from .boundaries import (
    BOUNDARY_KINDS, CLAUSE, SENTENCE, SPLITTERS, boundary_positions, sentence_char_spans, starts_sentence,
)
from .paragraphs import (
    WIKIPEDIA_NAME,
    Paragraph,
    hand_paragraphs,
    load_paragraph_file,
    load_wikipedia_paragraphs,
    select_paragraphs,
)
from .splice import (
    BOUNDARY_RULE,
    DEFAULT_DECILE,
    DEFAULT_MATCH_WIDTH,
    DEFAULT_STRATA,
    DEFAULT_WINDOW,
    DIVERGENCE_MEASURE,
    DISTANCE_MEASURE,
    Cut,
    admissible_cuts,
    compute_splices,
    cut_count,
    load_splices,
    paragraph_boundaries,
    preregistration,
    score_paragraph,
    splice_ids,
    splice_text,
)
from .task_9a import (
    analyse_9a,
    boundary_check,
    apply_audit,
    assign_pairs,
    audit_sample,
    flatten_cuts,
    load_audit_csv,
    write_audit_csv,
)
from .task_9a import validate_record as validate_record_9a
from .task_9a import verdict as verdict_9a
from .proforms import (
    DEFAULT_PROFORMS, DELETION, ByLengthClass, MinOverSet, ReplacementPolicy, parse_policy, substitute,
)
from .span_costs import (
    SpanCostTable, compute_span_costs, load_span_costs, load_span_costs_with_gold, variant_count,
)
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
    "BOUNDARY_KINDS", "CLAUSE", "SENTENCE", "SPLITTERS", "boundary_positions", "sentence_char_spans",
    "starts_sentence", "BOUNDARY_RULE", "boundary_check",
    "WIKIPEDIA_NAME", "Paragraph", "hand_paragraphs", "load_paragraph_file", "load_wikipedia_paragraphs",
    "select_paragraphs",
    "DEFAULT_DECILE", "DEFAULT_MATCH_WIDTH", "DEFAULT_STRATA", "DEFAULT_WINDOW", "DIVERGENCE_MEASURE",
    "DISTANCE_MEASURE", "Cut", "admissible_cuts", "compute_splices", "cut_count", "load_splices",
    "paragraph_boundaries", "preregistration", "score_paragraph", "splice_ids", "splice_text",
    "analyse_9a", "apply_audit", "assign_pairs", "audit_sample", "flatten_cuts", "load_audit_csv",
    "write_audit_csv", "validate_record_9a", "verdict_9a",
    "DEFAULT_PROFORMS", "DELETION", "ByLengthClass", "MinOverSet", "ReplacementPolicy", "parse_policy",
    "substitute",
    "SpanCostTable", "compute_span_costs", "load_span_costs", "load_span_costs_with_gold", "variant_count",
    "INDUCERS", "bracket_prf", "cky_induce", "crosses", "enumerate_spans", "greedy_induce",
    "left_branching", "random_binary", "rank_curve", "right_branching", "spans_to_brackets",
    "bootstrap_ci", "paired_bootstrap_diff", "trapezoid_area",
    "METHODS", "analyse_1b", "evaluate_sentence", "run_task_1b", "validate_record", "verdict",
    "CONVENTIONS", "PTB_NLTK_NAME", "PUNCT_TAGS", "TreebankSentence", "detokenize_ptb",
    "gold_spans_from_tree", "hand_examples", "load_gold_jsonl", "load_ptb_nltk",
    "ptb_tree_strings", "save_gold_jsonl", "theory_example",
]
