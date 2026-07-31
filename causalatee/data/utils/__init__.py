"""Shared helpers for causalatee's HF-datasets-native schema: entity-marker
parsing, whole-document -> per-sentence re-splitting, and same-granularity
schema derivations -- everything a dataset ``conversion_script.py`` or a
downstream evaluation harness needs that isn't specific to any one corpus.

Split into submodules by concern (all re-exported here, so
``from causalatee.data.utils import X`` keeps working unchanged):

* :mod:`.markers` -- :func:`parse_entity_markers` / :func:`insert_entity_markers`,
  the ``<eN>...</eN>`` marker <-> ``(clean_text, segments_by_eid)`` roundtrip
  used by causalatee's identification schema.
* :mod:`.splitting` -- :func:`split_document_by_sentence` (the generic core:
  given sentence ranges, entity segments, and "groups" of entity ids that
  must stay together, keep every group wholly inside one sentence, drop and
  COUNT the rest) plus :func:`split_identification_to_sentences` /
  :func:`split_extraction_to_sentences`, single-example wrappers over it.

  Several source corpora (BioCause, CaTeRS's brat annotations, CREST's own
  aggregation) annotate causal relations over a whole document or section,
  not a single sentence. A per-sentence schema cannot represent a relation
  whose participants straddle a sentence boundary (discourse-level
  causality really does occur, e.g. BioCause: "X requires Y ... [new
  sentence] This substitution decreases Z") -- silently re-basing such a
  span onto the wrong sentence's text is a real bug, not a hypothetical one
  (previously observed: a marker landing mid-word, "h<e1>istidine", in an
  early BioCause converter draft). :mod:`.splitting`'s job is to make the
  correct behaviour a single, tested, reusable piece of code instead of a
  duplicated ad-hoc guard in every converter.
* :mod:`.batch` -- ``Dataset.map(fn, batched=True, remove_columns=[...])``-
  ready adapters over the above: :func:`identification_batch_to_sentences`,
  :func:`extraction_batch_to_sentences`, :func:`identification_batch_to_detection_sentences`
  (re-splitting to sentence level), and :func:`identification_batch_to_detection`
  / :func:`identification_batch_to_extraction` (same-granularity detection/
  extraction tables derived from an identification table, for datasets like
  CTB/ESL that only ship identification).
* :mod:`.validation` -- :func:`verify_dataset`, structural sanity checks
  (conflicting/duplicate relation pairs, entity ids with no parseable
  marker span) run over a converted split. Every ``conversion_script.py``
  should call this right before writing a split's parquet file and print
  any returned errors as warnings.

IMPORTANT (see :mod:`.markers` for the full explanation): the ``<eN>...</eN>``
markers only *look* like XML/HTML tags -- they are not, and do not need to
be well-nested. Do not "fix" a non-nested marker sequence, and do not add
crossing-span detection/dropping logic on the assumption that it's needed
here -- it isn't.

Typical usage, re-splitting an already-loaded whole-document identification
split to sentence level (a detection or extraction split works the same
way with the matching batch function)::

    import spacy
    from causalatee.data.utils import identification_batch_to_sentences

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])

    def sentence_ranges(text):
        return [(s.start_char, s.end_char) for s in nlp(text).sents]

    per_sentence = whole_document_ds.map(
        lambda batch: identification_batch_to_sentences(batch, sentence_ranges),
        batched=True,
        remove_columns=whole_document_ds.column_names,
    )

Or, deriving a detection/extraction table at the SAME granularity for a
dataset that only ships identification (no sentence splitting involved)::

    from causalatee.data.utils import identification_batch_to_detection

    detection_ds = identification_ds.map(
        identification_batch_to_detection,
        batched=True,
        remove_columns=identification_ds.column_names,
    )

See ``conf-causality-repro/evaluation/data.py`` for a complete example
that dispatches all three sub-tasks this way.
"""

from __future__ import annotations

from .batch import (
    extraction_batch_to_sentences,
    identification_batch_to_detection,
    identification_batch_to_detection_sentences,
    identification_batch_to_extraction,
    identification_batch_to_sentences,
)
from .markers import Span, insert_entity_markers, parse_entity_markers
from .splitting import (
    split_document_by_sentence,
    split_extraction_to_sentences,
    split_identification_to_sentences,
)
from .validation import verify_dataset

__all__ = [
    "Span",
    "parse_entity_markers",
    "insert_entity_markers",
    "split_document_by_sentence",
    "split_identification_to_sentences",
    "split_extraction_to_sentences",
    "identification_batch_to_sentences",
    "extraction_batch_to_sentences",
    "identification_batch_to_detection_sentences",
    "identification_batch_to_detection",
    "identification_batch_to_extraction",
    "verify_dataset",
]
