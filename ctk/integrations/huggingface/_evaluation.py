from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from ctk.evaluation._spans import bio_to_spans, dataset_span_scores

if TYPE_CHECKING:
    from transformers import EvalPrediction


def span_compute_metrics(
    id2label: dict[int, str],
    eval_offset_mappings: Sequence[Sequence[tuple[int, int]]],
) -> Callable[[EvalPrediction], dict[str, float]]:
    """Return a ``compute_metrics`` function for :class:`~transformers.Trainer`.

    The returned callable decodes BIO token predictions to character-level spans
    using the provided offset mappings, then computes macro-averaged precision,
    recall, F1, granularity-penalised F1, and IoU — matching the Touché subtask-2
    evaluator.

    Parameters
    ----------
    id2label:
        Mapping from label index to BIO label string, e.g.
        ``{0: "O", 1: "B-CAUSE", 2: "I-CAUSE", ...}``.
        Typically ``model.config.id2label``.
    eval_offset_mappings:
        Per-token ``(char_start, char_end)`` pairs for every example in the
        evaluation split.  Capture these from the tokenised dataset *before*
        removing them with ``remove_columns``, then pass the captured list here::

            eval_offsets = tokenised_eval["offset_mapping"]
            trainer = Trainer(
                ...,
                compute_metrics=span_compute_metrics(model.config.id2label, eval_offsets),
            )

    Returns
    -------
    callable
        A function with the signature ``(EvalPrediction) -> dict[str, float]``
        accepted by :class:`~transformers.Trainer`.
    """
    offsets = list(eval_offset_mappings)  # snapshot; avoids HF Dataset lazy-eval surprises

    def _compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids

        # Accept both raw logits (3-D) and already argmax'd label ids (2-D).
        if predictions.ndim == 3:
            predictions = predictions.argmax(-1)

        n = len(predictions)
        all_preds = [bio_to_spans(predictions[i].tolist(), offsets[i], id2label) for i in range(n)]
        all_truths = [bio_to_spans(label_ids[i].tolist(), offsets[i], id2label) for i in range(n)]
        return dataset_span_scores(all_truths, all_preds)

    return _compute_metrics
