from __future__ import annotations

try:
    import torch
    import torchmetrics
except ImportError:
    raise ImportError(
        "The Lightning integration requires torch and torchmetrics.\n"
        "Install them with: pip install 'causalatee[lightning]'"
    ) from None

from causalatee.evaluation._spans import bio_to_spans, span_scores


class SpanMetric(torchmetrics.Metric):
    """Macro-averaged character-span metric for token-classification models.

    Primary interface (:meth:`update`) takes character spans directly — the same format that
    :class:`~causalatee.integrations.huggingface.CausalCandidateExtractionPipeline` returns in production.
    :meth:`update_from_bio` is a convenience wrapper for training loops where BIO token-label ids are more readily
    available.

    Compatible with multi-GPU / DDP: running sums are plain ``torch.Tensor`` states reduced with ``"sum"``.

    Parameters
    ----------
    id2label:
        Mapping from label index to BIO label string, e.g. ``{0: "O", 1: "B-CAUSE", 2: "I-CAUSE", ...}``.
        Typically ``model.config.id2label``.

    Example
    -------
    Production-style (spans already decoded by the pipeline)::

        class CausalNER(LightningModule):
            def __init__(self, model, pipe):
                super().__init__()
                self.model = model
                self.pipe = pipe                    # CausalCandidateExtractionPipeline
                self.val_span = SpanMetric(model.config.id2label)

            def validation_step(self, batch, batch_idx):
                pred_spans = [self.pipe(t) for t in batch["text"]]   # list[list[tuple]]
                gold_spans = batch["spans"]                           # list[list[tuple]]
                self.val_span.update(pred_spans, gold_spans)

            def on_validation_epoch_end(self):
                self.log_dict(self.val_span.compute(), prog_bar=True)
                self.val_span.reset()

    Training-time shortcut (BIO logits from ``model.forward``)::

            def validation_step(self, batch, batch_idx):
                logits = self.model(**batch).logits            # (B, L, C)
                self.val_span.update_from_bio(
                    predictions=logits.argmax(-1).tolist(),
                    label_ids=batch["labels"].tolist(),
                    offset_mappings=batch["offset_mapping"].tolist(),
                )
    """

    higher_is_better: bool = True
    full_state_update: bool = False

    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    f1_gran: torch.Tensor
    iou: torch.Tensor
    count: torch.Tensor

    def __init__(self, id2label: dict[int, str]) -> None:
        super().__init__()
        self.id2label = id2label
        # add_state() sets these attributes dynamically (for DDP reduction), so mypy can't infer their type
        # from it alone -- the class-level annotations above tell mypy they're always torch.Tensor rather than
        # falling back to Module.__getattr__'s generic Tensor | Module stub.
        for key in ("precision", "recall", "f1", "f1_gran", "iou"):
            self.add_state(key, default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        predictions: list[list[tuple[int, int]]],
        targets: list[list[tuple[int, int]]],
    ) -> None:
        """Accumulate one batch of character-span predictions.

        Both arguments use the same format that
        :class:`~causalatee.integrations.huggingface.CausalCandidateExtractionPipeline` returns at inference
        time, so evaluation and production use the same representation.

        Parameters
        ----------
        predictions:
            Predicted character spans per example, e.g.
            ``[[(0, 4), (14, 19)], [(3, 8)]]``.
        targets:
            Gold character spans per example in the same format.
        """
        for pred_spans, gold_spans in zip(predictions, targets):
            scores = span_scores(gold_spans, pred_spans)
            self.precision += scores["precision"]
            self.recall += scores["recall"]
            self.f1 += scores["f1"]
            self.f1_gran += scores["f1_gran"]
            self.iou += scores["intersection_over_union"]
            self.count += 1

    def update_from_bio(
        self,
        predictions: list[list[int]],
        label_ids: list[list[int]],
        offset_mappings: list[list[tuple[int, int]]],
    ) -> None:
        """Accumulate from raw BIO token predictions (training-time convenience).

        Use this when you have token-level logits / label ids rather than decoded spans — for example inside a
        ``validation_step`` that receives batched model outputs directly.

        Parameters
        ----------
        predictions:
            Predicted BIO label indices, one list per example.
        label_ids:
            Gold BIO label indices; ``-100`` positions are treated as padding.
        offset_mappings:
            Per-token ``(char_start, char_end)`` pairs, one list per example.
        """
        pred_spans = [bio_to_spans(p, o, self.id2label) for p, o in zip(predictions, offset_mappings)]
        gold_spans = [bio_to_spans(g, o, self.id2label) for g, o in zip(label_ids, offset_mappings)]
        self.update(pred_spans, gold_spans)

    def compute(self) -> dict[str, torch.Tensor]:
        """Return macro-averaged scores over all accumulated examples.

        Keys are prefixed with ``span/`` so they appear grouped in loggers that support namespacing
        (TensorBoard, W&B, etc.).
        """
        if self.count == 0:
            zero = torch.tensor(0.0)
            return {f"span/{k}": zero for k in ("precision", "recall", "f1", "f1_gran", "iou")}
        n = self.count.float()
        return {
            "span/precision": self.precision / n,
            "span/recall": self.recall / n,
            "span/f1": self.f1 / n,
            "span/f1_gran": self.f1_gran / n,
            "span/iou": self.iou / n,
        }
