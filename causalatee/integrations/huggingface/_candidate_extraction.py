from transformers import Pipeline


class CausalCandidateExtractionPipeline(Pipeline):
    """Extracts causal event candidate spans from a sentence.

    Expects a BIO token-classification model with labels such as
    ``B-CAUSE``, ``I-CAUSE``, ``B-EFFECT``, ``I-EFFECT``, ``O``.

    Returns a list of character-level span dicts::

        [{"start": 0, "end": 9, "entity": "CAUSE"}, ...]
    """

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
        )

    def _forward(self, model_inputs):
        offset_mapping = model_inputs.pop("offset_mapping")
        outputs = self.model(**model_inputs)
        return {"logits": outputs.logits, "offset_mapping": offset_mapping}

    def postprocess(self, model_outputs):
        logits = model_outputs["logits"][0]  # (seq_len, num_labels)
        offsets = model_outputs["offset_mapping"][0]  # (seq_len, 2)
        label_ids = logits.argmax(-1).tolist()

        spans = []
        current: dict | None = None

        for label_id, (char_start, char_end) in zip(label_ids, offsets.tolist()):
            # skip special tokens (offset (0, 0) except genuine first token)
            if char_start == 0 and char_end == 0:
                if current is not None:
                    spans.append(current)
                    current = None
                continue

            label = self.model.config.id2label[label_id]

            if label.startswith("B-"):
                if current is not None:
                    spans.append(current)
                current = {"start": char_start, "end": char_end, "entity": label[2:]}
            elif label.startswith("I-") and current is not None:
                current["end"] = char_end
            else:
                if current is not None:
                    spans.append(current)
                    current = None

        if current is not None:
            spans.append(current)

        return spans
