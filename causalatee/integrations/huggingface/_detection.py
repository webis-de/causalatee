from transformers import Pipeline


class CausalityDetectionPipeline(Pipeline):
    """Classifies whether a sentence expresses causal information."""

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, text):
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs):
        probs = model_outputs.logits[0].softmax(-1)
        label_id = probs.argmax().item()
        return {
            "label": self.model.config.id2label[label_id],
            "score": probs[label_id].item(),
        }
