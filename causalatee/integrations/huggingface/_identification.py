from transformers import Pipeline


class CausalityIdentificationPipeline(Pipeline):
    """Classifies the causal relation between two marked spans in a sentence.

    Expects input text with entity markers already inserted, e.g.::

        "<e1>The storm</e1> caused <e2>significant flooding</e2>."
    """

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
            "relation": self.model.config.id2label[label_id],
            "score": probs[label_id].item(),
        }
