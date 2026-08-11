"""Structural-typing, batch-calling, and pairwise->identification adapter tests for causalatee.models."""

from __future__ import annotations

from causalatee.data.constants import Relation
from causalatee.models import (
    CandidateExtraction,
    Detection,
    Extraction,
    Identification,
    PairwiseIdentification,
    identify_candidates,
    lift_pairwise_identification,
)


class FakeDetectionPipeline:
    """No inheritance from Detection anywhere -- structural conformance only."""

    def __call__(self, text):
        if isinstance(text, list):
            return [{"label": "Causal", "score": 0.99} for _ in text]
        return {"label": "Causal", "score": 0.99}


class FakeCandidateExtractionPipeline:
    def __call__(self, text):
        one = [{"start": 0, "end": 3, "entity": "CAUSE"}]
        if isinstance(text, list):
            return [one for _ in text]
        return one


class FakePairwiseIdentificationPipeline:
    """Records every text it was actually called with, batched or not, so tests can assert the lifted adapter makes ONE
    batched call rather than N single ones.

    Default behavior: "Countercausal" for any pair marking "Sugar" as <e1>, else "NoRelation" -- override via
    ``relation`` to make every pair unconditionally classify as ``relation`` instead (useful for tests that don't
    care about which specific direction/pair is picked out, only that all of them are)."""

    def __init__(self, relation: str | None = None):
        self._relation = relation
        self.calls: list[list[str]] = []

    def __call__(self, text):
        batch = [text] if isinstance(text, str) else text
        self.calls.append(list(batch))
        if self._relation is not None:
            results = [{"relation": self._relation, "score": 0.9} for _ in batch]
        else:
            results = [
                {"relation": "Countercausal", "score": 0.9}
                if "<e1>Sugar</e1>" in t
                else {"relation": "NoRelation", "score": 0.8}
                for t in batch
            ]
        return results[0] if isinstance(text, str) else results


class FakeExtractionPipeline:
    def __call__(self, text):
        one = [{"e1": "a", "e2": "b", "relation": "Causal", "score": 0.5}]
        if isinstance(text, list):
            return [one for _ in text]
        return one


class TestStructuralConformance:
    def test_plain_callables_satisfy_protocols_without_inheriting(self):
        assert isinstance(FakeDetectionPipeline(), Detection)
        assert isinstance(FakeCandidateExtractionPipeline(), CandidateExtraction)
        assert isinstance(FakePairwiseIdentificationPipeline(), PairwiseIdentification)
        assert isinstance(FakeExtractionPipeline(), Extraction)

    def test_object_missing_call_does_not_satisfy(self):
        assert not isinstance(object(), Detection)

    def test_batch_call_returns_one_result_per_input(self):
        model = FakeDetectionPipeline()
        assert model("a") == {"label": "Causal", "score": 0.99}
        assert model(["a", "b"]) == [
            {"label": "Causal", "score": 0.99},
            {"label": "Causal", "score": 0.99},
        ]


class TestLiftPairwiseIdentification:
    def test_enumerates_all_marked_pairs_and_drops_no_relation(self):
        model: PairwiseIdentification = FakePairwiseIdentificationPipeline()
        identifier: Identification = lift_pairwise_identification(model)

        text = "<e1>Sugar</e1> does not cause <e2>hyperactivity</e2>, says <e3>study</e3>."
        relations = identifier(text)

        assert {
            "relationship": Relation.Countercausal,
            "first": "e1",
            "second": "e2",
            "score": 0.9,
        } in relations
        assert all(r["relationship"] != Relation.NoRelation for r in relations)

    def test_single_text_makes_exactly_one_batched_call_to_the_underlying_model(self):
        model = FakePairwiseIdentificationPipeline()
        identifier = lift_pairwise_identification(model)

        text = "<e1>Sugar</e1> does not cause <e2>hyperactivity</e2>, says <e3>study</e3>."
        identifier(text)

        assert len(model.calls) == 1  # one call...
        assert len(model.calls[0]) == 6  # ...covering all 3*2 ordered pairs

    def test_batch_of_texts_flattens_into_a_single_underlying_call(self):
        model = FakePairwiseIdentificationPipeline()
        identifier = lift_pairwise_identification(model)

        texts = [
            "<e1>Sugar</e1> does not cause <e2>hyperactivity</e2>.",
            "<e1>The storm</e1> caused <e2>flooding</e2>.",
        ]
        results = identifier(texts)

        assert len(results) == 2
        assert len(model.calls) == 1  # both texts' pairs flattened into one call
        assert len(model.calls[0]) == 4  # 2 ordered pairs (e1,e2) per text x 2 texts


class TestIdentifyCandidates:
    """identify_candidates is the shared marking/enumeration/remapping logic factored out of _ComposedExtraction so
    causalatee.mining can reuse it directly, given already-known texts + candidate spans (no
    Detection/CandidateExtraction gating here -- that's the caller's job, see _ComposedExtraction for the example)."""

    def test_gates_out_texts_with_fewer_than_two_spans_without_calling_the_model(self):
        model = FakePairwiseIdentificationPipeline()
        identification = lift_pairwise_identification(model)

        texts = ["Fire.", "The storm caused flooding."]
        spans_per_text = [
            [{"start": 0, "end": 4, "entity": "CAUSE"}],  # only 1 span -> gated out
            [{"start": 4, "end": 9, "entity": "CAUSE"}, {"start": 17, "end": 25, "entity": "EFFECT"}],
        ]

        results = identify_candidates(texts, spans_per_text, identification)

        assert results[0] == []
        assert len(model.calls) == 1
        assert len(model.calls[0]) == 2  # only the second text's 2 ordered pairs

    def test_remaps_entity_ids_back_to_span_text(self):
        model = FakePairwiseIdentificationPipeline(relation="Causal")
        identification = lift_pairwise_identification(model)

        text = "The storm caused flooding."
        spans = [{"start": 4, "end": 9}, {"start": 17, "end": 25}]

        results = identify_candidates([text], [spans], identification)

        assert len(results) == 1
        assert {r["e1"] for r in results[0]} <= {"storm", "flooding"}
        assert {r["e2"] for r in results[0]} <= {"storm", "flooding"}
        assert all(r["relation"] == "Causal" for r in results[0])
