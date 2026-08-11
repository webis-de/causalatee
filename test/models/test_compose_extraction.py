"""Tests for compose_extraction (Detection + CandidateExtraction + PairwiseIdentification composed into Extraction), no
real model."""

from __future__ import annotations

from causalatee.models import Extraction, compose_extraction


class FakeDetection:
    def __init__(self, labels: dict[str, str] | None = None, default: str = "Causal"):
        self._labels = labels or {}
        self._default = default

    def __call__(self, text):
        batch = [text] if isinstance(text, str) else text
        results = [{"label": self._labels.get(t, self._default), "score": 0.9} for t in batch]
        return results[0] if isinstance(text, str) else results


class FakeCandidateExtraction:
    def __init__(self, spans_by_text: dict[str, list[dict]] | None = None, default: list[dict] | None = None):
        self._spans_by_text = spans_by_text or {}
        self._default = default if default is not None else []

    def __call__(self, text):
        batch = [text] if isinstance(text, str) else text
        results = [self._spans_by_text.get(t, self._default) for t in batch]
        return results[0] if isinstance(text, str) else results


class FakePairwiseIdentification:
    """Returns ``relation`` unconditionally, unless ``causal_direction`` is given, in which case only the pair text
    matching it is classified as ``relation`` and every other direction is classified ``NoRelation`` -- lets tests
    distinguish "both orderings get tried" (permutations, not just one direction per pair) from "the correct
    direction is picked out"."""

    def __init__(self, relation: str = "Causal", score: float = 0.95, causal_direction: str | None = None):
        self._relation = relation
        self._score = score
        self._causal_direction = causal_direction
        self.calls: list[list[str]] = []

    def __call__(self, text):
        batch = [text] if isinstance(text, str) else text
        self.calls.append(list(batch))
        results = [
            {"relation": self._relation, "score": self._score}
            if self._causal_direction is None or self._causal_direction in t
            else {"relation": "NoRelation", "score": self._score}
            for t in batch
        ]
        return results[0] if isinstance(text, str) else results


class TestComposeExtraction:
    def test_satisfies_extraction_protocol(self):
        extractor = compose_extraction(FakeDetection(), FakeCandidateExtraction(), FakePairwiseIdentification())
        assert isinstance(extractor, Extraction)

    def test_returns_empty_when_uncausal(self):
        text = "Some random sentence without causality."
        extractor = compose_extraction(
            FakeDetection({text: "Uncausal"}),
            FakeCandidateExtraction({text: [{"start": 0, "end": 4}, {"start": 5, "end": 9}]}),
            FakePairwiseIdentification(),
        )
        assert extractor(text) == []

    def test_returns_empty_when_fewer_than_two_spans(self):
        text = "Fire."
        extractor = compose_extraction(
            FakeDetection(),
            FakeCandidateExtraction({text: [{"start": 0, "end": 4}]}),
            FakePairwiseIdentification(),
        )
        assert extractor(text) == []

    def test_returns_empty_when_no_spans(self):
        extractor = compose_extraction(FakeDetection(), FakeCandidateExtraction(), FakePairwiseIdentification())
        assert extractor("Nothing here.") == []

    def test_skips_no_relation_pairs(self):
        text = "The storm caused flooding."
        extractor = compose_extraction(
            FakeDetection(),
            FakeCandidateExtraction({text: [{"start": 4, "end": 9}, {"start": 17, "end": 25}]}),
            FakePairwiseIdentification(relation="NoRelation"),
        )
        assert extractor(text) == []

    def test_returns_relation_for_causal_pair(self):
        # 2 spans -> both orderings are tried (permutations, not just text order); only the "storm first"
        # direction should survive as Causal here.
        text = "The storm caused flooding."
        extractor = compose_extraction(
            FakeDetection(),
            FakeCandidateExtraction({text: [{"start": 4, "end": 9}, {"start": 17, "end": 25}]}),
            FakePairwiseIdentification(relation="Causal", score=0.95, causal_direction="<e1>storm"),
        )
        result = extractor(text)
        assert len(result) == 1
        assert result[0]["e1"] == text[4:9]
        assert result[0]["e2"] == text[17:25]
        assert result[0]["relation"] == "Causal"
        assert abs(result[0]["score"] - 0.95) < 1e-6

    def test_produces_all_ordered_pairs_from_n_spans(self):
        text = "abc def ghi"
        spans = [{"start": 0, "end": 3}, {"start": 4, "end": 7}, {"start": 8, "end": 11}]
        extractor = compose_extraction(
            FakeDetection(),
            FakeCandidateExtraction({text: spans}),
            FakePairwiseIdentification(relation="Causal"),
        )
        result = extractor(text)
        assert len(result) == 6  # 3 spans -> 3*2 ordered pairs, all "Causal"

    def test_identification_receives_marked_text_and_is_called_once(self):
        text = "Fire smoke damage."
        identification = FakePairwiseIdentification()
        extractor = compose_extraction(
            FakeDetection(),
            FakeCandidateExtraction({text: [{"start": 0, "end": 4}, {"start": 5, "end": 10}]}),
            identification,
        )
        extractor(text)
        assert len(identification.calls) == 1
        # _LiftedPairwiseIdentification always re-marks the current pair as <e1>/<e2>, regardless of the
        # candidate-index ids (<e0>, <e1>, ...) compose_extraction used to mark the full set of candidate
        # spans upstream.
        assert all("<e1>" in t and "<e2>" in t for t in identification.calls[0])

    def test_batch_of_texts_flattens_identification_into_one_call(self):
        causal_text = "The storm caused flooding."
        uncausal_text = "She went to the store."
        identification = FakePairwiseIdentification()
        extractor = compose_extraction(
            FakeDetection({uncausal_text: "Uncausal"}),
            FakeCandidateExtraction({causal_text: [{"start": 4, "end": 9}, {"start": 17, "end": 25}]}),
            identification,
        )
        results = extractor([causal_text, uncausal_text])
        assert len(results) == 2
        assert results[1] == []  # uncausal text contributes nothing
        assert len(results[0]) == 2  # 1 pair (e0,e1) x 2 permutations
        assert len(identification.calls) == 1  # only the causal text's pairs were ever sent
