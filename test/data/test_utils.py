"""Tests for causalatee.data.utils."""

from __future__ import annotations

import re

from causalatee.data.utils import (
    extraction_batch_to_sentences,
    identification_batch_to_detection,
    identification_batch_to_detection_sentences,
    identification_batch_to_extraction,
    identification_batch_to_sentences,
    insert_entity_markers,
    parse_entity_markers,
    split_document_by_sentence,
    split_extraction_to_sentences,
    split_identification_to_sentences,
)


class TestMarkerRoundtrip:
    def test_simple_pair(self):
        text = "<e1>X</e1> caused <e2>Y</e2>."
        clean, spans = parse_entity_markers(text)
        assert clean == "X caused Y."
        assert spans == {"e1": [(0, 1)], "e2": [(9, 10)]}
        assert insert_entity_markers(clean, spans) == text

    def test_discontinuous_entity_two_occurrences(self):
        text = "<e1>Turn</e1> the engine <e1>on</e1> now."
        clean, spans = parse_entity_markers(text)
        assert clean == "Turn the engine on now."
        assert spans["e1"] == [(0, 4), (16, 18)]
        assert insert_entity_markers(clean, spans) == text

    def test_nested_spans_roundtrip(self):
        # e2 fully inside e1 -- exercises the stack-based (not naive
        # single-position) insertion order.
        text = "<e1>the <e2>big</e2> dog</e1> barked"
        clean, spans = parse_entity_markers(text)
        assert clean == "the big dog barked"
        assert insert_entity_markers(clean, spans) == text

    def test_adjacent_spans_close_then_open_correctly(self):
        # e1 ends exactly where e2 begins -- closing must come before
        # opening at a shared boundary, or the tags interleave.
        clean = "AB"
        spans = {"e1": [(0, 1)], "e2": [(1, 2)]}
        marked = insert_entity_markers(clean, spans)
        assert marked == "<e1>A</e1><e2>B</e2>"
        assert parse_entity_markers(marked) == (clean, spans)

    def test_no_markers(self):
        assert parse_entity_markers("plain text") == ("plain text", {})

    def test_crossing_spans_roundtrip_even_though_not_valid_xml(self):
        # e1=[0,24) and e2=[13,39) genuinely cross -- neither contains the
        # other. This is NOT well-nested XML/HTML (an XML parser could not
        # tell that </e1> closes <e1> rather than the more-recently-opened
        # <e2>), but these markers aren't XML: parse_entity_markers matches
        # each </eN> to its own id's open position, not to a shared nesting
        # stack, so crossing spans round-trip correctly regardless. This
        # comes up for real on CNC/BECauSEv2/AltLex, whose source
        # annotations have genuinely crossing argument spans across
        # different relations in the same sentence.
        text = "demanding the arrest of a jawan who tried to molest her"
        segments = {"e1": [(0, 24)], "e2": [(13, 39)]}
        marked = insert_entity_markers(text, segments)
        # Not well-nested: e2 opens before e1 closes, but e1 closes before e2 does.
        assert marked.index("<e2>") < marked.index("</e1>") < marked.index("</e2>")
        clean, recovered = parse_entity_markers(marked)
        assert clean == text
        assert recovered == segments

    def test_three_way_crossing_spans_roundtrip(self):
        # e1, e2, e3 pairwise overlap in a staggered, non-nested chain.
        text = "0123456789"
        segments = {"e1": [(0, 5)], "e2": [(2, 7)], "e3": [(4, 9)]}
        marked = insert_entity_markers(text, segments)
        clean, recovered = parse_entity_markers(marked)
        assert clean == text
        assert recovered == segments


class TestSplitDocumentBySentence:
    SENT_RANGES = [(0, 10), (10, 20)]  # two adjacent 10-char sentences

    def test_group_wholly_inside_one_sentence(self):
        entity_segments = {"a": [(1, 3)], "b": [(5, 8)]}
        rows, dropped = split_document_by_sentence(self.SENT_RANGES, entity_segments, [("a", "b")])
        assert dropped == 0
        assert rows[0]["group_indices"] == [0]
        assert rows[0]["entities"] == {"a": [(1, 3)], "b": [(5, 8)]}
        assert rows[1]["group_indices"] == []

    def test_group_straddling_sentence_boundary_is_dropped(self):
        # "a" lands in sentence 0, "b" in sentence 1 -- can't be kept together.
        entity_segments = {"a": [(1, 3)], "b": [(15, 18)]}
        rows, dropped = split_document_by_sentence(self.SENT_RANGES, entity_segments, [("a", "b")])
        assert dropped == 1
        assert rows[0]["group_indices"] == []
        assert rows[1]["group_indices"] == []

    def test_single_span_straddling_boundary_is_dropped(self):
        # one entity whose own span crosses the boundary, e.g. (8, 15).
        entity_segments = {"a": [(8, 15)]}
        rows, dropped = split_document_by_sentence(self.SENT_RANGES, entity_segments, [("a",)])
        assert dropped == 1

    def test_offsets_are_rebased_local_to_sentence(self):
        entity_segments = {"a": [(12, 14)]}
        rows, dropped = split_document_by_sentence(self.SENT_RANGES, entity_segments, [("a",)])
        assert dropped == 0
        assert rows[1]["entities"]["a"] == [(2, 4)]

    def test_rows_align_with_sentence_ranges_even_when_empty(self):
        rows, dropped = split_document_by_sentence(self.SENT_RANGES, {}, [])
        assert len(rows) == 2
        assert all(r["group_indices"] == [] for r in rows)

    def test_position_outside_any_sentence_range_is_dropped(self):
        entity_segments = {"a": [(25, 27)]}
        rows, dropped = split_document_by_sentence(self.SENT_RANGES, entity_segments, [("a",)])
        assert dropped == 1


class TestSplitIdentificationToSentences:
    def test_within_sentence_relation_survives(self):
        text = "<e1>X</e1> caused <e2>Y</e2>. Another sentence here."
        relations = [{"relationship": 1, "first": "e1", "second": "e2"}]
        # sent_ranges are offsets into the marker-STRIPPED text: "X caused Y."
        # is 11 chars, then a space, then "Another sentence here." (23 chars).
        clean_text, _ = parse_entity_markers(text)
        boundary = clean_text.index(". ") + 1
        sent_ranges = [(0, boundary), (boundary + 1, len(clean_text))]
        rows, dropped = split_identification_to_sentences(text, relations, sent_ranges)
        assert dropped == 0
        assert rows[0]["text"] == "<e1>X</e1> caused <e2>Y</e2>."
        assert rows[0]["relations"] == relations
        assert rows[1]["relations"] == []

    def test_cross_sentence_relation_is_dropped_not_corrupted(self):
        # Regression test for the BioCause bug: an effect span belonging to
        # the NEXT sentence must never be silently re-based onto the wrong
        # sentence's text (previously observed to land mid-word).
        text = "<e1>Histidine</e1> requires biosynthesis. This <e2>substitution</e2> decreases activity."
        relations = [{"relationship": 1, "first": "e1", "second": "e2"}]
        boundary = text.index(". ") + 2
        sent_ranges = [(0, boundary), (boundary, len(text))]
        rows, dropped = split_identification_to_sentences(text, relations, sent_ranges)
        assert dropped == 1
        assert all(r["relations"] == [] for r in rows)
        # Neither half's text is corrupted by a stray/misplaced tag.
        assert "<e1>" not in rows[1]["text"] and "</e1>" not in rows[1]["text"]
        assert "<e2>" not in rows[0]["text"] and "</e2>" not in rows[0]["text"]
        assert "h<e" not in (rows[0]["text"] + rows[1]["text"]).lower().replace("histidine", "")

    def test_entity_without_any_relation_is_dropped_from_output(self):
        # entities never referenced by a surviving relation don't leak in.
        text = "<e1>X</e1> caused <e2>Y</e2>."
        rows, dropped = split_identification_to_sentences(text, [], [(0, len(text))])
        assert dropped == 0
        assert rows[0]["relations"] == []
        assert "<e1>" not in rows[0]["text"]


class TestSplitExtractionToSentences:
    def test_entities_assigned_to_their_sentence_with_local_offsets(self):
        text = "The fire spread. Rain stopped it."
        boundary = text.index(". ") + 1
        sent_ranges = [(0, boundary), (boundary + 1, len(text))]
        entities = [[4, 8], [17, 21]]  # "fire" in sent 0, "Rain" in sent 1
        rows, dropped = split_extraction_to_sentences(text, entities, sent_ranges)
        assert dropped == 0
        assert rows[0]["entity"] == [[4, 8]]
        assert rows[0]["text"] == "The fire spread."
        assert rows[1]["entity"] == [[0, 4]]
        assert rows[1]["text"] == "Rain stopped it."

    def test_discontinuous_entity_kept_only_if_all_segments_in_one_sentence(self):
        text = "Turn the engine on now. Done."
        sent_ranges = [(0, 24), (24, len(text))]
        entities = [[0, 4, 16, 18]]  # "Turn" ... "on" -- both in sentence 0
        rows, dropped = split_extraction_to_sentences(text, entities, sent_ranges)
        assert dropped == 0
        assert rows[0]["entity"] == [[0, 4, 16, 18]]
        assert rows[1]["entity"] == []

    def test_discontinuous_entity_straddling_sentences_is_dropped(self):
        text = "Turn the engine on. Now do it."
        boundary = text.index(". ") + 2
        sent_ranges = [(0, boundary), (boundary, len(text))]
        entities = [[0, 4, boundary + 4, boundary + 6]]  # segment 2 in sentence 1
        rows, dropped = split_extraction_to_sentences(text, entities, sent_ranges)
        assert dropped == 1
        assert rows[0]["entity"] == []
        assert rows[1]["entity"] == []


def _split_on_period_space(text: str) -> list[tuple[int, int]]:
    """Toy segmenter for the batch-adapter tests: cuts after each ". ",
    keeping the period on the left side -- stands in for a real sentence
    segmenter (spaCy et al.) so these tests don't need one installed."""
    ranges = []
    start = 0
    for m in re.finditer(r"\. ", text):
        ranges.append((start, m.start() + 1))
        start = m.end()
    ranges.append((start, len(text)))
    return ranges


class TestIdentificationBatchToSentences:
    def test_flattens_across_the_whole_batch(self):
        batch = {
            "text": [
                "<e1>X</e1> caused <e2>Y</e2>. Another sentence.",
                "<e3>A</e3> triggers <e4>B</e4>.",
            ],
            "relations": [
                [{"relationship": 1, "first": "e1", "second": "e2"}],
                [{"relationship": 1, "first": "e3", "second": "e4"}],
            ],
        }
        out = identification_batch_to_sentences(batch, _split_on_period_space)
        # doc 0 -> 2 sentence rows, doc 1 -> 1 sentence row.
        assert len(out["text"]) == 3
        assert len(out["relations"]) == 3
        assert out["relations"][0] == batch["relations"][0]
        assert out["relations"][1] == []
        assert out["relations"][2] == batch["relations"][1]

    def test_cross_sentence_relation_dropped_across_batch(self):
        batch = {
            "text": ["<e1>X</e1> needs Y. This <e2>substitution</e2> helps."],
            "relations": [[{"relationship": 1, "first": "e1", "second": "e2"}]],
        }
        out = identification_batch_to_sentences(batch, _split_on_period_space)
        assert all(rel == [] for rel in out["relations"])


class TestExtractionBatchToSentences:
    def test_flattens_across_the_whole_batch(self):
        batch = {
            "text": ["The fire spread. It was hot.", "Rain fell. It helped."],
            "entity": [[[4, 8]], [[0, 4]]],
        }
        out = extraction_batch_to_sentences(batch, _split_on_period_space)
        assert len(out["text"]) == 4
        assert out["entity"] == [[[4, 8]], [], [[0, 4]], []]


class TestIdentificationBatchToDetectionSentences:
    def test_sentence_with_any_entity_is_causal_even_without_a_relation(self):
        # e2 is an unpaired (effect-only, no Cause) entity -- no relation
        # references it, but detection should still count it as causal.
        batch = {"text": ["<e1>X</e1> caused Y. This <e2>effect</e2> remained unlinked."]}
        out = identification_batch_to_detection_sentences(batch, _split_on_period_space)
        assert out["label"] == [1, 1]

    def test_sentence_with_no_entities_is_uncausal(self):
        batch = {"text": ["Nothing marked here at all."]}
        out = identification_batch_to_detection_sentences(batch, _split_on_period_space)
        assert out["label"] == [0]


class TestIdentificationBatchToDetection:
    def test_row_with_a_relation_is_causal(self):
        batch = {
            "text": ["<e1>X</e1> caused <e2>Y</e2>."],
            "relations": [[{"relationship": 1, "first": "e1", "second": "e2"}]],
        }
        out = identification_batch_to_detection(batch)
        assert out["text"] == ["X caused Y."]
        assert out["label"] == [1]

    def test_row_with_marked_entities_but_no_relation_is_uncausal(self):
        # Unlike the _sentences variant, mere entity presence isn't enough --
        # this is the behaviour datasets like SemEval-2007/2020 need, where a
        # "false"/no-consequent row still has both spans marked.
        batch = {
            "text": ["<e1>X</e1> happened near <e2>Y</e2>."],
            "relations": [[]],
        }
        out = identification_batch_to_detection(batch)
        assert out["label"] == [0]

    def test_row_with_no_entities_is_uncausal(self):
        batch = {"text": ["Nothing marked here."], "relations": [[]]}
        out = identification_batch_to_detection(batch)
        assert out["label"] == [0]

    def test_no_sentence_splitting_multi_sentence_row_stays_whole(self):
        batch = {
            "text": ["<e1>X</e1> caused <e2>Y</e2>. A second sentence, unrelated."],
            "relations": [[{"relationship": 1, "first": "e1", "second": "e2"}]],
        }
        out = identification_batch_to_detection(batch)
        assert out["text"] == ["X caused Y. A second sentence, unrelated."]
        assert out["label"] == [1]

    def test_explicit_norelation_entry_is_not_causal(self):
        # Regression test: some converters (UniCausal2HF-based ones, e.g.
        # CTB, SemEval2010T8) explicitly list a relationship=0 (NoRelation)
        # entry for every non-causal candidate pair rather than omitting it
        # -- verified directly against real CTB data (3047 NoRelation vs.
        # 270 Causal entries). A merely non-empty relations list is NOT
        # a valid causal check for those.
        batch = {
            "text": ["<e1>X</e1> happened near <e2>Y</e2>."],
            "relations": [[{"relationship": 0, "first": "e1", "second": "e2"}]],
        }
        out = identification_batch_to_detection(batch)
        assert out["label"] == [0]

    def test_mix_of_norelation_and_causal_is_causal(self):
        batch = {
            "text": ["<e1>X</e1>, <e2>Y</e2>, and <e3>Z</e3> all happened."],
            "relations": [
                [
                    {"relationship": 0, "first": "e1", "second": "e2"},
                    {"relationship": 1, "first": "e2", "second": "e3"},
                ]
            ],
        }
        out = identification_batch_to_detection(batch)
        assert out["label"] == [1]


class TestIdentificationBatchToExtraction:
    def test_relation_entities_are_extracted_with_markers_stripped(self):
        batch = {
            "text": ["<e1>X</e1> caused <e2>Y</e2>."],
            "relations": [[{"relationship": 1, "first": "e1", "second": "e2"}]],
        }
        out = identification_batch_to_extraction(batch)
        assert out["text"] == ["X caused Y."]
        assert out["entity"] == [[[0, 1], [9, 10]]]

    def test_row_with_no_relations_is_dropped_from_output(self):
        batch = {
            "text": ["<e1>X</e1> happened near <e2>Y</e2>.", "<e3>A</e3> caused <e4>B</e4>."],
            "relations": [[], [{"relationship": 1, "first": "e3", "second": "e4"}]],
        }
        out = identification_batch_to_extraction(batch)
        assert out["text"] == ["A caused B."]
        assert len(out["entity"]) == 1

    def test_entity_unreferenced_by_any_relation_is_excluded(self):
        # e3 is an unpaired (effect-only) entity like BioCause's -- present
        # in the text but not part of any relation, so not extracted.
        batch = {
            "text": ["<e1>X</e1> caused <e2>Y</e2>, and <e3>Z</e3> happened too."],
            "relations": [[{"relationship": 1, "first": "e1", "second": "e2"}]],
        }
        out = identification_batch_to_extraction(batch)
        assert out["entity"] == [[[0, 1], [9, 10]]]

    def test_discontinuous_entity_flattened_to_one_list(self):
        batch = {
            "text": ["<e1>Turn</e1> the engine <e1>on</e1>, which <e2>started it</e2>."],
            "relations": [[{"relationship": 1, "first": "e1", "second": "e2"}]],
        }
        out = identification_batch_to_extraction(batch)
        clean, segments = parse_entity_markers(batch["text"][0])
        expected_e1 = [x for seg in segments["e1"] for x in seg]
        assert out["entity"][0][0] == expected_e1

    def test_row_with_only_norelation_entries_is_dropped(self):
        # Regression test for the same NoRelation-not-omitted issue as
        # TestIdentificationBatchToDetection -- a row whose only relation
        # entries are relationship=0 has nothing causal to extract.
        batch = {
            "text": ["<e1>X</e1> happened near <e2>Y</e2>."],
            "relations": [[{"relationship": 0, "first": "e1", "second": "e2"}]],
        }
        out = identification_batch_to_extraction(batch)
        assert out["text"] == []
        assert out["entity"] == []

    def test_norelation_entity_excluded_even_alongside_a_real_relation(self):
        batch = {
            "text": ["<e1>X</e1>, <e2>Y</e2>, and <e3>Z</e3> all happened."],
            "relations": [
                [
                    {"relationship": 0, "first": "e1", "second": "e3"},
                    {"relationship": 1, "first": "e2", "second": "e3"},
                ]
            ],
        }
        out = identification_batch_to_extraction(batch)
        # e1 is only ever referenced by the NoRelation entry -- excluded.
        assert len(out["entity"]) == 1
        clean, segments = parse_entity_markers(batch["text"][0])
        e2_flat = [x for seg in segments["e2"] for x in seg]
        e3_flat = [x for seg in segments["e3"] for x in seg]
        assert sorted(out["entity"][0]) == sorted([e2_flat, e3_flat])
