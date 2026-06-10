import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Union

import pandas as pd

from ..constants import ClassLabel, Relation, Task
from ._converter import FormatConverter


def _extract_entities_and_relations(row) -> tuple:
    """
    Converts ["<ARG0>Bla</Arg0> bla <ARG1>Bla</ARG1>", "Bla <ARG0>bla</ARG0> <ARG1>Bla</ARG1>"] to
    ["Bla", " ", "bla", " ", "Bla"], [[1], [], [2], [], [3]], {relationship: 1, "first": 0, "second": 1}
    """
    text, withpairs = row["text"], row["causal_text_w_pairs"]
    # Step 1: Remove all <SIG> tags
    withpairs = [re.sub(r"<(/?)SIG(\d+)>", "", t) for t in withpairs]
    # Step 2: Check that all texts are the same if we remove the ARG tags
    without_tags = [re.sub(r"<(/?)ARG(\d+)>", "", t) for t in withpairs]
    # All texts without tags must be equal to text
    assert all(x == text for x in without_tags)
    # Step 3: Iteratively update the tags
    splits: list[str] = [text]
    tags: list[list[int]] = [[]]
    relations: list[tuple[int, int, int]] = []

    def split_at_charidx(idx: int) -> int:
        for i, s in enumerate(splits):
            if idx < len(s):
                if idx == 0:  # We are already at the beginning of a str; no need to split
                    return i
                else:  # Need to split
                    # Concurrent editing an iterating the list; OK here since we return right after
                    splits.insert(i + 1, s[idx:])
                    splits[i] = s[:idx]
                    tags.insert(i + 1, tags[i].copy())
                    return i + 1
            idx -= len(s)
        return len(splits)

    def minify(tags: list[list[int]]) -> tuple[list[list[int]], dict[int, int]]:
        """
        Joins entities that denote the same spans.
        Maps, e.g., [[0], [], [1, 2], [], [3], []] to [[0], [], [1], [], [2], []]
        """
        tag2pos = defaultdict(list)  # Maps tags to the positions they occur in
        for idx, lst in enumerate(tags):
            for x in lst:
                tag2pos[x].append(idx)
        pos2tag = defaultdict(list)
        for tag, pos in tag2pos.items():
            pos2tag[tuple(pos)].append(tag)
        newtags = [[] for _ in range(len(tags))]
        tagmap: dict[int, int] = dict()
        for i, (pos, tags) in enumerate(pos2tag.items()):
            for t in tags:
                tagmap[t] = i
            for p in pos:
                newtags[p].append(i)
        return newtags, tagmap

    nexttag: int = 0
    for t in withpairs:
        curtags: set[int] = set()
        offset: int = 0
        tagmap: dict[int, int] = dict()
        for match in re.finditer(r"(.*?)<(/?)ARG(\d+)>", t):
            # Put the text span from offset to offset+len(match[1]) into a new entity and set the entity label
            # appropriately; should do nothing if the entity is already separated.
            # Will never split but we can use it to get the index
            startidx = split_at_charidx(offset)
            stopidx = split_at_charidx(offset + len(match[1]))
            for i in range(startidx, stopidx):
                tags[i].extend(curtags)
            if match[2] == "":
                if int(match[3]) not in tagmap:
                    tagmap[int(match[3])] = nexttag
                    nexttag += 1
                curtags.add(tagmap[int(match[3])])
            else:
                curtags.remove(tagmap[int(match[3])])
            offset += len(match[1])
        # Each entry in withpairs contains exactly one cause (ARG0) and effect (ARG1)
        relations.append((Relation.Procausal, tagmap[0], tagmap[1]))
    tags, tagmap = minify(tags)
    for i in range(len(relations)):
        relations[i] = (relations[i][0], tagmap[relations[i][1]], tagmap[relations[i][2]])
    return (splits, tags, relations)


class UniCausal2HF(FormatConverter):
    def __init__(self, splits: dict[str, Path], target: Path, grouped: bool = True):
        super().__init__(target)
        self._splits = splits
        self._grouped = grouped

    def _load_df(self, split: str) -> pd.DataFrame:
        """Load a split CSV and return it in grouped format.

        Grouped splits (data/grouped/splits/) already have one row per sentence
        with causal_text_w_pairs as a Python list of annotated strings.

        Regular splits (data/splits/) have one row per entity pair with
        text_w_pairs as a single annotated string and explicit seq_label /
        pair_label columns.  This method normalises the latter into the same
        grouped representation so all downstream converters stay unchanged.
        """
        if self._grouped:
            return pd.read_csv(
                self._splits[split],
                converters={"causal_text_w_pairs": lambda x: ast.literal_eval(x) if x else []},
            )
        df = pd.read_csv(self._splits[split])
        rows = []
        for (corpus, doc_id, sent_id), group in df.groupby(["corpus", "doc_id", "sent_id"], sort=False):
            rows.append(
                {
                    "index": f"{corpus}_{doc_id}_{sent_id}",
                    "text": group["text"].iloc[0],
                    "causal_text_w_pairs": group.loc[group["pair_label"] == 1, "text_w_pairs"].tolist(),
                }
            )
        return pd.DataFrame(rows)

    def _convert(self, task: str, split: str) -> pd.DataFrame:
        converter: dict[Task, Callable[[str], pd.DataFrame]] = {
            Task.CausalityDetection: self._convert_causality_detection,
            Task.CausalCandidateExtraction: self._convert_causal_candidate_extraction,
            Task.CausalityIdentification: self._convert_causality_identification,
        }
        return converter.get(task)(split)

    def _convert_causality_detection(self, split: str) -> pd.DataFrame:
        df = self._load_df(split)
        df["label"] = df["causal_text_w_pairs"].apply(
            lambda x: ClassLabel.Uncausal if len(x) == 0 else ClassLabel.Causal
        )
        return df[["label", "text", "index"]].set_index("index")

    def _convert_causal_candidate_extraction(self, split: str) -> pd.DataFrame:
        def map_list_to_tokens(row):
            splits, tags, _ = _extract_entities_and_relations(row)
            spans: dict[int, tuple[int, int]] = dict()
            offset: int = 0
            for s, ts in zip(splits, tags):
                for t in ts:
                    if t not in spans:
                        spans[t] = (offset, offset + len(s))
                    else:
                        spans[t] = (spans[t][0], offset + len(s))
                offset += len(s)
            return pd.Series(("".join(splits), list(list(s) for s in spans.values())))

        df = self._load_df(split)
        df[["text", "entity"]] = df[["text", "causal_text_w_pairs"]].apply(map_list_to_tokens, axis=1)
        return df[["index", "text", "entity"]].set_index("index")

    def _convert_causality_identification(self, split: str) -> pd.DataFrame:
        def map_to_labels(row):
            splits, tags, relations = _extract_entities_and_relations(row)
            text: str = ""
            cur_ents: set[int] = set()
            for s, t in zip(splits, tags):
                for newent in set(t) - cur_ents:
                    text += f"<e{newent + 1}>"
                for oldent in cur_ents - set(t):
                    text += f"</e{oldent + 1}>"
                cur_ents = set(t)
                text += s
            reldict: list[dict[str, Union[int, str]]] = []
            for rtype, rfirst, rsecond in relations:
                reldict.append({"relationship": rtype, "first": f"e{rfirst + 1}", "second": f"e{rsecond + 1}"})
            return pd.Series((text, reldict))

        df = self._load_df(split)
        df[["text", "relations"]] = df[["text", "causal_text_w_pairs"]].apply(map_to_labels, axis=1)
        return df[["index", "text", "relations"]].set_index("index")
