"""SciteTokenizer — word/character tokenisation and BIO label alignment for SCITE.

Responsibilities
----------------
* Regex-based word tokenisation that preserves ``<eN>``/``</eN>`` entity markers during
  processing and strips them from the final output.
* Vocabulary construction from a processed corpus.
* Serialisation/deserialisation of the vocabulary.
* Encoding: ``text + causal relations → word_ids, char_ids, BIO labels, attention_mask``.
* BERT sub-word tokenisation (optional): when a HuggingFace BERT tokeniser is attached via
  :meth:`set_bert_tokenizer`, :meth:`encode` additionally returns ``bert_input_ids``,
  ``bert_attention_mask``, and ``bert_token_to_word`` — the alignment tensor the model
  uses to average sub-word hidden states back to word level.  No BERT tokenisation ever
  takes place inside the model itself.

Out of scope
------------
* Generating Flair or BERT contextual embeddings — that is the model's concern.
* Padding to a fixed batch length — use the accompanying data collator for that.
"""

import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------
_NUM_PAT = r"(?:[0-9]|1[0-9]|20)"
_MARKER_RE = re.compile(rf"</?e{_NUM_PAT}>")
_TOKEN_RE = re.compile(
    rf"(<e{_NUM_PAT}>|</e{_NUM_PAT}>)|([\w'-]+)|([,;:])|([^ \r\f\v])"
)
_STRIP_CHARS = frozenset("!#$&*+/<=>?@[\\]^_`{|}~\t\n.%")
_PAREN_RE = re.compile(r"\((.*?)\)")

# BIO label schema (integer ids)
LABEL_MAP: dict[str, int] = {
    "O": 0,
    "B-C": 1,
    "I-C": 2,
    "B-E": 3,
    "I-E": 4,
    "B-Emb": 5,
    "I-Emb": 6,
}


def token_char_offsets(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    """Best-effort original-text character span for each token.

    ``tokens`` is the cleaned output of :meth:`SciteTokenizer.tokenize` (or
    the ``"tokens"`` field of :meth:`SciteTokenizer.encode`) — entity
    markers already removed, parenthesised content already removed, and
    per-token punctuation already stripped. Neither removal step is
    reversible in general, so offsets are reconstructed by searching for
    each token as a substring of ``text``, scanning forward from the end
    of the previous match (tokens are, by construction, disjoint
    substrings of ``text`` in left-to-right order, since parenthesised
    spans are only ever skipped over, never reordered). Used to score
    SCITE's span predictions with the same character-span metrics as
    every other extraction approach (see ``evaluation.metrics.span_metrics``
    in conf-causality-repro) instead of only SCITE's own token-level ones.
    """
    offsets: list[tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        idx = text.find(tok, pos)
        if idx == -1:
            idx = pos
        offsets.append((idx, idx + len(tok)))
        pos = idx + len(tok)
    return offsets


class SciteTokenizer:
    """Tokeniser for SCITE.

    Usage — training
    ----------------
    .. code-block:: python

        tok = SciteTokenizer()
        processed = [tok.encode(ex["text"], relations=ex["relations"]) for ex in train]
        tok.build_vocab(processed)
        tok.save_vocab("checkpoints/scite-vocab")

    Usage — inference
    -----------------
    .. code-block:: python

        tok = SciteTokenizer()
        tok.load_vocab("checkpoints/scite-vocab")
        inputs = tok.encode("The <e1>storm</e1> caused <e2>floods</e2>.",
                            causes=["e1"], effects=["e2"])
    """

    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    # Fixed IDs for special tokens (must agree with SCITEConfig.word_vocab_size bookkeeping)
    PAD_WORD_ID = 0
    UNK_WORD_ID = 1
    PAD_CHAR_ID = 0
    UNK_CHAR_ID = 1

    def __init__(self, max_wlen: int = 58, max_clen: int = 23) -> None:
        self.max_wlen = max_wlen
        self.max_clen = max_clen
        self.word2index: dict[str, int] = {}
        self.char2index: dict[str, int] = {}
        self.word_vocab_size: int = 0
        self.char_vocab_size: int = 0
        self._bert_tokenizer = None  # set via set_bert_tokenizer()

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True after :meth:`build_vocab` or :meth:`load_vocab` has been called."""
        return bool(self.word2index and self.char2index)

    def build_vocab(self, examples: list[dict]) -> None:
        """Build word and character vocabularies from encoded examples.

        ``examples`` is a list of dicts as returned by :meth:`encode`; each must contain
        a ``"tokens"`` key (list of str, already stripped of entity markers).
        Call this *after* encoding the full training corpus so that all tokens are seen.
        """
        word_counts: Counter = Counter()
        char_counts: Counter = Counter()
        for ex in examples:
            word_counts.update(ex["tokens"])
            for tok in ex["tokens"]:
                char_counts.update(tok)

        # Words: index 0 = PAD, 1 = UNK, then by frequency descending
        vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        self.word2index = {self.PAD_TOKEN: self.PAD_WORD_ID, self.UNK_TOKEN: self.UNK_WORD_ID}
        self.word2index.update({w: i + 2 for i, w in enumerate(vocab)})
        self.word_vocab_size = len(self.word2index)

        # Characters: index 0 = PAD, 1 = UNK, then by frequency descending
        c_vocab = sorted(char_counts, key=char_counts.get, reverse=True)
        self.char2index = {self.PAD_TOKEN: self.PAD_CHAR_ID, self.UNK_TOKEN: self.UNK_CHAR_ID}
        for i, ch in enumerate(c_vocab):
            if ch not in self.char2index:
                self.char2index[ch] = i + 2
        self.char_vocab_size = len(self.char2index)

    def save_vocab(self, directory: Path | str) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        pickle.dump(self.word2index, open(directory / "word2index.pkl", "wb"))
        pickle.dump(self.char2index, open(directory / "char2index.pkl", "wb"))

    def load_vocab(self, directory: Path | str) -> None:
        directory = Path(directory)
        self.word2index = pickle.load(open(directory / "word2index.pkl", "rb"))
        self.char2index = pickle.load(open(directory / "char2index.pkl", "rb"))
        self.word_vocab_size = len(self.word2index)
        self.char_vocab_size = len(self.char2index)

    # ------------------------------------------------------------------
    # BERT tokeniser attachment (optional)
    # ------------------------------------------------------------------

    def set_bert_tokenizer(self, bert_tokenizer) -> None:
        """Attach a HuggingFace tokeniser for BERT embedding mode.

        When set, :meth:`encode` additionally returns ``bert_input_ids``,
        ``bert_attention_mask``, and ``bert_token_to_word`` so that the model can
        generate BERT contextual embeddings without calling any tokeniser internally.
        """
        self._bert_tokenizer = bert_tokenizer

    # ------------------------------------------------------------------
    # Core tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """Tokenise ``text`` into word tokens, preserving ``<eN>``/``</eN>`` markers.

        Parenthesized content is removed first, as in the official SCITE
        preprocessing (data_prep.ipynb: ``re.sub('\\((.*?)\\)', '', s)``);
        entity markers never appear inside parentheses in the corpus.
        Punctuation characters in :data:`_STRIP_CHARS` are removed from non-marker tokens.
        Markers are returned verbatim so that :meth:`encode` can use them for label
        alignment before stripping them from the final output.
        """
        text = _PAREN_RE.sub("", text)
        raw = [item for group in _TOKEN_RE.findall(text) for item in group if item]
        tokens: list[str] = []
        for tok in raw:
            if _MARKER_RE.fullmatch(tok):
                tokens.append(tok)
            else:
                cleaned = "".join(c for c in tok if c not in _STRIP_CHARS)
                if cleaned:
                    tokens.append(cleaned)
        return tokens

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        relations: Optional[list] = None,
        causes: Optional[list[str]] = None,
        effects: Optional[list[str]] = None,
    ) -> dict:
        """Encode a sentence with optional causal relations into model inputs.

        Parameters
        ----------
        text:
            Raw sentence, optionally containing ``<eN>…</eN>`` entity markers.
        relations:
            List of relation dicts with keys ``"first"``, ``"second"``, and
            ``"relationship"`` (1 = causal).  Causal relations are extracted;
            other values are ignored.  Mutually exclusive with ``causes``/``effects``.
        causes:
            List of cause entity IDs, e.g. ``["e1", "e3"]``.
        effects:
            Matching list of effect entity IDs.

        Returns
        -------
        dict with keys:

        ``tokens`` (list[str])
            Cleaned word tokens (entity markers removed), length ≤ ``max_wlen``.
        ``word_ids`` (list[int])
            Vocabulary indices.
        ``char_ids`` (list[list[int]])
            Character indices per token, each padded/truncated to ``max_clen``.
        ``labels`` (list[int])
            BIO integer labels (see :data:`LABEL_MAP`).
        ``attention_mask`` (list[int])
            1 for every real token.
        ``causal_relation_token_spans`` (list)
            ``[[cause_token_indices, effect_token_indices], …]`` for each relation.
        ``offsets`` (list[tuple[int, int]])
            Best-effort original-``text`` character span per token, same
            length and order as ``tokens`` — see :func:`token_char_offsets`.

        If a BERT tokeniser is attached (via :meth:`set_bert_tokenizer`), also:

        ``bert_input_ids`` (list[int])
        ``bert_attention_mask`` (list[int])
        ``bert_token_to_word`` (list[int])
            Maps each BERT sub-word token position to the corresponding word index in
            ``tokens`` (−1 for special tokens such as [CLS] and [SEP]).
        """
        if not self.is_ready:
            raise RuntimeError("Vocabulary not initialised — call build_vocab() or load_vocab() first.")

        # Resolve cause/effect lists from whichever representation was supplied
        if relations is not None and causes is None and effects is None:
            causes = [r["first"] for r in relations if r.get("relationship", 1) == 1]
            effects = [r["second"] for r in relations if r.get("relationship", 1) == 1]
        causes = list(causes or [])
        effects = list(effects or [])

        # --- Tokenise (entity markers preserved) ---
        raw_tokens = self.tokenize(text)

        # --- Build entity role map ---
        entity_roles: dict[str, dict[str, bool]] = {}
        for e_id in causes + effects:
            entity_roles.setdefault(e_id, {"is_cause": False, "is_effect": False})
        for e_id in causes:
            entity_roles[e_id]["is_cause"] = True
        for e_id in effects:
            entity_roles[e_id]["is_effect"] = True
        for role in entity_roles.values():
            role["is_embedded"] = role["is_cause"] and role["is_effect"]

        # --- Assign BIO labels to raw token positions ---
        raw_labels = [0] * len(raw_tokens)
        for e_id, role in entity_roles.items():
            open_tag, close_tag = f"<{e_id}>", f"</{e_id}>"
            try:
                start = raw_tokens.index(open_tag)
                end = raw_tokens.index(close_tag)
            except ValueError:
                continue
            base = 5 if role["is_embedded"] else 3 if role["is_effect"] else 1
            # First content token → B-*, rest → I-*
            for i in range(start + 1, end):
                raw_labels[i] = base if i == start + 1 else base + 1

        # --- Strip entity markers and build cleaned token list ---
        tokens: list[str] = []
        labels: list[int] = []
        orig_to_clean: dict[int, int] = {}
        for i, (tok, lbl) in enumerate(zip(raw_tokens, raw_labels)):
            if _MARKER_RE.fullmatch(tok):
                continue
            orig_to_clean[i] = len(tokens)
            tokens.append(tok)
            labels.append(lbl)

        # --- Causal relation token spans (in cleaned-token space) ---
        causal_relation_token_spans: list = []
        for c_id, e_id in zip(causes, effects):
            def _span(eid: str) -> Optional[list[int]]:
                try:
                    s = raw_tokens.index(f"<{eid}>")
                    en = raw_tokens.index(f"</{eid}>")
                except ValueError:
                    return None
                idx = [orig_to_clean[i] for i in range(s + 1, en) if i in orig_to_clean]
                return idx if idx else None

            c_span, e_span = _span(c_id), _span(e_id)
            if c_span and e_span:
                causal_relation_token_spans.append([c_span, e_span])

        # --- Truncate ---
        tokens = tokens[: self.max_wlen]
        labels = labels[: self.max_wlen]

        # Drop any relation whose cause/effect span extends past the
        # truncation point — its indices are no longer valid positions in
        # the truncated sequence, and it is no longer a gold relation this
        # example can be evaluated against.  Sentences longer than
        # max_wlen=58 (SCITE's own paper-specified limit, comfortably
        # covering its own dataset) are common on other, longer-sentence
        # corpora, so this must be filtered rather than left to crash the
        # first time an out-of-range index is looked up during evaluation.
        causal_relation_token_spans = [
            [c_span, e_span]
            for c_span, e_span in causal_relation_token_spans
            if max(c_span) < len(tokens) and max(e_span) < len(tokens)
        ]

        # --- Convert to ids ---
        word_ids = [self.word2index.get(t, self.UNK_WORD_ID) for t in tokens]
        char_ids = [self._chars_to_ids(t) for t in tokens]
        attention_mask = [1] * len(tokens)

        result = {
            "tokens": tokens,
            "word_ids": word_ids,
            "char_ids": char_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "causal_relation_token_spans": causal_relation_token_spans,
            "offsets": token_char_offsets(text, tokens),
        }

        # --- Optional BERT sub-word tokenisation ---
        if self._bert_tokenizer is not None:
            bert_enc = self._bert_tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors=None,
                truncation=True,
                max_length=self._bert_tokenizer.model_max_length,
            )
            # bert_token_to_word: -1 for special tokens, word index otherwise
            raw_word_ids = bert_enc.word_ids()
            bert_token_to_word = [-1 if wi is None else wi for wi in raw_word_ids]
            result["bert_input_ids"] = bert_enc["input_ids"]
            result["bert_attention_mask"] = bert_enc["attention_mask"]
            result["bert_token_to_word"] = bert_token_to_word

        return result

    def __call__(self, text: str, **kwargs) -> dict:
        return self.encode(text, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chars_to_ids(self, word: str) -> list[int]:
        ids = [self.char2index.get(c, self.UNK_CHAR_ID) for c in word[: self.max_clen]]
        ids += [self.PAD_CHAR_ID] * (self.max_clen - len(ids))
        return ids
