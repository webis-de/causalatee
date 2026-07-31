import pickle
from collections import Counter
from pathlib import Path
from typing import Optional

NOUN_SUPERSENSES: list[str] = [
    "noun.Tops",
    "noun.act",
    "noun.animal",
    "noun.artifact",
    "noun.attribute",
    "noun.body",
    "noun.cognition",
    "noun.communication",
    "noun.event",
    "noun.feeling",
    "noun.food",
    "noun.group",
    "noun.location",
    "noun.motive",
    "noun.object",
    "noun.person",
    "noun.phenomenon",
    "noun.plant",
    "noun.possession",
    "noun.process",
    "noun.quantity",
    "noun.relation",
    "noun.shape",
    "noun.state",
    "noun.substance",
    "noun.time",
]

VERB_SUPERSENSES: list[str] = [
    "verb.body",
    "verb.change",
    "verb.cognition",
    "verb.communication",
    "verb.competition",
    "verb.consumption",
    "verb.contact",
    "verb.creation",
    "verb.emotion",
    "verb.motion",
    "verb.perception",
    "verb.possession",
    "verb.social",
    "verb.stative",
    "verb.weather",
]

CAUSAL_FRAMES: set[str] = {
    # Paper (Li & Mao 2019, Algorithm 1 / Section 3.1.2): "40 causal frames
    # from FrameNet including 'Causation', 'Causation Scenario',
    # 'Triggering', 'Reason', 'Explaining the facts', 'Response' as well as
    # 34 frames start with 'Cause'." The installed FrameNet corpus (nltk)
    # has 33 frames matching ^Cause (verified via fn.frames(r'^Cause')) —
    # one fewer than the paper's 34, likely a FrameNet-version difference;
    # using all 33 real ones rather than guessing the missing name.
    "Causation",
    "Causation_scenario",
    "Triggering",
    "Reason",
    "Explaining_the_facts",
    "Response",
    "Cause_bodily_experience",
    "Cause_change",
    "Cause_change_of_consistency",
    "Cause_change_of_phase",
    "Cause_change_of_position_on_a_scale",
    "Cause_change_of_strength",
    "Cause_emotion",
    "Cause_expansion",
    "Cause_fluidic_motion",
    "Cause_harm",
    "Cause_impact",
    "Cause_motion",
    "Cause_proliferation_in_number",
    "Cause_temperature_change",
    "Cause_to_amalgamate",
    "Cause_to_be_dry",
    "Cause_to_be_included",
    "Cause_to_be_sharp",
    "Cause_to_be_wet",
    "Cause_to_burn",
    "Cause_to_continue",
    "Cause_to_end",
    "Cause_to_experience",
    "Cause_to_fragment",
    "Cause_to_land",
    "Cause_to_make_noise",
    "Cause_to_make_progress",
    "Cause_to_move_in_place",
    "Cause_to_perceive",
    "Cause_to_resume",
    "Cause_to_rot",
    "Cause_to_start",
    "Cause_to_wake",
}

ARG0_OPEN = "<ARG0>"
ARG0_CLOSE = "</ARG0>"
ARG1_OPEN = "<ARG1>"
ARG1_CLOSE = "</ARG1>"

_ARG_MARKERS = frozenset([ARG0_OPEN, ARG0_CLOSE, ARG1_OPEN, ARG1_CLOSE])


class KCNNTokenizer:
    def __init__(self, max_seq_length: int = 200) -> None:
        self.word2index: dict[str, int] | None = None
        self.max_seq_length = max_seq_length
        self._framenet_scores: dict | None = None
        self._wordnet_cat2idx: dict[str, int] = {
            cat: i for i, cat in enumerate(NOUN_SUPERSENSES + VERB_SUPERSENSES)
        }

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        if self.word2index is None:
            return 0
        return len(self.word2index)

    @property
    def is_ready(self) -> bool:
        return self.word2index is not None

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        # Split on whitespace, keeping ARG markers intact as single tokens.
        for part in text.split():
            if part in _ARG_MARKERS:
                tokens.append(part)
            else:
                tokens.append(part.lower())
        return tokens

    def build_vocab(self, texts: list[str]) -> None:
        counts: Counter = Counter()
        for text in texts:
            for tok in self.tokenize(text):
                if tok not in _ARG_MARKERS:
                    counts[tok] += 1
        self.word2index = {"[PAD]": 0, "[UNK]": 1}
        for word, _ in counts.most_common():
            if word not in self.word2index:
                self.word2index[word] = len(self.word2index)

    def save_vocab(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.word2index, f)

    def load_vocab(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.word2index = pickle.load(f)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> dict:
        tokens = self.tokenize(text)

        e1_words = self._find_entity_words(tokens, ARG0_OPEN, ARG0_CLOSE)
        e2_words = self._find_entity_words(tokens, ARG1_OPEN, ARG1_CLOSE)

        # Find head positions (first word of each entity span) in the token list
        e1_head_idx = self._find_head_idx(tokens, ARG0_OPEN)
        e2_head_idx = self._find_head_idx(tokens, ARG1_OPEN)

        # Strip ARG markers to get surface tokens
        surface = [t for t in tokens if t not in _ARG_MARKERS]
        surface = surface[: self.max_seq_length]

        # Adjust head indices for removed markers
        e1_head_surface = self._surface_idx(tokens, e1_head_idx)
        e2_head_surface = self._surface_idx(tokens, e2_head_idx)

        word_ids = [
            self.word2index.get(t, 1) if self.word2index else 1
            for t in surface
        ]
        position_ids = self._compute_positions(surface, e1_head_surface, e2_head_surface)
        attention_mask = [1] * len(surface)

        # K-channel: words between entities, lemmatized
        between_words = self._extract_between_words(tokens)
        between_lemmatized = self._lemmatize_tokens(between_words)
        k_channel_ids = [
            self.word2index.get(w, 1) if self.word2index else 1
            for w in between_lemmatized
        ]

        wordnet_features = self._wordnet_features(e1_words, e2_words)
        framenet_scores = self._framenet_scores_for(tokens)

        return {
            "word_ids": word_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "k_channel_ids": k_channel_ids,
            "wordnet_features": wordnet_features,
            "framenet_scores": framenet_scores,
        }

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def _find_entity_words(
        self, tokens: list[str], open_tag: str, close_tag: str
    ) -> list[str]:
        try:
            start = tokens.index(open_tag)
            end = tokens.index(close_tag)
        except ValueError:
            return []
        return [t for t in tokens[start + 1 : end] if t not in _ARG_MARKERS]

    def _find_head_idx(self, tokens: list[str], open_tag: str) -> int:
        try:
            marker_pos = tokens.index(open_tag)
        except ValueError:
            return 0
        # Head is the first non-marker token after the open tag
        for i in range(marker_pos + 1, len(tokens)):
            if tokens[i] not in _ARG_MARKERS:
                return i
        return marker_pos

    def _surface_idx(self, tokens: list[str], raw_idx: int) -> int:
        # Count how many markers appear before raw_idx
        markers_before = sum(1 for t in tokens[:raw_idx] if t in _ARG_MARKERS)
        return raw_idx - markers_before

    def _extract_between_words(self, tokens: list[str]) -> list[str]:
        """Return surface words strictly between the two entity spans."""
        try:
            arg0_open = tokens.index(ARG0_OPEN)
        except ValueError:
            arg0_open = len(tokens)
        try:
            arg1_open = tokens.index(ARG1_OPEN)
        except ValueError:
            arg1_open = len(tokens)

        if arg0_open < arg1_open:
            try:
                start = tokens.index(ARG0_CLOSE) + 1
                end = tokens.index(ARG1_OPEN)
            except ValueError:
                return []
        else:
            try:
                start = tokens.index(ARG1_CLOSE) + 1
                end = tokens.index(ARG0_OPEN)
            except ValueError:
                return []

        return [t for t in tokens[start:end] if t not in _ARG_MARKERS]

    def _lemmatize(self, word: str) -> str:
        try:
            from nltk.stem import WordNetLemmatizer
            if not hasattr(self, "_lemmatizer"):
                self._lemmatizer = WordNetLemmatizer()
            return self._lemmatizer.lemmatize(word)
        except (ImportError, LookupError):
            return word

    def _lemmatize_tokens(self, words: list[str]) -> list[str]:
        """POS-aware WordNet lemmatization of a token sequence.

        ``WordNetLemmatizer.lemmatize`` defaults to noun POS, which leaves
        verbs inflected ("caused" → "caused").  The paper lemmatizes the
        K-channel input "to base form", which requires POS information; we
        POS-tag the sequence and map Treebank tags to WordNet POS.  Falls
        back to noun-only lemmatization if the tagger is unavailable.
        """
        if not words:
            return []
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer

            if not hasattr(self, "_lemmatizer"):
                self._lemmatizer = WordNetLemmatizer()
            try:
                tagged = nltk.pos_tag(words)
            except LookupError:
                nltk.download("averaged_perceptron_tagger_eng", quiet=True)
                nltk.download("averaged_perceptron_tagger", quiet=True)
                tagged = nltk.pos_tag(words)
            pos_map = {"J": "a", "V": "v", "N": "n", "R": "r"}
            return [
                self._lemmatizer.lemmatize(w, pos_map.get(tag[:1], "n"))
                for w, tag in tagged
            ]
        except (ImportError, LookupError):
            return [self._lemmatize(w) for w in words]

    def _compute_positions(
        self,
        tokens: list[str],
        e1_head_idx: int,
        e2_head_idx: int,
    ) -> list[list[int]]:
        max_dist = self.max_seq_length - 1
        positions: list[list[int]] = []
        for i in range(len(tokens)):
            d1 = 0 if i == e1_head_idx else i - e1_head_idx
            d2 = 0 if i == e2_head_idx else i - e2_head_idx
            # Clamp to valid embedding range [-max_dist, +max_dist]
            d1 = max(-max_dist, min(max_dist, d1))
            d2 = max(-max_dist, min(max_dist, d2))
            positions.append([d1, d2])
        return positions

    # ------------------------------------------------------------------
    # WordNet supersense features
    # ------------------------------------------------------------------

    def _wordnet_features(
        self, e1_words: list[str], e2_words: list[str]
    ) -> list[int]:
        n_cats = len(NOUN_SUPERSENSES) + len(VERB_SUPERSENSES)
        vec = [0] * (2 * n_cats)
        try:
            from nltk.corpus import wordnet as wn

            def _fill(words: list[str], offset: int) -> None:
                for word in words:
                    for synset in wn.synsets(word):
                        lex = synset.lexname()
                        if lex in self._wordnet_cat2idx:
                            vec[offset + self._wordnet_cat2idx[lex]] = 1

            _fill(e1_words, 0)
            _fill(e2_words, n_cats)
        except (ImportError, LookupError):
            pass
        return vec

    # ------------------------------------------------------------------
    # FrameNet causal score features
    # ------------------------------------------------------------------

    def _framenet_scores_for(self, tokens: list[str]) -> list[float]:
        try:
            scores = self._load_framenet_scores()
        except (ImportError, LookupError, Exception):
            return [0.0, 0.0, 0.0, 0.0]

        surface = [t for t in tokens if t not in _ARG_MARKERS]

        # Determine boundary indices in the surface token list
        # We track where ARG0 and ARG1 spans begin and end in the surface list.
        arg0_start, arg0_end = self._entity_surface_span(tokens, ARG0_OPEN, ARG0_CLOSE)
        arg1_start, arg1_end = self._entity_surface_span(tokens, ARG1_OPEN, ARG1_CLOSE)

        # Define region boundaries (sorted so regions are always well-defined)
        first_start = min(arg0_start, arg1_start)
        first_end = min(arg0_end, arg1_end)
        second_start = max(arg0_start, arg1_start)

        def _region_score(region: list[str]) -> float:
            total = 0.0
            for tok in region:
                key = tok + ".v"
                if key in scores:
                    total += scores[key]
                key2 = tok + ".n"
                if key2 in scores:
                    total += scores[key2]
            return total

        total_score = _region_score(surface)
        before_score = _region_score(surface[:first_start])
        between_score = _region_score(surface[first_end:second_start])
        after_score = _region_score(surface[max(arg0_end, arg1_end):])

        return [total_score, before_score, between_score, after_score]

    def _entity_surface_span(
        self, tokens: list[str], open_tag: str, close_tag: str
    ) -> tuple[int, int]:
        try:
            raw_start = tokens.index(open_tag)
            raw_end = tokens.index(close_tag)
        except ValueError:
            return (0, 0)
        start = self._surface_idx(tokens, raw_start + 1)
        end = self._surface_idx(tokens, raw_end)
        return (start, end)

    def _load_framenet_scores(self) -> dict[str, float]:
        if self._framenet_scores is not None:
            return self._framenet_scores

        from nltk.corpus import framenet as fn

        lu_counts: dict[str, int] = {}
        causal_counts: dict[str, int] = {}

        for frame in fn.frames():
            is_causal = frame.name in CAUSAL_FRAMES
            for lu in frame.lexUnit.values():
                # lu.name is like "cause.v" or "reason.n"
                key = lu.name
                lu_counts[key] = lu_counts.get(key, 0) + 1
                if is_causal:
                    causal_counts[key] = causal_counts.get(key, 0) + 1

        self._framenet_scores = {
            key: causal_counts.get(key, 0) / total
            for key, total in lu_counts.items()
            if total > 0
        }
        return self._framenet_scores
