"""Reproduction training and evaluation of k-CNN.

Usage
-----
    python repro.py [--dataset DATASET] [--config CONFIG]
                    [--word-embeddings PATH] [--aux-cache DIR]
                    [--folds N] [--epochs N] [--batch-size N] [--lr LR]
                    [--no-wordnet] [--no-framenet] [--no-k-channel]
                    [--no-auto-download] [--seed N] [--output-dir DIR]

The script trains KCNNForSequenceClassification on pair-level causality
identification.  The causality-identification split is flattened to one
example per entity pair: causal pairs → label=1, non-causal sentences
(first two entities used as the pair) → label=0.

Word embeddings
---------------
The paper uses Levy & Goldberg (2014) dependency-based word embeddings
(deps.words, 300-dim, pre-normalised unit vectors).  The original download
URL (https://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2) is no longer
accessible.  This script automatically downloads the embeddings from the
Wayback Machine archive (see _DEPS_WORDS_WAYBACK_URL below) on first run and
caches them in --aux-cache.  Pass --no-auto-download to skip this and fall
back to random initialisation, or pass --word-embeddings to supply a custom
embedding file in word2vec text format.

See notes.txt for underspecifications identified during reproduction.

Dependencies
------------
    pip install torch transformers datasets scikit-learn numpy
    pip install nltk        # for WordNet features and FrameNet filter building
    nltk.download('wordnet')
    nltk.download('framenet_v17')
"""

from __future__ import annotations

import argparse
import bz2
import math
import pickle
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.model_selection import KFold
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from configuration_kcnn import KCNNConfig
from modeling_kcnn import KCNNForSequenceClassification
from tokenization_kcnn import (
    CAUSAL_FRAMES,
    NOUN_SUPERSENSES,
    VERB_SUPERSENSES,
    KCNNTokenizer,
)

# ---------------------------------------------------------------------------
# Levy-Goldberg dependency-based word embeddings
# ---------------------------------------------------------------------------

# The original distribution URL (u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2)
# is no longer accessible as of 2026-06.  The Wayback Machine archived a copy
# on 2021-04-20; this is the only known surviving source of these embeddings.
# The `if_` suffix tells Wayback Machine to serve the raw archived content
# without an HTML interstitial page (which would otherwise break binary downloads).
# Note: the 2021-04-20 snapshot is truncated at ~102MB; use the 2022-10-12 snapshot
# which is a complete 320MB archive.
_DEPS_WORDS_WAYBACK_URL = (
    "https://web.archive.org/web/20221012052827if_/"
    "https://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2"
)


def _ensure_levy_goldberg(aux_cache_dir: Path) -> Optional[Path]:
    """Return path to the decompressed deps.words file, downloading if needed.

    The Levy-Goldberg embeddings are distributed as a bzip2-compressed word2vec
    text file (~320 MB compressed, ~1.1 GB decompressed).  The original BIU
    server is offline; this function fetches from the Wayback Machine archive.
    """
    cache_path = aux_cache_dir / "deps.words"
    if cache_path.exists():
        return cache_path

    aux_cache_dir.mkdir(parents=True, exist_ok=True)
    bz2_path = aux_cache_dir / "deps.words.bz2"

    if not bz2_path.exists():
        print("Downloading Levy-Goldberg dependency embeddings (~320 MB) …")
        print("  NOTE: Original URL (u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2)")
        print("        is no longer accessible. Fetching from Wayback Machine archive.")
        print(f"  URL: {_DEPS_WORDS_WAYBACK_URL}")

        def _progress(block_num, block_size, total_size):
            pct = block_num * block_size / total_size * 100
            if block_num % 500 == 0:
                print(f"  {min(pct, 100):.0f}%", end="\r", flush=True)

        urllib.request.urlretrieve(_DEPS_WORDS_WAYBACK_URL, str(bz2_path), _progress)
        print()

    print(f"Decompressing {bz2_path.name} (~1.1 GB uncompressed) …")
    with bz2.open(str(bz2_path), "rb") as f_in, open(str(cache_path), "wb") as f_out:
        while chunk := f_in.read(1 << 20):
            f_out.write(chunk)
    bz2_path.unlink()
    print(f"  Saved to {cache_path}")
    return cache_path


# ---------------------------------------------------------------------------
# Knowledge filter construction
# ---------------------------------------------------------------------------

def _load_word2vec_text(path: str, vocab: set[str]) -> dict[str, np.ndarray]:
    """Return {word: vector} for words in vocab from a word2vec text-format file.

    Handles both bare text files and files with a leading ``N D`` header line
    (standard word2vec output).  Compatible with Levy-Goldberg deps.words and
    wiki-extvec (Komninos & Manandhar 2016).
    """
    embeddings: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) < 2:
                continue
            word = parts[0]
            if word in vocab:
                try:
                    embeddings[word] = np.asarray(parts[1:], dtype="float32")
                except ValueError:
                    pass  # skip malformed lines (e.g. header "N D")
    return embeddings


def _load_wiki_extvec(path: str, vocab: set[str]) -> dict[str, np.ndarray]:
    """Alias kept for backwards compatibility; delegates to _load_word2vec_text."""
    return _load_word2vec_text(path, vocab)


def _extract_causal_lus() -> tuple[list[list[str]], list[list[str]], list[list[str]]]:
    """Extract causal lexical units from NLTK FrameNet, grouped by n-gram length."""
    try:
        from nltk.corpus import framenet as fn
        from nltk.corpus import wordnet as wn
    except LookupError:
        import nltk
        nltk.download("framenet_v17", quiet=True)
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import framenet as fn
        from nltk.corpus import wordnet as wn

    lu1: list[list[str]] = []
    lu2: list[list[str]] = []
    lu3: list[list[str]] = []

    for frame_name in CAUSAL_FRAMES:
        try:
            frame = fn.frame(frame_name)
        except Exception:
            continue
        for lu_name in frame.lexUnit:
            word_part = lu_name.rsplit(".", 1)[0].replace("_", " ")
            words = word_part.lower().split()
            if len(words) == 1:
                lu1.append(words)
            elif len(words) == 2:
                lu2.append(words)
            elif len(words) == 3:
                lu3.append(words)

    # Extend unigrams with WordNet synonyms/hyponyms whose definitions mention causality.
    # Exact keyword set from the paper's Algorithm 1, Step 2 (previously had
    # 3 extra keywords here — "trigger", "produce", "lead" — not in the paper).
    _CAUSAL_KEYWORDS = {"cause", "effect", "causal", "causation", "result", "reason",
                        "because", "responsible"}

    extended_lu1: set[tuple] = set(tuple(w) for w in lu1)
    extended_lu2: set[tuple] = set(tuple(w) for w in lu2)
    extended_lu3: set[tuple] = set(tuple(w) for w in lu3)

    for word_list in lu1:
        word = word_list[0]
        for synset in wn.synsets(word):
            gloss = synset.definition().lower()
            if any(kw in gloss for kw in _CAUSAL_KEYWORDS):
                for lemma in synset.lemmas():
                    parts = lemma.name().replace("_", " ").lower().split()
                    if len(parts) == 1:
                        extended_lu1.add(tuple(parts))
                    elif len(parts) == 2:
                        extended_lu2.add(tuple(parts))
                    elif len(parts) == 3:
                        extended_lu3.add(tuple(parts))

    return (
        [list(t) for t in extended_lu1],
        [list(t) for t in extended_lu2],
        [list(t) for t in extended_lu3],
    )


def build_knowledge_filters(
    embeddings_path: Optional[str],
    aux_cache_dir: Path,
    embedding_dim: int = 300,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Build knowledge-oriented filters for the K-channel.

    Returns (filters_dict, k_filters_per_size) where:
    - filters_dict: {str(window_size): tensor [n_filters, embedding_dim, ws]}
    - k_filters_per_size: [n1, n2, n3] matching k_filter_sizes=[1, 2, 3]
    """
    cache_path = aux_cache_dir / "kcnn_knowledge_filters.pkl"
    if cache_path.exists():
        print(f"Loading cached knowledge filters from {cache_path}")
        return pickle.load(open(cache_path, "rb"))

    aux_cache_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting FrameNet causal lexical units …")
    lu1, lu2, lu3 = _extract_causal_lus()
    print(f"  unigrams={len(lu1)}  bigrams={len(lu2)}  trigrams={len(lu3)}")

    # Collect all words that appear in any LU
    all_words: set[str] = set()
    for lu_list in (lu1, lu2, lu3):
        for words in lu_list:
            all_words.update(words)

    emb: dict[str, np.ndarray] = {}
    if embeddings_path is not None:
        print(f"Loading word embeddings for K-channel filters from {embeddings_path} …")
        emb = _load_word2vec_text(embeddings_path, all_words)
        print(f"  found {len(emb)}/{len(all_words)} LU words in embeddings")
    else:
        print("WARNING: No embeddings path provided. Knowledge filters will be random.")

    rng = np.random.RandomState(42)

    def _word_vec(w: str) -> np.ndarray:
        if w in emb:
            return emb[w]
        # Random unit vector for OOV
        v = rng.randn(embedding_dim).astype("float32")
        return v / (np.linalg.norm(v) + 1e-9)

    def _make_filters(lu_list: list[list[str]], ws: int) -> torch.Tensor | None:
        mats = []
        for words in lu_list:
            if len(words) != ws:
                continue
            # Stack word embeddings → [ws, embedding_dim] → transpose → [embedding_dim, ws]
            mat = np.stack([_word_vec(w) for w in words], axis=0).T
            mats.append(mat)
        if not mats:
            return None
        arr = np.stack(mats, axis=0)  # [n_filters, embedding_dim, ws]
        return torch.tensor(arr, dtype=torch.float32)

    filters_dict: dict[str, torch.Tensor] = {}
    k_filters_per_size: list[int] = []

    for ws, lu_list in zip([1, 2, 3], [lu1, lu2, lu3]):
        t = _make_filters(lu_list, ws)
        if t is not None:
            filters_dict[str(ws)] = t
            k_filters_per_size.append(t.shape[0])
            print(f"  window_size={ws}: {t.shape[0]} filters")
        else:
            k_filters_per_size.append(0)

    result = (filters_dict, k_filters_per_size)
    pickle.dump(result, open(cache_path, "wb"))
    return result


# ---------------------------------------------------------------------------
# ANOVA filter selection + K-means clustering (paper §3.2)
# ---------------------------------------------------------------------------

def compute_k_channel_activations(
    filters_dict: dict[str, torch.Tensor],
    word_embedding_matrix: torch.Tensor,
    encoded_examples: list[dict],
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """Max-pooled cosine-similarity activation of every filter on every example.

    Mirrors the model's K-channel forward pass (unit-normalised embeddings,
    conv / window size, max over positions).  Returns
    ``{window_size_str: [n_examples, n_filters] array}``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emb = word_embedding_matrix.float().to(device)

    activations: dict[str, np.ndarray] = {}
    for ws_str, filt in filters_dict.items():
        ws = int(ws_str)
        f = filt.float().to(device)
        f = F.normalize(f.reshape(f.size(0), -1), dim=1).reshape_as(f)
        rows = []
        for start in range(0, len(encoded_examples), batch_size):
            chunk = encoded_examples[start:start + batch_size]
            max_len = max(max((len(ex["k_channel_ids"]) for ex in chunk), default=1), ws)
            ids = torch.zeros(len(chunk), max_len, dtype=torch.long)
            for i, ex in enumerate(chunk):
                k = ex["k_channel_ids"][:max_len]
                ids[i, : len(k)] = torch.tensor(k, dtype=torch.long)
            x = F.normalize(emb[ids.to(device)], p=2, dim=-1).permute(0, 2, 1)
            out = F.conv1d(x, f) / ws
            rows.append(out.max(dim=2).values.cpu().numpy())
        activations[ws_str] = np.concatenate(rows, axis=0)
    return activations


def refine_knowledge_filters(
    filters_dict: dict[str, torch.Tensor],
    word_embedding_matrix: torch.Tensor,
    encoded_examples: list[dict],
    labels: list[int],
    f_critical: float = 2.9957,
    cluster: bool = True,
    seed: int = 42,
) -> tuple[dict[str, torch.Tensor], list[int], dict[str, list[int]]]:
    """ANOVA F-ratio filter selection followed by K-means filter clustering.

    Paper: keep filters whose one-way ANOVA F-ratio between the class groups
    of their max-pooled activations exceeds the critical value (2.9957,
    α=5%); cluster the surviving unigram and bigram filters separately into
    floor(n/2) K-means clusters (trigrams are kept as-is) and max-pool the
    activations within each cluster.

    Returns (selected_filters, filters_per_size aligned to window sizes
    [1, 2, 3], cluster_ids usable as ``KCNNConfig.k_cluster_ids``).
    """
    from scipy.stats import f_oneway
    from sklearn.cluster import KMeans

    y = np.asarray(labels)
    activations = compute_k_channel_activations(filters_dict, word_embedding_matrix, encoded_examples)

    selected: dict[str, torch.Tensor] = {}
    per_size: list[int] = []
    cluster_ids: dict[str, list[int]] = {}

    for ws in (1, 2, 3):
        ws_str = str(ws)
        if ws_str not in filters_dict:
            per_size.append(0)
            continue
        acts = activations[ws_str]
        groups = [acts[y == cls] for cls in np.unique(y)]
        f_stats = f_oneway(*groups).statistic
        keep = np.flatnonzero(np.nan_to_num(f_stats, nan=0.0) > f_critical)
        print(f"  ws={ws}: ANOVA keeps {len(keep)}/{acts.shape[1]} filters (F > {f_critical})")
        if len(keep) == 0:
            per_size.append(0)
            continue
        kept_filters = filters_dict[ws_str][keep]
        selected[ws_str] = kept_filters
        per_size.append(len(keep))

        if cluster and ws in (1, 2) and len(keep) > 1:
            n_clusters = max(len(keep) // 2, 1)
            flat = kept_filters.reshape(len(keep), -1).numpy()
            km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(flat)
            cluster_ids[ws_str] = km.labels_.tolist()
            print(f"    K-means: {len(keep)} filters → {n_clusters} clusters")
        else:
            cluster_ids[ws_str] = list(range(len(keep)))  # trigrams kept as-is

    return selected, per_size, cluster_ids


# ---------------------------------------------------------------------------
# Dataset flattening: causality-identification → pair-level examples
# ---------------------------------------------------------------------------

_ENTITY_TAG_RE = re.compile(r"</?e\d+>")


def _get_entity_ids_in_order(text: str) -> list[str]:
    """Return entity IDs in the order their opening tags appear."""
    seen: list[str] = []
    for m in re.finditer(r"<(e\d+)>", text):
        eid = m.group(1)
        if eid not in seen:
            seen.append(eid)
    return seen


def flatten_to_pairs(dataset, directed: bool = True) -> list[dict]:
    """Convert causality-identification examples to pair-level examples.

    ``directed=True`` (default): enumerate all ordered entity pairs —
    causal pairs → label=1, every other ordered pair (including the reverse
    direction of a causal pair) → label=0.

    ``directed=False`` (paper protocol): one example per unordered entity
    pair, entities assigned to ARG0/ARG1 in textual order, label=1 if EITHER
    direction is annotated causal.  This matches the k-CNN paper's binary
    setup, where "the direction of causality is not used" and SemEval's
    Cause-Effect(e1,e2) and Cause-Effect(e2,e1) both count as causal.

    Sentences with no entity markers are skipped (k-CNN requires marked
    pairs).  See notes.txt gap #11 for why non-causal sentences are dropped.
    """
    examples = []
    for ex in dataset:
        text: str = ex["text"]
        relations: list = list(ex.get("relations") or [])
        entity_order = _get_entity_ids_in_order(text)

        if len(entity_order) < 2:
            # No entity markers: skip (see notes.txt gap #11)
            continue

        # Build set of causal ordered pairs
        causal_set: set[tuple[str, str]] = {
            (r["first"], r["second"])
            for r in relations
            if r.get("relationship", 0) == 1
        }

        # Enumerate entity pairs from the marked entities
        for i, a in enumerate(entity_order):
            for b in entity_order[i + 1:]:
                if directed:
                    directions = [(a, b), (b, a)]
                else:
                    directions = [(a, b)]  # textual order
                for src, tgt in directions:
                    if directed:
                        label = 1 if (src, tgt) in causal_set else 0
                    else:
                        label = 1 if ((src, tgt) in causal_set or (tgt, src) in causal_set) else 0
                    paired_text = text
                    paired_text = paired_text.replace(f"<{src}>", "<ARG0>").replace(f"</{src}>", "</ARG0>")
                    paired_text = paired_text.replace(f"<{tgt}>", "<ARG1>").replace(f"</{tgt}>", "</ARG1>")
                    paired_text = _ENTITY_TAG_RE.sub("", paired_text)
                    # Normalise spacing: ensure markers are surrounded by spaces
                    # so whitespace-split tokenization handles <ARG0>word</ARG0>
                    # correctly (original texts may have no space between tag and word).
                    for marker in ("<ARG0>", "</ARG0>", "<ARG1>", "</ARG1>"):
                        paired_text = paired_text.replace(marker, f" {marker} ")
                    paired_text = " ".join(paired_text.split())
                    examples.append({"text": paired_text, "label": label})

    return examples


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

@dataclass
class KCNNDataCollator:
    kcnn_tokenizer: KCNNTokenizer
    max_seq_length: int = 200

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = min(self.max_seq_length, max(len(f["word_ids"]) for f in features))
        max_k_len = max((len(f["k_channel_ids"]) for f in features), default=0)
        # Ensure at least 1 token for the K-channel (Conv1d requires non-empty input)
        max_k_len = max(max_k_len, 1)

        word_ids_batch = []
        k_ids_batch = []
        pos_ids_batch = []
        masks_batch = []
        wordnet_batch = []
        framenet_batch = []
        labels_batch = []

        for f in features:
            n = len(f["word_ids"])
            pad = max_len - n
            if pad >= 0:
                word_ids_batch.append(f["word_ids"][:max_len] + [0] * pad)
                pos = f["position_ids"][:max_len]
                pos = pos + [[0, 0]] * (max_len - len(pos))
                pos_ids_batch.append(pos)
                masks_batch.append(f["attention_mask"][:max_len] + [0] * pad)
            else:
                word_ids_batch.append(f["word_ids"][:max_len])
                pos_ids_batch.append(f["position_ids"][:max_len])
                masks_batch.append(f["attention_mask"][:max_len])

            k = f["k_channel_ids"][:max_k_len]
            k_ids_batch.append(k + [0] * (max_k_len - len(k)))

            wordnet_batch.append(f["wordnet_features"])
            framenet_batch.append(f["framenet_scores"])
            labels_batch.append(f["label"])

        return {
            "input_ids": torch.tensor(word_ids_batch, dtype=torch.long),
            "k_channel_ids": torch.tensor(k_ids_batch, dtype=torch.long),
            "d_channel_position_ids": torch.tensor(pos_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(masks_batch, dtype=torch.long),
            "wordnet_features": torch.tensor(wordnet_batch, dtype=torch.float),
            "framenet_scores": torch.tensor(framenet_batch, dtype=torch.float),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Macro-averaged F1 (matches paper evaluation protocol)
    f1s, precisions, recalls = [], [], []
    for cls in [0, 1]:
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    accuracy = float((preds == labels).mean()) if len(labels) > 0 else 0.0
    return {
        "precision": precisions[1],   # causal class, for reference
        "recall": recalls[1],
        "f1": float(np.mean(f1s)),    # macro-averaged
        "accuracy": accuracy,
    }


# ---------------------------------------------------------------------------
# Dataset encoding
# ---------------------------------------------------------------------------

def encode_dataset(examples: list[dict], tokenizer: KCNNTokenizer) -> list[dict]:
    encoded = []
    for ex in examples:
        enc = tokenizer.encode(ex["text"])
        enc["label"] = ex["label"]
        encoded.append(enc)
    return encoded


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce k-CNN training and evaluation.")
    p.add_argument("--dataset", default="thagen/SemEval2010T8")
    p.add_argument("--config", default="causality identification",
                   help="Dataset configuration name")
    p.add_argument("--local-data-dir", default=None,
                   help="Load parquets from a local dataset directory instead of HF Hub. "
                        "Expects {dir}/{config}/train.parquet and {dir}/{config}/test.parquet, "
                        "where {config} is --config with spaces replaced by hyphens. "
                        "Example: --local-data-dir /path/to/causalatee/resources/datasets/SemEval2010T8")
    p.add_argument("--word-embeddings", default=None,
                   help="Path to a word2vec text-format embedding file (Levy-Goldberg deps.words "
                        "or wiki-extvec).  If omitted, the script auto-downloads the Levy-Goldberg "
                        "dependency embeddings from the Wayback Machine (see _DEPS_WORDS_WAYBACK_URL). "
                        "Use --no-auto-download to disable this and fall back to random initialisation.")
    p.add_argument("--no-auto-download", action="store_true",
                   help="Do not attempt to auto-download Levy-Goldberg embeddings. "
                        "Falls back to random word embeddings (lower performance).")
    p.add_argument("--aux-cache", default="./kcnn-aux-cache",
                   help="Directory for caching pre-built knowledge filters and word embeddings.")
    p.add_argument("--no-k-channel", action="store_true",
                   help="Disable the knowledge-oriented channel (ablation).")
    p.add_argument("--no-wordnet", action="store_true",
                   help="Disable WordNet features.")
    p.add_argument("--no-framenet", action="store_true",
                   help="Disable FrameNet causal scores.")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=1.0,
                   help="Adadelta learning rate (scaling factor; default 1.0)")
    p.add_argument("--early-stopping", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="./kcnn-output")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    aux_cache_dir = Path(args.aux_cache)

    # ------------------------------------------------------------------
    # 1. Load raw dataset and flatten to pair-level examples
    # ------------------------------------------------------------------
    if args.local_data_dir:
        config_dir = args.config.replace(" ", "-")
        data_dir = Path(args.local_data_dir)
        train_path = str(data_dir / config_dir / "train.parquet")
        test_path  = str(data_dir / config_dir / "test.parquet")
        print(f"\nLoading local parquets from {data_dir / config_dir} …")
        raw = load_dataset("parquet", data_files={"train": train_path, "test": test_path})
    else:
        print(f"\nLoading '{args.dataset}' / '{args.config}' …")
        raw = load_dataset(args.dataset, args.config)
    print(raw)

    print("Flattening to pair-level examples …")
    train_pairs = flatten_to_pairs(raw["train"])
    test_pairs  = flatten_to_pairs(raw["test"])
    train_pos = sum(x['label'] for x in train_pairs)
    train_neg = sum(1 - x['label'] for x in train_pairs)
    test_pos  = sum(x['label'] for x in test_pairs)
    test_neg  = sum(1 - x['label'] for x in test_pairs)
    print(f"  train pairs: {len(train_pairs)}  (pos={train_pos} neg={train_neg})")
    print(f"  test  pairs: {len(test_pairs)}  (pos={test_pos} neg={test_neg})")
    if train_neg == 0:
        print(
            "ERROR: No negative training examples found.\n"
            "  The causality-identification format in causalatee only preserves entity\n"
            "  markers for causal (positive) examples. Non-causal sentences are stored\n"
            "  as plain text without markers and are skipped (see notes.txt gap #11).\n"
            "  k-CNN requires entity markers for all examples. Training aborted."
        )
        return

    # ------------------------------------------------------------------
    # 2. Build vocabulary and word-embedding matrix
    # ------------------------------------------------------------------
    tokenizer = KCNNTokenizer()

    all_texts = [ex["text"] for ex in train_pairs + test_pairs]
    print("\nBuilding vocabulary …")
    tokenizer.build_vocab(all_texts)
    print(f"  vocab size: {tokenizer.vocab_size}")

    # Resolve word embedding path: explicit > auto-download > random
    emb_dim = 300
    word_emb_path: Optional[str] = args.word_embeddings
    if word_emb_path is None and not args.no_auto_download:
        levy_path = _ensure_levy_goldberg(aux_cache_dir)
        if levy_path is not None:
            word_emb_path = str(levy_path)

    # Build word embedding matrix (or random fallback)
    word_emb_cache = aux_cache_dir / "word_embeddings.pkl"
    word_embedding_matrix = None
    if word_emb_cache.exists():
        print(f"Loading cached word embeddings from {word_emb_cache}")
        word_embedding_matrix = pickle.load(open(word_emb_cache, "rb"))
        if word_embedding_matrix.shape[0] != tokenizer.vocab_size:
            print(f"  Vocab size mismatch (cache={word_embedding_matrix.shape[0]}, "
                  f"current={tokenizer.vocab_size}). Rebuilding.")
            word_emb_cache.unlink()
            word_embedding_matrix = None
    if word_embedding_matrix is None:
        aux_cache_dir.mkdir(parents=True, exist_ok=True)
        vocab = set(tokenizer.word2index.keys())
        rng = np.random.RandomState(args.seed)
        # Paper: OOV words get random UNIT vectors (embeddings are unit-normalised).
        random_init = rng.randn(tokenizer.vocab_size, emb_dim).astype("float32")
        random_init /= np.linalg.norm(random_init, axis=1, keepdims=True) + 1e-9
        random_init[0] = 0.0  # PAD
        word_embedding_matrix = torch.tensor(random_init)
        if word_emb_path is not None:
            print(f"Loading word embeddings for vocabulary from {word_emb_path} …")
            loaded = _load_word2vec_text(word_emb_path, vocab)
            found = 0
            for word, idx in tokenizer.word2index.items():
                if word in loaded:
                    word_embedding_matrix[idx] = torch.tensor(loaded[word])
                    found += 1
            print(f"  {found}/{tokenizer.vocab_size} vocab words found in embeddings")
        else:
            print("WARNING: No word embeddings available — using random initialisation.")
        pickle.dump(word_embedding_matrix, open(word_emb_cache, "wb"))

    # ------------------------------------------------------------------
    # 3. Build knowledge filters (K-channel)
    # ------------------------------------------------------------------
    knowledge_filters = None
    k_filters_per_size = [0, 0, 0]

    if not args.no_k_channel:
        try:
            knowledge_filters, k_filters_per_size = build_knowledge_filters(
                embeddings_path=word_emb_path,
                aux_cache_dir=aux_cache_dir,
                embedding_dim=emb_dim,
            )
        except Exception as e:
            print(f"WARNING: Could not build knowledge filters ({e}). K-channel disabled.")
            k_filters_per_size = [0, 0, 0]

    # ------------------------------------------------------------------
    # 4. Encode datasets
    # ------------------------------------------------------------------
    print("\nEncoding dataset …")
    train_encoded = encode_dataset(train_pairs, tokenizer)
    test_encoded  = encode_dataset(test_pairs,  tokenizer)

    # ------------------------------------------------------------------
    # 5. Model config
    # ------------------------------------------------------------------
    config = KCNNConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=emb_dim,
        pos_embedding_dim=20,
        max_seq_length=200,
        num_labels=2,
        dropout_rate=0.4,
        k_filter_sizes=[1, 2, 3],
        k_filters_per_size=k_filters_per_size,
        d_filter_sizes=[3, 4],
        num_d_filters=25,
        use_wordnet_features=not args.no_wordnet,
        use_framenet_scores=not args.no_framenet,
    )
    print(f"Config: K-channel={config.k_channel_output_dim} D-channel={config.d_channel_output_dim} "
          f"WordNet={config.wordnet_dim} FrameNet={config.framenet_dim} "
          f"classifier_in={config.classifier_input_dim}")

    collator = KCNNDataCollator(kcnn_tokenizer=tokenizer, max_seq_length=config.max_seq_length)

    # ------------------------------------------------------------------
    # 6. 10-fold cross-validation
    # ------------------------------------------------------------------
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    train_indices = np.arange(len(train_encoded))
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_indices)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{args.folds}  |  train={len(tr_idx)}  val={len(val_idx)}")
        print(f"{'='*60}")

        from datasets import Dataset as HFDataset
        train_fold = [train_encoded[i] for i in tr_idx]
        val_fold   = [train_encoded[i] for i in val_idx]
        train_hf = HFDataset.from_list(train_fold)
        val_hf   = HFDataset.from_list(val_fold)

        model = KCNNForSequenceClassification(
            config,
            knowledge_filters=knowledge_filters,
            word_embeddings=word_embedding_matrix,
        )
        model.to(device)

        fold_output = Path(args.output_dir) / f"fold_{fold + 1}"
        training_args = TrainingArguments(
            output_dir=str(fold_output),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            remove_unused_columns=False,
            report_to="none",
            max_grad_norm=5.0,
            seed=args.seed,
        )

        # Adadelta as specified in the paper (Zeiler 2012)
        optimizer = torch.optim.Adadelta(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            rho=0.95,
            eps=1e-6,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping)],
            data_collator=collator,
            optimizers=(optimizer, None),
        )

        trainer.train()

        val_results = trainer.evaluate()
        print(f"\nFold {fold + 1} validation results:")
        for k, v in sorted(val_results.items()):
            if "runtime" not in k:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        fold_results.append(val_results)

        # Evaluate on shared test set
        trainer.compute_metrics = compute_metrics
        test_hf = HFDataset.from_list(test_encoded)
        test_results = trainer.evaluate(eval_dataset=test_hf)
        print(f"\nFold {fold + 1} test results:")
        for k, v in sorted(test_results.items()):
            if "runtime" not in k:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 7. Aggregate
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("10-fold cross-validation summary (validation set)")
    print("="*60)
    metric_keys = [k for k in fold_results[0] if "runtime" not in k and "steps" not in k and "samples" not in k]
    for k in sorted(metric_keys):
        vals = [r[k] for r in fold_results if k in r]
        if vals and isinstance(vals[0], float):
            print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
