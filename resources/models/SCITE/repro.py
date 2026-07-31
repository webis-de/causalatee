"""Reproduction training and evaluation of SCITE.

Usage
-----
    python repro.py [--dataset DATASET_HF_ID] [--word-embeddings PATH]
                   [--folds N] [--epochs N] [--seed N] [--output-dir DIR]
                   [--embedding-source {flair,bert}]

The script trains ``SCITEForTokenClassification`` on the SCITE
*causality-identification* dataset using 10-fold cross-validation over the
training split and evaluates each fold on the held-out validation set and on
the shared test split.

Two groups of metrics are reported:

* **BIO span metrics** — seqeval-based precision / recall / F1 for the *C*
  (cause) and *E* (effect) tag classes independently.
* **Causal triplet metrics** — the pair-level F1 used in the original SCITE
  paper: a predicted relation (cause-span, effect-span) is a true positive only
  when *both* spans exactly match a gold relation.

Dependencies
------------
    pip install -r requirements.txt
    pip install datasets seqeval scikit-learn

Optional:
    flair          (for ``--embedding-source flair``)

Wiki-extvec word embeddings
---------------------------
SCITE's original setup uses Wikipedia + Gigaword Extended Vectors
(Komninos & Manandhar 2016) as pre-trained word embeddings.  Pass the path to
the saved numpy matrix via ``--word-embeddings``.  Without it the script uses
randomly initialised word embeddings (lower performance but still trains).
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import KFold
from torch.optim import NAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer, EarlyStoppingCallback, Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from configuration_scite import SCITEConfig
from modeling_scite import SCITEForBiaffineSpanClassification, SCITEForTokenClassification
from tokenization_scite import SciteTokenizer

# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


@dataclass
class SciteDataCollator:
    """Pad a batch of ``SciteTokenizer.encode()`` outputs to the same length.

    Handles word/char/label tensors plus optional BERT sub-word tensors.
    ``tokens`` and ``causal_relation_token_spans`` are kept as Python lists.
    ``input_ids`` carries the sequential example index so compute_metrics can
    look up the original encoded example.
    """

    scite_tokenizer: SciteTokenizer
    max_wlen: int = 58
    bert_tokenizer: Any = None  # set when embedding_source="bert"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_len = min(self.max_wlen, max(len(f["word_ids"]) for f in features))

        word_ids, char_ids, labels, masks = [], [], [], []
        for f in features:
            n = len(f["word_ids"])
            pad = batch_len - n
            word_ids.append(f["word_ids"][:batch_len] + [SciteTokenizer.PAD_WORD_ID] * pad)
            char_ids.append(
                f["char_ids"][:batch_len]
                + [[SciteTokenizer.PAD_CHAR_ID] * self.scite_tokenizer.max_clen] * pad
            )
            labels.append(f["labels"][:batch_len] + [-100] * pad)
            masks.append(f["attention_mask"][:batch_len] + [0] * pad)

        batch: Dict[str, Any] = {
            "input_ids": torch.tensor([f["index"] for f in features], dtype=torch.long),
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "char_ids": torch.tensor(char_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "tokens": [f["tokens"] for f in features],
            "causal_relation_token_spans": [f["causal_relation_token_spans"] for f in features],
        }

        # BERT sub-word tensors (only when embedding_source="bert")
        if self.bert_tokenizer is not None and "bert_input_ids" in features[0]:
            bert_ids = [f["bert_input_ids"] for f in features]
            bert_masks = [f["bert_attention_mask"] for f in features]
            bert_w2t = [f["bert_token_to_word"] for f in features]
            bert_len = max(len(ids) for ids in bert_ids)
            pad_id = self.bert_tokenizer.pad_token_id

            def _pad(seqs, pad_val, dtype=torch.long):
                return torch.tensor(
                    [s + [pad_val] * (bert_len - len(s)) for s in seqs], dtype=dtype
                )

            batch["bert_input_ids"] = _pad(bert_ids, pad_id)
            batch["bert_attention_mask"] = _pad(bert_masks, 0)
            batch["bert_token_to_word"] = _pad(bert_w2t, -1)

        return batch


# ---------------------------------------------------------------------------
# Triplet extraction (ported from Zhaoning Li's tag2triplet.py)
# ---------------------------------------------------------------------------

_CONJUNCTION = [
    ",", ";", "and", "plus", "also", "to", "then", "of",
    ", and", "and ,", "plus ,", ", plus", ", also", "also ,", ", of", "of ,",
    "; and", "and ;", "plus ;", "; plus", "; also", "also ;", "; of", "of ;",
]
_COORD_CJC = [",", "and", "or", "also", ";", "as well as", "comparable with", "either", "plus"]


def _find_spans(ls: List[int]) -> List[List[int]]:
    result = []
    for b_tag, i_tag in [(1, 2), (3, 4), (5, 6)]:
        normed = "".join(str(b_tag) if v in (b_tag, i_tag) else "0" for v in ls)
        for m in re.finditer(f"{b_tag}+", normed):
            result.append(list(range(m.start(), m.end())))
    return sorted(result)


def _check_degree(n, edges, out_deg, in_deg):
    out_d, in_d = [0] * n, [0] * n
    for e in edges:
        out_d[e[0]] = 1
        in_d[e[-1]] = 1
    return out_d == out_deg and in_d == in_deg


def _rule_one(sw, out_deg, in_deg, idx):
    edges = []
    c_idx = [i for i, v in enumerate(out_deg) if v == 1]
    e_idx = [i for i, v in enumerate(in_deg) if v == 1]
    c_spans = [(max(idx[c_idx[i]]) + 1, min(idx[c_idx[i + 1]])) for i in range(len(c_idx) - 1)]
    e_spans = [(max(idx[e_idx[i]]) + 1, min(idx[e_idx[i + 1]])) for i in range(len(e_idx) - 1)]
    c_flag = sum(any(cjc in " ".join(sw[s:e]) for cjc in _COORD_CJC) for s, e in c_spans)
    e_flag = sum(any(cjc in " ".join(sw[s:e]) for cjc in _COORD_CJC) for s, e in e_spans)
    if c_flag == len(c_spans) or e_flag == len(e_spans) or sum(out_deg) == 1 or sum(in_deg) == 1:
        for x in range(len(idx)):
            if out_deg[x]:
                for z in range(len(idx)):
                    if in_deg[z] and x != z:
                        edges.append((x, z))
    return edges


def _rule_n(sw, ls, out_deg, in_deg, idx):
    from itertools import combinations as _comb

    candidate = []
    for x in range(len(idx)):
        if not out_deg[x]:
            continue
        for z in range(len(idx)):
            if not in_deg[z] or x == z:
                continue
            lo, hi = (z, x) if x > z else (x, z)
            between = " ".join(sw[max(idx[lo]) + 1: min(idx[hi])])
            if not any(c == between for c in _CONJUNCTION):
                candidate.append((x, z))

    record = []
    for t in range(max(sum(out_deg), sum(in_deg)), len(candidate) + 1):
        found = False
        for combo in _comb(candidate, t):
            if _check_degree(len(idx), combo, out_deg, in_deg):
                record.append((sum(abs(e[0] - e[1]) for e in combo), list(combo)))
                found = True
        if found:
            break
    return min(record)[-1] if record else 0


def tags_to_triplets(ls: List[int], sw: List[str]):
    """Convert a BIO label sequence to (cause_indices, effect_indices) pairs, or 0."""
    ls = ls[: len(sw)]
    idx = _find_spans(ls)
    if not idx or set(ls) == {0}:
        return 0

    out_deg = [0] * len(idx)
    in_deg = [0] * len(idx)
    for i, span in enumerate(idx):
        first = ls[span[0]]
        if first == 1:
            out_deg[i] = 1
        elif first == 3:
            in_deg[i] = 1
        elif first == 5:
            out_deg[i] = 1
            in_deg[i] = 1

    if not sum(out_deg) or not sum(in_deg):
        return 0
    if 5 in ls and (sum(out_deg) < 2 or sum(in_deg) < 2):
        return 0

    edges = []
    if 5 not in ls:
        c_runs = list(re.finditer("1+", "".join(str(v) for v in out_deg)))
        e_runs = list(re.finditer("1+", "".join(str(v) for v in in_deg)))
        if len(c_runs) == 1 and len(e_runs) == 1:
            edges = _rule_one(sw, out_deg, in_deg, idx)
    if 5 in ls or not edges:
        edges = _rule_n(sw, ls, out_deg, in_deg, idx)

    if not edges:
        return 0
    return [[idx[e] for e in edge] for edge in edges]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

ID2LABEL = {0: "O", 1: "B-C", 2: "I-C", 3: "B-E", 4: "I-E", 5: "B-Emb", 6: "I-Emb"}


def _f1(tp, ap, pp):
    pre = tp / pp if pp else 0.0
    rec = tp / ap if ap else 0.0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
    return pre, rec, f1


def compute_triplet_f1(pred_seqs: List[List[int]], dataset_entries: List[Dict]) -> Dict:
    """Pair-level (triplet) precision / recall / F1."""
    ap = pp = tp = 0
    for entry, preds in zip(dataset_entries, pred_seqs):
        tokens = entry["tokens"]
        gold = entry["causal_relation_token_spans"]
        ap += len(gold)
        pred_triplets = tags_to_triplets(preds[: len(tokens)], tokens)
        if not pred_triplets:
            continue
        pp += len(pred_triplets)
        for g in gold:
            g_c = [tokens[i] for i in g[0]]
            g_e = [tokens[i] for i in g[1]]
            for p in pred_triplets:
                p_c = [tokens[i] for i in p[0] if i < len(tokens)]
                p_e = [tokens[i] for i in p[-1] if i < len(tokens)]
                if g_c == p_c and g_e == p_e:
                    tp += 1
    pre, rec, f1 = _f1(tp, ap, pp)
    return {"tri_precision": pre, "tri_recall": rec, "tri_f1": f1,
            "tri_tp": tp, "tri_ap": ap, "tri_pp": pp}


def compute_bio_f1(pred_seqs: List[List[int]], label_seqs: List[List[int]]) -> Dict:
    """Seqeval-style span-level F1 broken down by entity class."""
    from seqeval.metrics import classification_report

    true_strs, pred_strs = [], []
    flat_true, flat_pred = [], []
    for preds, labels in zip(pred_seqs, label_seqs):
        t_row, p_row = [], []
        for p, l in zip(preds, labels):
            if l == -100:
                continue
            t_row.append(ID2LABEL[l])
            p_row.append(ID2LABEL.get(p, "O"))
            flat_true.append(l)
            flat_pred.append(p)
        true_strs.append(t_row)
        pred_strs.append(p_row)

    report = classification_report(true_strs, pred_strs, output_dict=True, zero_division=0)

    cls_metrics = {}
    for name, ids in [("C", (1, 2)), ("E", (3, 4)), ("Emb", (5, 6))]:
        # Official evaluation_ctl: token-level per-tag counts summed over the
        # group's B-/I- tags — a true positive requires the EXACT tag to match
        # (B-C predicted where gold is I-C is not a tp), unlike group
        # membership matching.
        tp = sum(p == l and l in ids for p, l in zip(flat_pred, flat_true))
        ap = sum(l in ids for l in flat_true)
        pp = sum(p in ids for p in flat_pred)
        pre, rec, f1 = _f1(tp, ap, pp)
        cls_metrics.update({f"f1_{name}": f1, f"pre_{name}": pre, f"rec_{name}": rec})

    return {
        "f1_micro": report.get("micro avg", {}).get("f1-score", 0.0),
        "f1_macro": report.get("macro avg", {}).get("f1-score", 0.0),
        **cls_metrics,
    }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def encode_dataset(raw_dataset, tokenizer: SciteTokenizer) -> List[Dict]:
    """Apply ``tokenizer.encode`` to every example; returns plain list of dicts.

    Skips (with a printed count) any example that encodes to zero tokens --
    ``tokenizer.tokenize`` removes parenthesised content wholesale, matching
    the official SCITE preprocessing (data_prep.ipynb), so a "sentence"
    that is wholly a parenthetical aside (e.g. ``"(Fig. 3)"``,
    ``"(data not shown).\\n"``) encodes to an empty token list. SCITE's own
    corpus doesn't have these (its sentences come pre-segmented by the
    original annotators), but a caller re-splitting a whole document into
    sentences with a general-purpose segmenter can produce them (seen on
    BioCause). An empty ``tokens`` list would otherwise reach the CRF layer
    as a sequence with no real timestep at all, crashing deep inside
    torchcrf with an opaque "mask of the first timestep must all be on" --
    filtering here, once, is far clearer than that error.
    """
    encoded = []
    skipped_empty = 0
    for ex in raw_dataset:
        rels = list(ex.get("relations") or [])
        enc = tokenizer.encode(ex["text"], relations=rels)
        if not enc["tokens"]:
            skipped_empty += 1
            continue
        enc["index"] = len(encoded)
        encoded.append(enc)
    if skipped_empty:
        print(f"[encode_dataset] skipped {skipped_empty} example(s) that encoded to zero tokens")
    return encoded


# ---------------------------------------------------------------------------
# Trainer compute_metrics factory
# ---------------------------------------------------------------------------


def make_compute_metrics(val_fold: List[Dict]):
    """Return compute_metrics bound to the given encoded validation split.

    Examples are matched positionally: the i-th prediction corresponds to the
    i-th entry in val_fold.  This avoids the need for include_for_metrics.
    """
    def compute_metrics(eval_pred):
        # EvalPrediction unpacks as (predictions, label_ids)
        predictions = eval_pred.predictions
        label_ids = eval_pred.label_ids

        pred_seqs = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
        lab_seqs = label_ids.tolist() if hasattr(label_ids, "tolist") else list(label_ids)

        bio = compute_bio_f1(pred_seqs, lab_seqs)
        # Strip padding from pred_seqs before triplet eval
        clean_preds = [
            [p for p, l in zip(ps, ls) if l != -100]
            for ps, ls in zip(pred_seqs, lab_seqs)
        ]
        tri = compute_triplet_f1(clean_preds, val_fold)
        return {**bio, **tri}

    return compute_metrics


# ---------------------------------------------------------------------------
# LR scheduler callback — steps ReduceLROnPlateau on training loss each epoch
# (paper: halve LR if training loss does not fall for >10 epochs)
# ---------------------------------------------------------------------------


class ReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, scheduler: ReduceLROnPlateau) -> None:
        self.scheduler = scheduler

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs) -> None:
        train_loss = next(
            (e["loss"] for e in reversed(state.log_history) if "loss" in e),
            None,
        )
        if train_loss is not None:
            self.scheduler.step(train_loss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Reproduce SCITE training and evaluation.")
    p.add_argument("--dataset", default="thagen/SCITE")
    p.add_argument("--config", default="causality identification",
                   help="Dataset configuration name")
    p.add_argument("--word-embeddings", default=None,
                   help="Path to wiki-extvec numpy matrix [vocab, 300]. "
                        "Omit to use randomly initialised word embeddings.")
    p.add_argument("--embedding-source", default="bert", choices=["flair", "bert"])
    p.add_argument("--bert-model", default="google-bert/bert-base-uncased")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stopping", type=int, default=30)
    p.add_argument("--output-dir", default="./scite-output")
    p.add_argument("--smha-concat", action="store_true")
    p.add_argument("--no-word-embeddings", action="store_true",
                   help="Disable word embedding lookup table entirely.")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load raw dataset
    # ------------------------------------------------------------------
    print(f"\nLoading {args.dataset!r} / {args.config!r} …")
    raw = load_dataset(args.dataset, args.config)
    print(raw)

    # ------------------------------------------------------------------
    # 2. Set up tokeniser (+ BERT sub-word tokeniser if needed)
    # ------------------------------------------------------------------
    tokenizer = SciteTokenizer()
    bert_tok = None

    if args.embedding_source == "bert":
        print(f"\nLoading BERT tokeniser: {args.bert_model} …")
        bert_tok = AutoTokenizer.from_pretrained(args.bert_model)
        tokenizer.set_bert_tokenizer(bert_tok)

    # ------------------------------------------------------------------
    # 3. Build vocabulary from raw training text, then encode
    # ------------------------------------------------------------------
    _marker_re = re.compile(r"</?e\d+>")
    print("\nBuilding vocabulary …")
    pre_tokenized = [
        {"tokens": [t for t in tokenizer.tokenize(ex["text"]) if not _marker_re.fullmatch(t)]}
        for ex in raw["train"]
    ]
    tokenizer.build_vocab(pre_tokenized)
    print(f"  word vocab: {tokenizer.word_vocab_size}  char vocab: {tokenizer.char_vocab_size}")

    print("Encoding dataset …")
    encoded = {split: encode_dataset(raw[split], tokenizer) for split in raw}

    # ------------------------------------------------------------------
    # 4. Model config
    # ------------------------------------------------------------------
    use_word_emb = not args.no_word_embeddings
    config = SCITEConfig(
        word_vocab_size=tokenizer.word_vocab_size,
        char_vocab_size=tokenizer.char_vocab_size,
        use_word_embeddings=use_word_emb,
        embedding_source=args.embedding_source,
        bert_model_name=args.bert_model,
        smha_concat=args.smha_concat,
        num_labels=7,
        dropout_lstm=0.0,  # single-layer LSTM; dropout only applies between layers
    )

    collator = SciteDataCollator(
        scite_tokenizer=tokenizer,
        max_wlen=tokenizer.max_wlen,
        bert_tokenizer=bert_tok,
    )
    test_encoded = encoded.get("test", [])

    # ------------------------------------------------------------------
    # 5. K-fold cross-validation
    # ------------------------------------------------------------------
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    train_indices = np.arange(len(encoded["train"]))

    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_indices)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{args.folds}  |  train={len(tr_idx)}  val={len(val_idx)}")
        print(f"{'='*60}")

        train_fold = [encoded["train"][i] for i in tr_idx]
        val_fold = [encoded["train"][i] for i in val_idx]

        from datasets import Dataset as HFDataset
        train_hf = HFDataset.from_list(train_fold)
        val_hf = HFDataset.from_list(val_fold)

        # Model
        model = SCITEForTokenClassification(config, word_embeddings_path=args.word_embeddings)
        model.to(device)

        # Optimiser + scheduler
        optimizer = NAdam(model.parameters(), lr=args.lr)
        # Official: Keras ReduceLROnPlateau(monitor='loss' [min mode],
        # factor=0.5, patience=10, cooldown=5, min_lr=5e-5).  mode MUST be
        # "min" — the callback below steps this with the training loss.
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, cooldown=5, min_lr=5e-5
        )

        fold_output = Path(args.output_dir) / f"fold_{fold + 1}"
        training_args = TrainingArguments(
            output_dir=str(fold_output),
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="tri_f1",
            greater_is_better=True,
            remove_unused_columns=False,

            report_to="none",
            max_grad_norm=5.0,
            seed=args.seed,
            save_safetensors=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_hf,
            eval_dataset=val_hf,
            compute_metrics=make_compute_metrics(val_fold),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=args.early_stopping),
                ReduceLROnPlateauCallback(scheduler),
            ],
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

        # Test-set evaluation for this fold's best model
        if test_encoded:
            test_hf = HFDataset.from_list(test_encoded)
            # Use a fresh compute_metrics bound to the test split
            trainer.compute_metrics = make_compute_metrics(test_encoded)
            test_results = trainer.evaluate(eval_dataset=test_hf, metric_key_prefix="test")
            print(f"\nFold {fold + 1} test results:")
            for k, v in sorted(test_results.items()):
                if "runtime" not in k:
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # 6. Aggregate fold results
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Mean ± std across folds")
    print(f"{'='*60}")
    keys = [k for k in fold_results[0] if isinstance(fold_results[0][k], float) and "runtime" not in k]
    for k in sorted(keys):
        vals = [r[k] for r in fold_results]
        print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()
