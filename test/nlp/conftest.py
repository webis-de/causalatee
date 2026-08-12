"""Shared setup for test/nlp/: both test_connectives.py and test_sdp.py load a spacy model at
module import time (once per test session, not per test)."""

from __future__ import annotations

import spacy


def load_spacy_model(name: str) -> spacy.language.Language:
    """Load a spacy model, downloading it first if it isn't installed yet.

    The ``baselines`` extra installs the spacy library but not any language model -- models are
    separate pip packages, fetched on demand rather than declared as a dependency, same as how
    HuggingFace model weights aren't declared dependencies either.
    """

    try:
        return spacy.load(name)
    except OSError:
        spacy.cli.download(name)
        return spacy.load(name)
