import sys
from enum import Enum, IntEnum

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        # Match Python 3.11 StrEnum semantics: str() yields the value, not
        # the member name ("causality detection", not "Task.CausalityDetection").
        def __str__(self) -> str:
            return str(self.value)


class Task(StrEnum):
    """The three tasks a dataset/model can support: detection, candidate extraction, identification."""

    CausalityDetection = "causality detection"
    CausalCandidateExtraction = "causal candidate extraction"
    CausalityIdentification = "causality identification"


class ClassLabel(IntEnum):
    """Sentence-level causality-detection label -- does the text discuss a causal relationship at all
    (asserted or refuted), independent of which relation type it turns out to be."""

    # The text does not contain any causal information
    Uncausal = 0
    # The text contains come causal information (causal or countercausal)
    Causal = 1


class Relation(IntEnum):
    """A relation *type*, in the set-theoretic sense: e.g. the causal relation is one particular subset
    of E x E (entities/events/concepts paired with entities/events/concepts). Contrast with a single
    *relationship* -- one element/pair of a relation, i.e. one instance of it (see the
    [glossary](../glossary.md))."""

    NoRelation = 0
    Causal = 1
    Countercausal = 2
