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
    CausalityDetection = "causality detection"
    CausalCandidateExtraction = "causal candidate extraction"
    CausalityIdentification = "causality identification"


class ClassLabel(IntEnum):
    # The text does not contain any causal information
    Uncausal = 0
    # The text contains come causal information (pro- or concausal)
    Causal = 1


class Relation(IntEnum):
    NoRelation = 0
    Procausal = 1
    Concausal = 2
