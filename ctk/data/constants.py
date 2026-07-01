from enum import IntEnum
try:
    from enum import StrEnum
except ImportError:  # Python < 3.11
    from enum import Enum

    class StrEnum(str, Enum):
        pass


class Task(StrEnum):
    CausalityDetection: str = "causality detection"
    CausalCandidateExtraction: str = "causal candidate extraction"
    CausalityIdentification: str = "causality identification"


class ClassLabel(IntEnum):
    # The text does not contain any causal information
    Uncausal: int = 0
    # The text contains come causal information (pro- or concausal)
    Causal: int = 1


class Relation(IntEnum):
    NoRelation: int = 0
    Procausal: int = 1
    Concausal: int = 2
