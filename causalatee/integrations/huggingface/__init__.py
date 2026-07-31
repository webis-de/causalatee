try:
    from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification

    # PIPELINE_REGISTRY is not re-exported from the transformers top level anymore
    # as of transformers 4.57; transformers.pipelines works for old versions too.
    from transformers.pipelines import PIPELINE_REGISTRY
except ImportError:
    raise ImportError(
        "The HuggingFace integration requires transformers.\nInstall it with: pip install 'causalatee[huggingface]'"
    ) from None

from ._candidate_extraction import CausalCandidateExtractionPipeline
from ._causality_extraction import CausalityExtractionPipeline
from ._detection import CausalityDetectionPipeline
from ._evaluation import span_compute_metrics
from ._identification import CausalityIdentificationPipeline

PIPELINE_REGISTRY.register_pipeline(
    "causality-detection",
    pipeline_class=CausalityDetectionPipeline,
    pt_model=AutoModelForSequenceClassification,
)
PIPELINE_REGISTRY.register_pipeline(
    "causal-candidate-extraction",
    pipeline_class=CausalCandidateExtractionPipeline,
    pt_model=AutoModelForTokenClassification,
)
PIPELINE_REGISTRY.register_pipeline(
    "causality-identification",
    pipeline_class=CausalityIdentificationPipeline,
    pt_model=AutoModelForSequenceClassification,
)

__all__ = [
    "CausalityDetectionPipeline",
    "CausalCandidateExtractionPipeline",
    "CausalityIdentificationPipeline",
    "CausalityExtractionPipeline",
    "span_compute_metrics",
]
