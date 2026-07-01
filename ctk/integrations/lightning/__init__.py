try:
    import torch  # noqa: F401
    import torchmetrics  # noqa: F401
except ImportError:
    raise ImportError(
        "The Lightning integration requires torch and torchmetrics.\n"
        "Install them with: pip install 'causalatee[lightning]'"
    ) from None

from ._evaluation import SpanMetric

__all__ = ["SpanMetric"]
