"""Cross-sectional factor research with look-ahead detection built in."""

from .panel import Panel, synthetic_panel
from .lookahead import detect, assert_clean, LookaheadReport
from . import factors, evaluate

__all__ = ["Panel", "synthetic_panel", "detect", "assert_clean",
           "LookaheadReport", "factors", "evaluate"]
