"""Structured tactic detectors.

Detectors in this package turn already-segmented stroke sequences into
frontend-ready tactical evidence. Match summaries and generic statistics live
in :mod:`ai_classifier.analytics.match_analysis` instead.
"""

from .coachai_plus import TACTIC_LABELS, analyze_coachai_plus_tactics

__all__ = ["TACTIC_LABELS", "analyze_coachai_plus_tactics"]
