"""Shared types for rule-based technique error detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RangeCheck:
    """Reference range for one scalar feature.

    `buffer_ratio` widens the range (mục 12.4) so values just outside the
    reference are flagged as `review` instead of a confident `deviation`.
    """

    low: float
    high: float
    buffer_ratio: float = 0.05
    direction: str = "two_sided"  # "lower", "upper", or "two_sided"

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.buffer_ratio < 0:
            raise ValueError("buffer_ratio must be >= 0")
        if self.direction not in {"lower", "upper", "two_sided"}:
            raise ValueError("direction must be lower, upper, or two_sided")


@dataclass(frozen=True)
class ErrorEvent:
    """One rule violation: an observed value outside its reference range."""

    rule_name: str
    message: str
    observed: float
    reference_low: float
    reference_high: float
    severity: str  # "deviation" or "review"


@dataclass(frozen=True)
class RuleDefinition:
    """Declarative technique rule loaded from a reference YAML file."""

    name: str
    feature: str
    phase: str
    direction: str
    reference: RangeCheck
    min_confidence: float = 0.6
    min_valid_frames: int = 3
    min_valid_ratio: float = 0.6
    compatible_views: tuple[str, ...] = ("front", "side")


@dataclass(frozen=True)
class RuleResult:
    rule_name: str
    status: Literal["pass", "review", "deviation", "insufficient_data"]
    observed: float | None
    reference_low: float
    reference_high: float
    valid_frames: int
    reason: str | None = None
