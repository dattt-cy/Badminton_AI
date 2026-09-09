"""Validated, declarative catalogue of techniques and their reference assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TechniqueRuleSpec:
    name: str
    feature: str
    phase: str
    direction: str
    min_confidence: float = 0.6
    min_valid_frames: int = 3
    min_valid_ratio: float = 0.6
    compatible_views: tuple[str, ...] = ("front", "side")


@dataclass(frozen=True)
class TechniqueView:
    reference: Path
    clip_list: Path | None = None


@dataclass(frozen=True)
class TechniqueDefinition:
    name: str
    display_name: str
    pose_dir: Path
    rules: tuple[TechniqueRuleSpec, ...]
    views: dict[str, TechniqueView]
    minimum_reference_clips: int = 10


@dataclass(frozen=True)
class TechniqueRegistry:
    path: Path
    techniques: dict[str, TechniqueDefinition]

    def technique(self, name: str) -> TechniqueDefinition:
        try:
            return self.techniques[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.techniques))
            raise ValueError(f"Unknown technique '{name}'. Available: {available}") from exc


def load_technique_registry(path: Path) -> TechniqueRegistry:
    """Load and validate a registry; relative asset paths use the project cwd."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_techniques = document.get("techniques")
    if not isinstance(raw_techniques, dict) or not raw_techniques:
        raise ValueError(f"Registry must contain a non-empty 'techniques' mapping: {path}")

    techniques: dict[str, TechniqueDefinition] = {}
    for name, raw in raw_techniques.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Technique '{name}' must be a mapping")
        raw_rules = raw.get("rules", [])
        raw_views = raw.get("views", {})
        if not raw_rules:
            raise ValueError(f"Technique '{name}' has no rules")
        if not raw_views:
            raise ValueError(f"Technique '{name}' has no views")

        rules = tuple(_load_rule(name, item) for item in raw_rules)
        rule_names = [rule.name for rule in rules]
        if len(rule_names) != len(set(rule_names)):
            raise ValueError(f"Technique '{name}' has duplicate rule names")
        views = {
            view_name: TechniqueView(
                reference=Path(entry["reference"]),
                clip_list=Path(entry["clip_list"]) if entry.get("clip_list") else None,
            )
            for view_name, entry in raw_views.items()
        }
        techniques[name] = TechniqueDefinition(
            name=name,
            display_name=str(raw.get("display_name", name)),
            pose_dir=Path(raw.get("pose_dir", f"data/processed/poses/{name}/single_player")),
            rules=rules,
            views=views,
            minimum_reference_clips=int(raw.get("minimum_reference_clips", 10)),
        )
    return TechniqueRegistry(path=path, techniques=techniques)


def _load_rule(technique: str, raw: dict[str, Any]) -> TechniqueRuleSpec:
    required = {"name", "feature", "phase", "direction"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Technique '{technique}' rule is missing: {sorted(missing)}")
    direction = str(raw["direction"])
    if direction not in {"lower", "upper", "two_sided"}:
        raise ValueError(f"Invalid direction '{direction}' in technique '{technique}'")
    views = tuple(raw.get("compatible_views", ("front", "side")))
    if not views:
        raise ValueError(f"Rule '{raw['name']}' must have compatible_views")
    min_valid_ratio = float(raw.get("min_valid_ratio", 0.6))
    if not 0.0 <= min_valid_ratio <= 1.0:
        raise ValueError(f"Rule '{raw['name']}' min_valid_ratio must be between 0 and 1")
    return TechniqueRuleSpec(
        name=str(raw["name"]), feature=str(raw["feature"]), phase=str(raw["phase"]),
        direction=direction, min_confidence=float(raw.get("min_confidence", 0.6)),
        min_valid_frames=int(raw.get("min_valid_frames", 3)),
        min_valid_ratio=min_valid_ratio,
        compatible_views=views,
    )
