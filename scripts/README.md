# Operator and research scripts

Scripts are thin command-line entrypoints, not importable application modules.

- `data/`: dataset download, audit and feature preparation.
- `evaluation/`: benchmarks, comparisons and report generation.
- `inference/`: manual/operator inference commands.
- `training/`: experiment and training commands.

Reusable behavior belongs under `src/ai_classifier`. Production services must
not import modules from this directory.
