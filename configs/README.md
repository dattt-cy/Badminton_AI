# Configuration layout

- `pose/`: pose estimator and smoothing presets.
- `action_recognition/datasets/`: dataset discovery and export settings.
- `action_recognition/experiments/`: PySKL/ST-GCN++ training configurations.
- `biomechanics/`: technique registry and reference profiles by technique.
- `rag/`: retrieval configuration when that subsystem is implemented.

Paths are repository-relative so commands should be run from the project root.
Create a new experiment by extending a nearby config in `experiments/`; do not
mix dataset preparation settings into model training configs.
