# Project structure

The repository follows three boundaries: reusable code, executable workflows,
and generated artifacts. Keeping those boundaries explicit prevents notebooks
and one-off data jobs from becoming hidden production dependencies.

## Reusable code

All importable application code belongs in `src/ai_classifier/` and is grouped
by domain:

- `pose/`: pose extraction interfaces, YOLO implementation, and rendering.
- `preprocessing/`: transformations applied to pose sequences.
- `segmentation/`: motion proposals and temporal boundary refinement.
- `action_recognition/`: PySKL conversion and ST-GCN++ inference.
- `biomechanics/`: 2D/3D features, phases, and technique definitions.
- `reference_analysis/`: aggregate expert samples into reference profiles.
- `error_detection/`: typed findings and rule evaluation.
- `inference/`: future orchestration for the end-to-end runtime.
- `rag/`: future retrieval and coaching-response generation.
- `common/`: only genuinely cross-domain schemas and utilities.

Code used by more than one CLI must be moved into the appropriate source
domain. Files under `scripts/` should remain thin entry points.

## Workflows and configuration

`scripts/` is organized by the operation being run. Data commands are split
into `augmentation`, `datasets`, `export`, `pose`, and `references`; training,
evaluation, and inference remain separate top-level workflows. See
`scripts/README.md` for the command index.

`configs/action_recognition/datasets/` describes how datasets are discovered
and exported. `configs/action_recognition/experiments/` contains executable
PySKL experiment configurations. Technique-specific biomechanics references
stay under `configs/biomechanics/<technique>/`.

## Data and artifacts

The intended flow is:

```text
data/raw -> data/interim -> data/processed -> data/annotations
                                      |
                                      +-> models/checkpoints -> models/exports
                                      +-> outputs
```

- `raw`: immutable source recordings.
- `interim`: temporary or partially transformed data.
- `processed`: reusable pose tensors and feature tables.
- `annotations`: versionable CSV metadata and generated PySKL annotations.
- `models/checkpoints`: downloaded or trained model weights, grouped by model.
- `models/exports`: deployable model packages.
- `outputs`: disposable run results, previews, logs, and reports.
- `work_dirs`: disposable framework training state.

Large data, weights, and run outputs are ignored by Git. Do not put a model
checkpoint or generated report at repository root.

## Tests

`tests/unit/` mirrors the domains in `src/ai_classifier/`. Tests for executable
workflows live under `tests/unit/scripts/`. Cross-component tests belong in
`tests/integration/`.

When adding a feature, add its reusable logic to one source domain, its CLI to
one workflow directory, its configuration to the matching config group, and
its tests to the mirrored test directory.
