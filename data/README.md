# Data layout

- `raw/`: immutable source recordings.
- `interim/`: temporary conversion results.
- `processed/`: pose arrays and reusable derived features.
- `annotations/`: CSV metadata and generated PySKL annotation files.
- `manual_clips*`: curated or augmented manual-clip datasets.
- `multisense/`: local MultiSense source material.
- `expert_reference/`: expert samples used to build reference profiles.
- `inference/`: local inputs used for manual inference checks.

Large recordings and generated arrays are ignored by Git. Keep only small,
reviewable metadata files under version control. Generated run reports belong
in `outputs/`, not here.
