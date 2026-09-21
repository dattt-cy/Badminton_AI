# Script index

Run commands from the repository root after `python -m pip install -e .`.

## Long-video stroke timeline

The canonical inference path is:

```text
HIT scan -> learned selector -> FastTrackNet -> fusion -> physics -> timeline.json
```

Run it on a rally, match segment, or a video of at least 10 minutes:

```powershell
python scripts/inference/analyze_long_video.py "C:\path\match.mp4" `
  --output-dir work_dirs/long_video_eval `
  --hit-checkpoint work_dirs/r2plus1d18_hit_full/best.pth `
  --hit-selector outputs/hit_selector_val/selector.json
```

The output directory stores the dense HIT scan, event-level results,
event-frame-aware trajectory caches, and the resumable `timeline.json`. Re-run
the same command to continue incomplete work. Use `--force-scan` or
`--force-events` only when the corresponding cached stage must be rebuilt.

## Data workflows

- `data/pose/`: batch pose extraction for raw, manual, and reference clips.
- `data/augmentation/`: temporal/motion-onset variants and class materialization.
- `data/datasets/`: MultiSense clip construction, legacy migration, overfit
  subsets, and dataset inventory.
- `data/export/`: PySKL annotations and 2D/3D feature tables.
- `data/references/`: candidate selection and biomechanics reference profiles.

Build validation-only MultiSense 3D expert ranges from exported phase features:

```powershell
python scripts/data/references/build_multisense_3d_reference.py `
  outputs/multisense_part2_correction_audit.csv `
  --holdout-csv outputs/multisense_part3_correction_audit.csv `
  --output-yaml configs/biomechanics/multisense_3d_expert_reference.yaml `
  --output-csv outputs/multisense_3d_expert_reference.csv
```

The defaults retain only `Expert`, valid-phase, `Good`-contact strokes with
contact confidence at least 0.5. These 3D ranges are for validation or
2D-to-3D-corrected features, never direct comparison with raw 2D angles.
Part2 fits the intervals and Part3 is reported only as an independent holdout.
The generated profile remains `provisional` when holdout feature coverage is
below the 70% validation target. After reporting the holdout result, a separate
deployment candidate may be refit on both parts and should use an explicit
`*_combined.yaml` filename; this does not replace the preserved holdout report.

Build timestamp-paired 2D/3D elbow samples with wrist-motion alignment, then
train a fail-closed correction audit:

```powershell
python scripts/data/export/build_multisense_2d3d_pairs.py `
  data/expert_reference/multisense outputs/multisense_2d3d_motion_aligned_pairs.csv `
  --feature-csv outputs/multisense_part2_correction_audit.csv `
                  outputs/multisense_part3_correction_audit.csv `
  --archive-root "C:\Users\ADMIN\Downloads\Data Archive Part2" `
                 "C:\Users\ADMIN\Downloads\Data Archive Part3"

python scripts/data/references/train_multisense_2d3d_correction.py `
  outputs/multisense_2d_3d_elbow_validation.csv `
  outputs/multisense_elbow_2d3d_correction_audit.json
```

A correction is marked `ready` only when subject-held-out MAE improves at
least 10% over raw 2D, beats the training-median prior by 5%, and improves P90.
Rejected models are never applied by `predict_elbow_3d`.

Existing manually cut YOLOv8n poses can also be paired without cutting or
extracting video again. The exporter parses each clip's `HH.MM.SS.mmm` range,
aligns all clip intervals to the HDF5 annotations, then refines locally by
wrist-motion correlation:

```powershell
python scripts/data/export/build_multisense_2d3d_pairs.py `
  data/processed/manual_clips_2class_poses `
  outputs/multisense_manual_clips_2d3d_pairs_aligned.csv `
  --feature-csv outputs/multisense_part2_correction_audit.csv `
                  outputs/multisense_part3_correction_audit.csv `
  --archive-root "C:\Users\ADMIN\Downloads\Data Archive Part2" `
                 "C:\Users\ADMIN\Downloads\Data Archive Part3" `
  --min-alignment-correlation 0.30 `
  --pair-mode frames --frame-stride 3
```

Frame mode keeps only frames inside the matched annotation stroke and adds
observable 2D shoulder, reach, elbow-to-torso, wrist-height, and torso features.
The default runtime gate additionally requires holdout MAE <= 15 degrees,
P90 <= 30 degrees, and at least 50% of errors within 10 degrees. Relative
improvement alone is not sufficient to activate a correction model.

Build the Part3 holdout feature registry used by the 2D rule engine:

```powershell
python scripts/evaluation/build_multisense_feature_validation.py `
  outputs/multisense_projected_perspective_guarded.csv `
  configs/biomechanics/multisense_feature_validation_projected.yaml
```

`check_technique_rules.py` loads this registry by default. A `validated`
feature can produce pass/deviation, `review` is forced to review, and a
`rejected` or missing feature becomes insufficient data. Use
`--no-feature-validation` only for controlled comparisons with the old output.

Generate the same-space comparison table before building that registry:

```powershell
python scripts/data/export/project_multisense_3d_to_video.py `
  data/processed/manual_clips_2class_poses `
  outputs/multisense_projected_perspective_guarded.csv `
  --feature-csv outputs/multisense_part2_correction_audit.csv `
                  outputs/multisense_part3_correction_audit.csv `
  --archive-root "C:\Users\ADMIN\Downloads\Data Archive Part2" `
                 "C:\Users\ADMIN\Downloads\Data Archive Part3" `
  --min-alignment-correlation 0.30 --frame-stride 3 `
  --camera-model perspective
```

Camera fitting uses shoulders, hips, knees, and ankles. Racket-arm joints are
excluded from fitting and retained for independent evaluation. The validation
registry applies stricter holdout gates (motion correlation >= 0.50, median
camera error <= 0.15 torso lengths, P90 <= 0.30).
Perspective projection is the default after its holdout audit retained more
clips and reduced rejected feature/view combinations versus weak perspective.

Audit joint mapping, residual timing, and projection overlays:

```powershell
python scripts/evaluation/audit_multisense_projection.py `
  outputs/multisense_projected_perspective_guarded.csv `
  outputs/multisense_projection_perspective_audit.json `
  --overlay-dir outputs/multisense_projection_perspective_overlays
```

Overlays render YOLO in green and projected MultiSense in magenta. The audit
also compares normal versus left/right-swapped arm mapping and estimates
residual frame lag without changing runtime calibration.

Build relative 2D indicators directly from all manual pose clips. Skill level
is joined by subject and technique, so this step does not require exact HDF5
timestamps:

```powershell
python scripts/data/export/export_multisense_relative_2d_features.py `
  outputs/multisense_relative_2d_all_manual_strokes.csv `
  --pose-root data/processed/manual_clips_2class_poses `
  --annotation-csv outputs/multisense_part2_correction_audit.csv `
                   outputs/multisense_part3_correction_audit.csv

python scripts/evaluation/build_relative_2d_rules.py `
  outputs/multisense_relative_2d_all_manual_strokes.csv `
  configs/biomechanics/multisense_relative_2d_rules.yaml
```

Threshold directions have biomechanics sanity constraints and must work on
both Part2 and the subject-independent Part3 holdout. Runtime presents these
as pass/review indicators only; failed indicators never become hard coaching
deviations.

Technique reports also include a score-free `user_feedback` block intended
for the phone-video pilot. It separates validated good signals, measurements
shown only for review, and unavailable checks with plain-language reasons.
Rejected experimental indicators remain in the technical audit but are hidden
from this user-facing block.

The phone-video preview adds technique-specific observable criteria with
`experimental_heuristic` evidence level. Forehand clear checks overhead and
forward contact, non-racket-arm sequencing, leg loading, body transfer, and
follow-through. Backhand drive replaces overhead contact with a drive-height
zone and uses the non-racket arm as a counterbalance cue. These observations
are displayed separately from validated good signals and never create a
numeric score or a hard deviation.

## End-to-end phone video preview

Run pose extraction, quality gates, technique rules, observable criteria, and
the score-free user report with one command:

```powershell
python scripts/inference/analyze_technique.py `
  "C:\Users\ADMIN\Downloads\stroke.mp4" auto `
  --view side --handedness right --run-classifier-wsl
```

`auto` is also the default when the positional technique is omitted. An
accepted classifier prediction selects the rule set; below the confidence
gate the command stops and asks the caller to provide a technique explicitly.

The default output is `outputs/<video>_analysis/` with `pose.npz`,
`pose_preview.mp4`, `quality.json`, `technique_report.json`, and the combined
`analysis.json`. Use `--pose existing_pose.npz` to skip pose extraction and
`--no-preview` to skip rendering. Technique, view, and handedness are
user-confirmed; an optional `--classifier-json` is recorded as a suggestion
and never selects the rule set.
The WSL option uses the two-class motion-onset checkpoint by default and
writes `classification.json`. `analysis.json` exposes a compact `user_summary`
alongside the complete technical report. `user_report.json` is the compact
frontend payload; `user_report.md` is the Vietnamese report a user can read
directly without understanding internal rule names.

The end-to-end command defaults to `configs/pose/yolov8.yaml`, matching the
YOLOv8n/640 pose distribution used to train the action checkpoint. This same
pose is reused by the quality and geometry stages, avoiding duplicate pose
extraction and inconsistent classifier confidence. The stricter
`yolov8_high_accuracy.yaml` profile remains available through `--pose-config`
for controlled geometry experiments, but should not feed the current action
checkpoint unless that checkpoint is retrained on the same profile.

## Model workflows

- `training/`: compatibility wrappers around external training frameworks.
- `evaluation/`: feature audits, feasibility checks, contact sheets, and rule
  evaluation.
- `inference/`: pose extraction, pose rendering, and action classification.

Scripts are entry points, not a second application package. Shared logic used
by multiple scripts belongs in `src/ai_classifier/`.
