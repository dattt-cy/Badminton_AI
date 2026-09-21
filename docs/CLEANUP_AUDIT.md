# Cleanup audit after project pivot

Date: 16/09/2026

## Removed now

- `README_PRESENTATION.md`: presentation for the previous two-class technique
  analysis direction.
- `docs/JHSE_Vol21_Issue2_Art1278.pdf`: local biomechanics-smash paper; the
  new core is match stroke recognition.
- `configs/action_recognition/experiments/stgcnpp_manual_clips_2class_temporal_augmented_single_clip_val.py`:
  untracked validation-only experiment for the old checkpoint.
- `scripts/evaluation/render_training_evidence.py`: untracked evidence renderer
  for the old two-class presentation.
- `work_dirs/stgcnpp_manual_clips_2class_temporal_augmented/`: old training log
  and checkpoint artifacts.
- Failed/partial BST experiments: `outputs/bst_088`,
  `outputs/bst_forehand_net_shot_7`, and the accidental parent-level output.
- `external/TrackNetV3/TrackNetV3_ckpts.zip`: redundant archive after the
  checkpoints had been extracted to `external/TrackNetV3/ckpts/`.
- Empty `ai_classifier.rag` package.

These generated/untracked items are not recoverable from the current Git
history. The old checkpoint must be restored from a separate backup or retrained
if it is ever needed again.

## Retained because it supports the new direction

- `pose`, `preprocessing`: player extraction and quality checks.
- `segmentation`: motion/event baselines.
- `action_recognition`: ST-GCN++ baseline infrastructure.
- `api`, web UI, job store and artifacts.
- pose viewer/export and visualization.
- BST and TrackNet repositories/checkpoints needed for reproducible baselines.
- `outputs/bst_089`: successful external-video proof of concept.
- dataset export/split infrastructure that is model-agnostic.

## Legacy code retained temporarily

The following domains are no longer part of the main research result, but are
not deleted yet because the current API backend and segmentation classifier
still import them:

- `src/ai_classifier/biomechanics`
- `src/ai_classifier/error_detection`
- `src/ai_classifier/reference_analysis`
- `configs/biomechanics`
- `scripts/inference/analyze_technique.py`
- MultiSense/reference generation and evaluation scripts/tests

Deleting these now would break the working upload pipeline. Remove them only
after the new `video -> StrokeEvent[]` backend has replaced
`analyze_technique.py` and motion features have moved into a neutral module.

## Required second cleanup

1. Introduce `StrokeEvent` and the new inference backend.
2. Move wrist/elbow motion features needed by segmentation out of
   `biomechanics`.
3. Change the API artifact contract from technique reports to stroke timeline
   artifacts.
4. Remove the legacy domains and their configs/scripts/tests in one commit.
5. Rewrite the root README for the new product before the next presentation.

## Workspace cleanup on 17/09/2026

- Added `data/cache/` to `.gitignore`; cached RGB crops are reproducible and
  must not appear in source control.
- Moved 7,779 cached crop files (about 4.47 GB) to the Windows Recycle Bin.
- Removed generated outputs from the previous technique-analysis,
  MultiSense, presentation, viewer-debug and API-debug workflows.
- Retained only outputs aligned with the current plan: Fine-Badminton,
  ShuttleSet, R(2+1)D-18 and the successful `bst_089` proof of concept.
- Removed duplicate smoke/pilot work directories and all old two-class
  ST-GCN++ work directories. Retained the full Fine-Badminton run and the
  latest resumed ShuttleSet run.
- Ignored the local BST and TrackNet vendor checkouts. Project changes belong
  in adapters under `src/ai_classifier/integrations/` or `scripts/`, not in
  those checkout directories.

## Workspace cleanup during hit-detector training

- Removed obsolete hit-verifier caches, smoke checkpoints, partial archive
  benchmarks, preview sheets, empty BST outputs, root-level test JSON files,
  and generated Python bytecode (about 1.98 GB).
- Removed embedded virtual environments from vendor checkouts (about 1.54 GB);
  dependencies remain reproducible from their requirement files.
- Removed the unused `external/badminton_shot_type` MMPose experiment and its
  Colab notebook (about 0.47 GB). The maintained TrackNetV3 integration and the
  BST stroke-classification baseline remain available.
- Preserved the epoch-20 mixed classifier, Fine-Badminton source checkpoint,
  current app-compatible fusion checkpoints, manifests, source videos, mixed
  RGB caches, and the active full hit-detector run/cache.
- Removed 24 tests tied exclusively to the retired biomechanics, MultiSense,
  manual two-class and legacy technique-evaluation workflows. Retained tests
  for the active API, pose/localization, ShuttleSet/Fine-Badminton datasets,
  RGB multitask classifier, hit detector, fusion, and core PySKL baseline.
- Removed the remaining PySKL/ST-GCN++, legacy `analyze_technique`, and
  motion-segmentation tests after the decision to stop maintaining those
  legacy test paths. No backup copy was retained.

The legacy source domains listed above are intentionally still present. Their
removal is blocked by imports in the current upload pipeline, so deleting them
before the `StrokeEvent[]` backend is in place would break the working product.

## Archive evaluation cleanup on 21/09/2026

The canonical external-archive evaluation path is now:

```text
HIT model -> learned selector -> motion tracking/crop -> FastTrackNet
          -> fusion -> physics refinement -> StrokeEvent result
```

- Retained `scripts/evaluation/evaluate_archive_fusion_from_hit_report.py` as
  the reproducible evaluator for this path.
- Removed the superseded RGB-only, legacy upload-pipeline, and window-oracle
  archive evaluators. Their generated reports remain historical evidence, not
  active production paths.
- Trajectory caches are keyed by the resolved video and selected event frame.
  A cache created for an older selector result must never be reused around a
  different hit frame.
- Evaluation reports now record model paths, pipeline stages, raw predictions,
  and physics adjustments so future slide metrics can be audited precisely.
