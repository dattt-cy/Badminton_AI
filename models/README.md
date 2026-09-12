# Model artifacts

- `checkpoints/pose/`: downloaded pose-estimation weights.
- `checkpoints/action_recognition/`: trained classifier weights.
- `exports/`: deployable ONNX, TensorRT, or other packaged models.

Weights are generated/downloaded artifacts and are intentionally ignored by
Git. Keep experiment logs in `work_dirs/` and inference results in `outputs/`.
