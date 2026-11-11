#!/usr/bin/env bash
# Uploads the worker's checkpoints to a GitHub Release of Badminton_AI, where setup_gpu_server.sh downloads them.
# Run from the AI_Classifier folder on the laptop (Git Bash), with gh logged in:
#   bash scripts/deploy/publish_models.sh            # creates or updates release models-v1
#   bash scripts/deploy/publish_models.sh models-v2  # after retraining, then set MODELS_TAG=models-v2 on the server
set -euo pipefail

TAG="${1:-models-v1}"
REPO="dattt-cy/Badminton_AI"
CKPT_DIR="src/ai_classifier/models/checkpoints"
FILES=(hit_detector.pth kap_classifier.pth yolov8n-pose.pt tracknet_best.pt visual_backbone.pth rally_boundary_detector.joblib)

cd "$CKPT_DIR"
for f in "${FILES[@]}"; do [ -f "$f" ] || { echo "Thiếu $CKPT_DIR/$f" >&2; exit 1; }; done
# Git Bash marks names with "*" (binary mode); plain "hash  name" lines read the same everywhere.
sha256sum "${FILES[@]}" | sed 's/ \*/  /' > SHA256SUMS
cat SHA256SUMS

if ! gh release view "$TAG" -R "$REPO" >/dev/null 2>&1; then
    gh release create "$TAG" -R "$REPO" --title "Model checkpoints ($TAG)" \
        --notes "Checkpoints used by the PBL6 AI worker. Downloaded by scripts/deploy/setup_gpu_server.sh."
fi
gh release upload "$TAG" -R "$REPO" --clobber SHA256SUMS "${FILES[@]}"
rm SHA256SUMS
echo "Đã đưa model lên https://github.com/$REPO/releases/tag/$TAG"
