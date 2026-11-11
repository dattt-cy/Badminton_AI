#!/usr/bin/env bash
# Sets up a fresh rented Ubuntu GPU machine (EzyCloudX, AWS, ...) to run the PBL6 AI worker, then starts it.
#
#   1. Paste the worker's .env (backend/integrations/ai-worker/.env from the laptop) into ~/.env
#   2. curl -fsSL https://raw.githubusercontent.com/dattt-cy/Badminton_AI/main/scripts/deploy/setup_gpu_server.sh | bash
#
# Running it again on the same machine is safe: it updates the code and skips what is already there.
# Logs of the running worker:  journalctl -u pbl6-worker -f
set -euo pipefail

ENV_FILE="${ENV_FILE:-$HOME/.env}"
BASE="${BASE:-$HOME/pbl6}"
MODELS_TAG="${MODELS_TAG:-models-v1}"
PBL6_REPO="https://github.com/chugl3-founder/PBL6.git"
AI_REPO="https://github.com/dattt-cy/Badminton_AI.git"
TRACKNET_REPO="https://github.com/qaz812345/TrackNetV3.git"
TRACKNET_COMMIT="6eda442ada1740573f200f836d93edc9a541ee86"
MODELS_URL="https://github.com/dattt-cy/Badminton_AI/releases/download/$MODELS_TAG"

AI_DIR="$BASE/AI_Classifier"
WORKER_DIR="$BASE/PBL6/backend/integrations/ai-worker"
CKPT_DIR="$AI_DIR/src/ai_classifier/models/checkpoints"
VENV="$BASE/venv"
PY="$VENV/bin/python"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mLỖI: %s\033[0m\n' "$*" >&2; exit 1; }

SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"

step "Kiểm tra máy"
[ -f "$ENV_FILE" ] || die "Chưa có $ENV_FILE. Hãy dán nội dung file .env của worker vào đó trước (nano $ENV_FILE)."
command -v nvidia-smi >/dev/null || die "Không thấy nvidia-smi: máy này chưa có driver NVIDIA."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

step "Cài gói hệ thống (git, ffmpeg, python3-venv, ...)"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq git curl ffmpeg libgl1 libglib2.0-0 python3-venv python3-pip >/dev/null

step "Tải code (PBL6, AI_Classifier, TrackNetV3)"
mkdir -p "$BASE"
clone_or_pull() {  # url dir
    if [ -d "$2/.git" ]; then git -C "$2" pull --ff-only -q; else git clone -q --depth 1 "$1" "$2"; fi
}
clone_or_pull "$PBL6_REPO" "$BASE/PBL6"
clone_or_pull "$AI_REPO" "$AI_DIR"
if [ ! -d "$AI_DIR/external/TrackNetV3/.git" ]; then
    git clone -q "$TRACKNET_REPO" "$AI_DIR/external/TrackNetV3"
fi
git -C "$AI_DIR/external/TrackNetV3" checkout -q "$TRACKNET_COMMIT"

step "Cài Python packages (torch 2.8 cu128 khoảng 3 GB, lâu nhất ở bước này)"
[ -x "$PY" ] || python3 -m venv "$VENV"
"$PY" -m pip install -q --upgrade pip
# cu128 is required by RTX 50xx (Blackwell) and also runs older cards such as the T4.
"$PY" -m pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -q -r "$AI_DIR/requirements.txt" -r "$WORKER_DIR/requirements.txt"
# yt-dlp needs a JavaScript runtime for YouTube; the deno wheel puts one in the venv.
"$PY" -m pip install -q "yt-dlp[default]" deno

step "Tải model ($MODELS_TAG)"
mkdir -p "$CKPT_DIR"
curl -fsSL -o "$CKPT_DIR/SHA256SUMS" "$MODELS_URL/SHA256SUMS"
while read -r _ name; do
    name="${name#\*}"
    if (cd "$CKPT_DIR" && grep " $name\$" SHA256SUMS | sha256sum -c --status 2>/dev/null); then
        echo "  $name: đã có"
    else
        echo "  $name: đang tải"
        curl -fL --retry 3 -# -o "$CKPT_DIR/$name" "$MODELS_URL/$name"
    fi
done < "$CKPT_DIR/SHA256SUMS"
(cd "$CKPT_DIR" && sha256sum -c --quiet SHA256SUMS) || die "Model tải về bị lỗi, hãy chạy lại script."

step "Cấu hình worker"
# The laptop's AI_CLASSIFIER_ROOT is a Windows path; point it at this machine's copy.
grep -v '^AI_CLASSIFIER_ROOT=' "$ENV_FILE" > "$WORKER_DIR/.env"
{
    echo "AI_CLASSIFIER_ROOT=$AI_DIR"
    echo "AI_WARMUP_VIDEO=$BASE/PBL6/demo/videos/demo_8s.mp4"
} >> "$WORKER_DIR/.env"
chmod 600 "$WORKER_DIR/.env"

step "Kiểm tra GPU và tải trước model EasyOCR"
"$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch không thấy GPU"
print(f"  torch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}")
import easyocr
easyocr.Reader(["en"], gpu=True, verbose=False)
print("  EasyOCR sẵn sàng")
EOF

step "Chạy thử pipeline trên clip 8 giây"
(cd "$WORKER_DIR" && "$PY" - <<'EOF'
import os, tempfile, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
import worker
clip = Path(os.environ["AI_WARMUP_VIDEO"])
started = time.time()
with tempfile.TemporaryDirectory() as tmp:
    result = worker.run_ai_pipeline(clip, Path(tmp) / "out.json", "Upper", "Lower")
print(f"  OK sau {time.time() - started:.0f}s")
EOF
) || die "Pipeline chạy thử bị lỗi (xem log ở trên)."

step "Bật worker chạy nền"
if pidof systemd >/dev/null 2>&1; then
    $SUDO tee /etc/systemd/system/pbl6-worker.service >/dev/null <<EOF
[Unit]
Description=PBL6 AI worker
After=network-online.target
Wants=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$WORKER_DIR
Environment=PYTHONUNBUFFERED=1
Environment=PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$PY worker.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable -q pbl6-worker
    $SUDO systemctl restart pbl6-worker
    LOGS="journalctl -u pbl6-worker -f"
else
    # Containers without systemd: plain background process.
    pkill -f "$PY worker.py" 2>/dev/null || true
    (cd "$WORKER_DIR" && PATH="$VENV/bin:$PATH" nohup "$PY" worker.py > "$BASE/worker.log" 2>&1 &)
    LOGS="tail -f $BASE/worker.log"
fi

printf '\n\033[1;32mXONG. Worker đang chạy và nghe hàng đợi CloudAMQP.\033[0m\n'
echo "  Xem log:   $LOGS"
echo "  Nhớ tắt worker trên laptop (chạy START_WEB_ONLY.bat thay cho START_ALL_SERVICES.bat)."
