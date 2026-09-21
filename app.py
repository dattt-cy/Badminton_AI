"""
FastAPI backend for Badminton Shot Classification.
Provides endpoints to upload video and get classification results.
"""
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.inference.auto_court_detection import detect_court_corners, draw_court_corners

app = FastAPI(title="Badminton Shot Classifier")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Serve static files (HTML frontend)
STATIC_DIR = REPO_ROOT / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Job storage
JOBS_DIR = REPO_ROOT / "outputs" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = REPO_ROOT / "outputs" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

TRACKNET_SCRIPT = REPO_ROOT / "external" / "TrackNetV3" / "predict.py"
TRACKNET_MODEL = REPO_ROOT / "external" / "TrackNetV3" / "ckpts" / "TrackNet_best.pt"
INPAINT_MODEL = REPO_ROOT / "external" / "TrackNetV3" / "ckpts" / "InpaintNet_best.pt"
EVENTS_SCRIPT = REPO_ROOT / "scripts" / "inference" / "analyze_shuttle_trajectory_events.py"
FUSION_SCRIPT = REPO_ROOT / "scripts" / "inference" / "classify_shuttleset_fusion_video.py"
RGB_CHECKPOINT = REPO_ROOT / "work_dirs" / "r2plus1d18_shuttleset_mixed_crop_full_e8_e10_b4" / "best.pth"


def job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def write_job(job_id: str, data: dict):
    with open(job_path(job_id), "w") as f:
        json.dump(data, f, indent=2)


def read_job(job_id: str) -> dict:
    p = job_path(job_id)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _run_cmd(cmd: list, timeout: int = 600) -> tuple[int, str, str]:
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=str(REPO_ROOT)
    )
    return result.returncode, result.stdout, result.stderr


async def process_video(job_id: str, video_path: Path):
    """Full pipeline: auto court detect → TrackNet → events → Fusion"""
    job = read_job(job_id)

    try:
        # Step 1: Auto detect court corners
        write_job(job_id, {**job, "status": "processing", "step": "Đang nhận diện mặt sân...", "progress": 10})
        try:
            corners = detect_court_corners(str(video_path))
            ordered = corners.tolist()  # [[BL], [BR], [TR], [TL]]
            # Save court visualization
            cap = cv2.VideoCapture(str(video_path))
            ok, frame = cap.read()
            cap.release()
            if ok:
                vis = draw_court_corners(frame, corners)
                court_img_path = JOBS_DIR / f"{job_id}_court.jpg"
                cv2.imwrite(str(court_img_path), vis)
            court_ok = True
        except ValueError as e:
            # Use default fallback corners
            ordered = [[520, 900], [1400, 900], [1620, 500], [300, 500]]
            court_ok = False

        flat_corners = [v for pt in ordered for v in pt]

        # Step 2: TrackNet
        write_job(job_id, {**read_job(job_id), "step": "Đang theo dõi quỹ đạo cầu (TrackNet)...", "progress": 20,
                            "court_auto": court_ok, "corners": ordered})
        tracknet_out = JOBS_DIR / f"{job_id}_tracknet"
        tracknet_out.mkdir(exist_ok=True)

        # Find output csv - TrackNet saves alongside video due to Windows path quirk
        stem = video_path.stem
        expected_csv = video_path.parent / f"{stem}_ball.csv"

        if not expected_csv.exists():
            code, out, err = _run_cmd([
                sys.executable, str(TRACKNET_SCRIPT),
                "--video_file", str(video_path),
                "--tracknet_file", str(TRACKNET_MODEL),
                "--inpaintnet_file", str(INPAINT_MODEL),
                "--save_dir", str(tracknet_out)
            ], timeout=900)

            # Check both possible output locations
            tracknet_csv = tracknet_out / f"{stem}_ball.csv"
            if not tracknet_csv.exists():
                tracknet_csv = expected_csv

            if not tracknet_csv.exists():
                raise RuntimeError(f"TrackNet failed: {err}")
        else:
            tracknet_csv = expected_csv

        # Step 3: Find hit event
        write_job(job_id, {**read_job(job_id), "step": "Đang tìm điểm chạm vợt...", "progress": 60})
        events_out = JOBS_DIR / f"{job_id}_events.json"
        code, out, err = _run_cmd([
            sys.executable, str(EVENTS_SCRIPT),
            str(video_path),
            "--trajectory", str(tracknet_csv),
            "--output", str(events_out)
        ])

        if not events_out.exists():
            raise RuntimeError(f"Event detection failed: {err}")

        with open(events_out) as f:
            events_data = json.load(f)

        events = events_data.get("events", [])
        if not events:
            raise RuntimeError("Không tìm thấy điểm chạm vợt trong video. Video có thể quá ngắn hoặc không phát hiện được quỹ đạo cầu.")

        # Use first event
        event = events[0]
        event_frame = event["frame"]
        player_side = event["player_side"]

        # Step 4: Fusion model
        write_job(job_id, {**read_job(job_id), "step": "Đang phân loại cú đánh (AI Fusion)...", "progress": 80})
        fusion_out = JOBS_DIR / f"{job_id}_fusion.json"

        corner_args = [str(v) for v in flat_corners]
        code, out, err = _run_cmd([
            sys.executable, str(FUSION_SCRIPT),
            "--rgb-checkpoint", str(RGB_CHECKPOINT),
            "--trajectory", str(tracknet_csv),
            "--court-corners", *corner_args,
            "--event-frame", str(event_frame),
            "--player-side", player_side,
            "--output", str(fusion_out),
            str(video_path)
        ], timeout=300)

        if not fusion_out.exists():
            raise RuntimeError(f"Fusion failed: {err}")

        with open(fusion_out) as f:
            fusion_result = json.load(f)

        # Done!
        final = {
            **read_job(job_id),
            "status": "done",
            "step": "Hoàn thành!",
            "progress": 100,
            "result": {
                "stroke": fusion_result["stroke"],
                "stroke_side": fusion_result["stroke_side"],
                "stroke_ranking": fusion_result["stroke_ranking"],
                "stroke_side_ranking": fusion_result["stroke_side_ranking"],
                "event_frame": event_frame,
                "player_side": player_side,
                "court_auto_detected": court_ok,
            }
        }
        if (JOBS_DIR / f"{job_id}_court.jpg").exists():
            final["court_image"] = f"/jobs/{job_id}_court.jpg"
        write_job(job_id, final)

    except Exception as e:
        write_job(job_id, {**read_job(job_id), "status": "error", "step": f"Lỗi: {str(e)}", "progress": 0})


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/classify")
async def classify(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "Chỉ hỗ trợ file video MP4, AVI, MOV, MKV")

    job_id = str(uuid.uuid4())[:8]
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    write_job(job_id, {
        "job_id": job_id,
        "filename": file.filename,
        "status": "queued",
        "step": "Đang chuẩn bị...",
        "progress": 0,
        "created_at": time.time()
    })

    background_tasks.add_task(process_video, job_id, video_path)
    return {"job_id": job_id}


@app.post("/classify_path")
async def classify_path(background_tasks: BackgroundTasks, data: dict):
    """Accept a local file path directly - no file copy needed."""
    path_str = data.get("path", "").strip().strip('"').strip("'").strip()
    if not path_str:
        raise HTTPException(400, "Thiếu đường dẫn file")

    video_path = Path(path_str)
    if not video_path.exists():
        raise HTTPException(400, f"Không tìm thấy file: {path_str}")
    if not video_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
        raise HTTPException(400, "Chỉ hỗ trợ file video MP4, AVI, MOV, MKV")

    job_id = str(uuid.uuid4())[:8]
    write_job(job_id, {
        "job_id": job_id,
        "filename": video_path.name,
        "status": "queued",
        "step": "Đang chuẩn bị...",
        "progress": 0,
        "created_at": time.time()
    })

    background_tasks.add_task(process_video, job_id, video_path)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = read_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/jobs/{filename}")
def get_job_file(filename: str):
    path = JOBS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(path))


if __name__ == "__main__":
    print("Starting Badminton Shot Classifier server...")
    print("Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000)

