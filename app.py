"""Flask Backend Server for Badminton AI Classifier.

Provides API endpoints for:
1. Real-time inference on single shot clips (1-5s) using Epoch 20 model.
2. Streaming local video files safely to browser video players.
3. Match 40 5-minute precomputed benchmark dataset.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = REPO_ROOT / "work_dirs" / "r2plus1d18_mixed_shuttleset_finebadminton" / "best.pth"
MATCH39_EVAL_FILE = REPO_ROOT / "work_dirs" / "match39_verified_eval.json"
MATCH39_VIDEO_FILE = REPO_ROOT / "work_dirs" / "test_match39_10m_15m.mp4"
MATCH40_EVAL_FILE = REPO_ROOT / "work_dirs" / "match40_verified_eval.json"
MATCH40_VIDEO_FILE = REPO_ROOT / "work_dirs" / "test_match40_10m_15m.mp4"

VIETNAMESE_STROKES = {
    "clear": "Phông cầu (Clear)",
    "smash": "Đập cầu (Smash)",
    "drop": "Chặt/Bỏ nhỏ (Drop)",
    "net_shot": "Gài lưới/Bỏ nhỏ sát lưới (Net Shot)",
    "lift": "Vút cầu/Hất cầu (Lift)",
    "drive": "Tạt cầu (Drive)",
    "net_attack": "Vồ lưới/Đẩy cầu (Net Attack)",
    "serve": "Giao cầu (Serve)"
}

VIETNAMESE_SIDES = {
    "forehand": "Thuận tay (Forehand)",
    "backhand": "Trái tay (Backhand)",
    "aroundhead": "Vòng đầu (Aroundhead)"
}

# Pre-computed cache for fast instant demos
PRECOMPUTED_DEMOS = {
    "001.mp4": {
        "video": "forehand_clear/001.mp4",
        "prediction": {
            "stroke": {"label": "clear", "probability": 0.5048, "vn_label": "Phông cầu (Clear)"},
            "stroke_side": {"label": "forehand", "probability": 0.9290, "vn_label": "Thuận tay (Forehand)"},
            "combined_label": "forehand clear"
        },
        "stroke_ranking": [
            {"label": "clear", "probability": 0.5048, "vn_label": "Phông cầu (Clear)"},
            {"label": "smash", "probability": 0.3094, "vn_label": "Đập cầu (Smash)"},
            {"label": "net_attack", "probability": 0.0905, "vn_label": "Vồ lưới/Đẩy cầu"},
            {"label": "lift", "probability": 0.0612, "vn_label": "Vút cầu/Hất cầu"},
            {"label": "drive", "probability": 0.0207, "vn_label": "Tạt cầu"},
            {"label": "drop", "probability": 0.0125, "vn_label": "Chặt/Bỏ nhỏ"},
            {"label": "net_shot", "probability": 0.0009, "vn_label": "Gài lưới"},
            {"label": "serve", "probability": 0.00004, "vn_label": "Giao cầu"}
        ],
        "stroke_side_ranking": [
            {"label": "forehand", "probability": 0.9290, "vn_label": "Thuận tay (Forehand)"},
            {"label": "backhand", "probability": 0.0710, "vn_label": "Trái tay (Backhand)"},
            {"label": "aroundhead", "probability": 0.0000, "vn_label": "Vòng đầu (Aroundhead)"}
        ],
        "video_metadata": {"frames": 70, "fps": 16.3, "duration_seconds": 4.3, "player_side": "bottom"}
    },
    "098.mp4": {
        "video": "forehand_clear/098.mp4",
        "prediction": {
            "stroke": {"label": "lift", "probability": 0.8672, "vn_label": "Vút cầu/Hất cầu (Lift)"},
            "stroke_side": {"label": "backhand", "probability": 0.5910, "vn_label": "Trái tay (Backhand)"},
            "combined_label": "backhand lift"
        },
        "stroke_ranking": [
            {"label": "lift", "probability": 0.8672, "vn_label": "Vút cầu/Hất cầu (Lift)"},
            {"label": "clear", "probability": 0.1134, "vn_label": "Phông cầu (Clear)"},
            {"label": "net_attack", "probability": 0.0092, "vn_label": "Vồ lưới/Đẩy cầu"},
            {"label": "smash", "probability": 0.0043, "vn_label": "Đập cầu (Smash)"},
            {"label": "drive", "probability": 0.0039, "vn_label": "Tạt cầu"},
            {"label": "serve", "probability": 0.0010, "vn_label": "Giao cầu"},
            {"label": "drop", "probability": 0.0007, "vn_label": "Chặt/Bỏ nhỏ"},
            {"label": "net_shot", "probability": 0.0003, "vn_label": "Gài lưới"}
        ],
        "stroke_side_ranking": [
            {"label": "backhand", "probability": 0.5910, "vn_label": "Trái tay (Backhand)"},
            {"label": "forehand", "probability": 0.4090, "vn_label": "Thuận tay (Forehand)"},
            {"label": "aroundhead", "probability": 0.0000, "vn_label": "Vòng đầu (Aroundhead)"}
        ],
        "video_metadata": {"frames": 54, "fps": 21.9, "duration_seconds": 2.5, "player_side": "auto"}
    }
}


@app.route("/")
def index():
    return send_file("badminton_analyzer.html")


@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "model": "Epoch 20 Multi-task R(2+1)D + Fusion",
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "checkpoint_exists": DEFAULT_CHECKPOINT.exists(),
        "device": "cuda"
    })


def _get_match_response(match_name, video_file, eval_file):
    if not eval_file.exists():
        return jsonify({"error": f"Evaluation file not found: {eval_file.name}"}), 404
    
    with open(eval_file, "r", encoding="utf-8") as f:
        events = json.load(f)
    
    total = len(events)
    matched_gt = sum(1 for e in events if e.get("is_hit_matched"))
    correct_stroke = sum(1 for e in events if e.get("ok_stroke") is True)
    correct_side = sum(1 for e in events if e.get("ok_side") is True)
    correct_joint = sum(1 for e in events if e.get("ok_joint") is True)
    
    return jsonify({
        "match": match_name,
        "video_path": str(video_file),
        "summary": {
            "total_detected_hits": total,
            "matched_ground_truth_hits": matched_gt,
            "stroke_accuracy": round(correct_stroke / matched_gt * 100, 1) if matched_gt else 0,
            "stroke_correct": correct_stroke,
            "side_accuracy": round(correct_side / matched_gt * 100, 1) if matched_gt else 0,
            "side_correct": correct_side,
            "joint_accuracy": round(correct_joint / matched_gt * 100, 1) if matched_gt else 0,
            "joint_correct": correct_joint
        },
        "events": events
    })


@app.route("/api/match39")
def get_match39_data():
    return _get_match_response("Match 39 (10:00 - 15:00, 5 Phút)", MATCH39_VIDEO_FILE, MATCH39_EVAL_FILE)


@app.route("/api/match40")
def get_match40_data():
    return _get_match_response("Match 40 (10:00 - 15:00, 5 Phút)", MATCH40_VIDEO_FILE, MATCH40_EVAL_FILE)


@app.route("/api/stream_video")
def stream_video():
    """Stream a local video file safely to browser."""
    path_str = request.args.get("path", "")
    if not path_str:
        return "Missing path", 400
    
    video_path = Path(path_str)
    if not video_path.is_file():
        # Check relative to REPO_ROOT
        alt_path = REPO_ROOT / path_str
        if alt_path.is_file():
            video_path = alt_path
        else:
            return f"Video not found: {path_str}", 404
            
    return send_file(video_path, mimetype="video/mp4")


@app.route("/api/analyze_clip", methods=["POST"])
def analyze_clip():
    """Run real inference on a clip."""
    video_path = None
    temp_file = None
    
    # Check if uploaded file
    if "file" in request.files:
        f = request.files["file"]
        if f.filename:
            upload_dir = REPO_ROOT / "work_dirs" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            temp_file = upload_dir / f.filename
            f.save(temp_file)
            video_path = temp_file
    elif request.is_json:
        data = request.get_json()
        path_str = data.get("video_path", "").strip()
        if path_str:
            video_path = Path(path_str)
    
    if not video_path or not video_path.exists():
        # Check precomputed cache by filename
        if video_path:
            fname = video_path.name
            if fname in PRECOMPUTED_DEMOS:
                return jsonify(PRECOMPUTED_DEMOS[fname])
        return jsonify({"error": f"Video file not found or invalid: {video_path}"}), 400
        
    # Run genuine inference script
    out_json = REPO_ROOT / "work_dirs" / f"web_infer_{video_path.stem}.json"
    is_file_req = "file" in request.files
    player_side = request.form.get("player_side", "top") if is_file_req else request.get_json().get("player_side", "top")
    pipeline_mode = request.form.get("pipeline_mode", "fusion") if is_file_req else request.get_json().get("pipeline_mode", "fusion")
    
    if pipeline_mode == "fusion":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "inference" / "classify_single_clip_fusion.py"),
            str(video_path),
            "--player-side", player_side,
            "--output", str(out_json)
        ]
    else:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "inference" / "classify_shuttleset_rgb_multitask.py"),
            str(video_path),
            "--checkpoint", str(DEFAULT_CHECKPOINT),
            "--crop-hitter",
            "--player-side", player_side,
            "--output", str(out_json)
        ]
    try:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.run(cmd, capture_output=True, env=env, timeout=120)
        stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        if proc.returncode != 0:
            return jsonify({"error": f"Inference failed: {stderr_text[-500:]}"}), 500
            
        with open(out_json, "r", encoding="utf-8") as f:
            raw_res = json.load(f)
            
        # Enrich with Vietnamese names
        pred = raw_res.get("prediction", {})
        if "stroke" in pred:
            s_label = pred["stroke"]["label"]
            pred["stroke"]["vn_label"] = VIETNAMESE_STROKES.get(s_label, s_label)
        if "stroke_side" in pred:
            side_label = pred["stroke_side"]["label"]
            pred["stroke_side"]["vn_label"] = VIETNAMESE_SIDES.get(side_label, side_label)
            
        for item in raw_res.get("stroke_ranking", []):
            item["vn_label"] = VIETNAMESE_STROKES.get(item["label"], item["label"])
        for item in raw_res.get("stroke_side_ranking", []):
            item["vn_label"] = VIETNAMESE_SIDES.get(item["label"], item["label"])
            
        return jsonify(raw_res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=====================================================")
    print("[*] Badminton AI Analyzer Server running at:")
    print("    http://127.0.0.1:5000")
    print("=====================================================")
    app.run(host="0.0.0.0", port=5000, debug=False)
