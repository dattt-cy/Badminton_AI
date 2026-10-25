import cv2, json, csv
from pathlib import Path

video_path = Path("AI_Classifier/work_dirs/match44_20min_clip.mp4")
cap = cv2.VideoCapture(str(video_path))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = frames / fps if fps else 0
cap.release()

print(f"Video: {video_path.name} | Total Frames: {frames} | FPS: {fps} | Duration: {duration:.1f}s ({duration/60:.2f} mins)")

# Check existing 20min json results if any
old_20min_file = Path("AI_Classifier/work_dirs/match44_20min_pipeline_with_tactical_decoder.json")
if old_20min_file.exists():
    with old_20min_file.open("r", encoding="utf-8") as f:
        d = json.load(f)
    print("Found existing 20min baseline json:")
    print("  Video info:", d.get("video", {}))
    print("  Hit count:", d.get("summary", {}).get("hit_count"))
    print("  Rally count:", d.get("summary", {}).get("estimated_rally_count"))
    print("  Estimated score:", d.get("summary", {}).get("estimated_score"))

