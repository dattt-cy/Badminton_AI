import torch
import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from ai_classifier.localization.fast_tracknet import FastTrackNet
from scripts.inference.classify_shuttleset_fusion_video import classify_fusion_event
from scripts.inference.analyze_long_video import default_corners
from scripts.inference.auto_court_detection import detect_court_corners
from scripts.training.train_shuttleset_feature_fusion import FusionHead
from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D

def test_clip(video_path_str: str, player_side: str = "upper"):
    video_path = Path(video_path_str)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    event_frame = total_frames // 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        corners = detect_court_corners(video_path)
    except Exception:
        corners = default_corners(w, h)

    pose_model = YOLO('models/checkpoints/pose/yolov8n-pose.pt')
    rgb_ckpt = torch.load('work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth', map_location='cpu', weights_only=False)
    rgb_model = MultiTaskR2Plus1D()
    rgb_model.load_state_dict(rgb_ckpt['model'])
    rgb_model.to(device).eval()

    fusion_ckpt = torch.load('work_dirs/shuttleset_fusion_epoch20_full_b4/best.pth', map_location='cpu', weights_only=False)
    structured_dim = fusion_ckpt['model']['structured.1.weight'].shape[1]
    fusion_model = FusionHead(512, structured_dim)
    fusion_model.load_state_dict(fusion_ckpt['model'])
    fusion_model.to(device).eval()

    tracker = FastTrackNet(device=device)
    trajectory = tracker.predict_window(video_path, event_frame, window_before=30, window_after=30)

    res = classify_fusion_event(
        video_path, trajectory, event_frame, player_side, corners,
        pose_model, rgb_model, fusion_model, rgb_ckpt, fusion_ckpt, device
    )

    print(f"FUSION PREDICTION FOR {video_path.name}:")
    print(f"  Stroke: {res['stroke']['label']} ({res['stroke']['probability']*100:.1f}%)")
    print(f"  Side:   {res['stroke_side']['label']} ({res['stroke_side']['probability']*100:.1f}%)")
    top3 = [f"{x['label']}: {x['probability']*100:.1f}%" for x in res['stroke_ranking'][:3]]
    print(f"  Top 3:  {', '.join(top3)}")

if __name__ == "__main__":
    test_clip('C:/Users/ADMIN/Downloads/archive/forehand_clear/001.mp4', 'upper')
