import json, sys, time, cv2, torch, numpy as np
from pathlib import Path
from torchvision.transforms import functional as vision_functional
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D, select_hitter_box, padded_square, STROKE_CLASSES, SIDE_CLASSES

def eval_ten_clips(ckpt_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    model = MultiTaskR2Plus1D()
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    
    pose_model = YOLO("models/checkpoints/pose/yolov8n-pose.pt")
    
    # Load valid clips from archive_30
    report30 = json.load(open("work_dirs/archive_30clips_full_selector_fasttrack_fusion_physics_exact_eval/report.json"))
    clips = [c for c in report30["results"] if "event_frame" in c][:10]
    
    print(f"\nEvaluating {len(clips)} archive clips on {ckpt_path.name}...")
    print("-" * 75)
    print(f"{'#':<3} | {'Clip':<22} | {'Expected':<18} | {'Predicted':<18} | {'Status'}")
    print("-" * 75)
    
    correct_stroke = 0
    correct_side = 0
    correct_joint = 0
    
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
    
    for idx, c in enumerate(clips, 1):
        vpath = Path(c["video"])
        exp_stroke = c["expected_stroke"]
        exp_side = c["expected_side"]
        hit_f = c["event_frame"]
        
        # Read clip
        cap = cv2.VideoCapture(str(vpath))
        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Detect hitter crop around hit frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, hit_f))
        ok, f = cap.read()
        crop_box = None
        if ok:
            res = pose_model.predict(f, imgsz=960, conf=0.20, verbose=False)[0]
            box = select_hitter_box(res, "bottom", w, h)
            if box is not None:
                crop_box = padded_square(box, w, h, 0.55)
        
        # Subsample 16 frames in [hit_f - 30, hit_f + 30]
        start_f = max(0, hit_f - 30)
        end_f = min(tot_frames - 1, hit_f + 30)
        wanted = np.rint(np.linspace(start_f, end_f, 16)).astype(int)
        
        frames = []
        for fi in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if crop_box:
                    x1, y1, x2, y2 = crop_box
                    frame = frame[y1:y2, x1:x2]
            frame_t = torch.from_numpy(frame).permute(2, 0, 1)
            frame_t = vision_functional.resize(frame_t, [112, 112], antialias=False)
            frames.append(frame_t)
        cap.release()
        
        tensor = torch.stack(frames).float().div_(255.0).permute(1, 0, 2, 3).unsqueeze(0).to(device) # (1, 3, 16, 112, 112)
        tensor = (tensor - mean) / std
        
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                s_logits, side_logits = model(tensor)
                p_stroke = STROKE_CLASSES[s_logits.argmax(1).item()]
                p_side = SIDE_CLASSES[side_logits.argmax(1).item()]
        
        ok_stroke = (p_stroke == exp_stroke)
        ok_side = (p_side == exp_side)
        ok_joint = (ok_stroke and ok_side)
        
        if ok_stroke: correct_stroke += 1
        if ok_side: correct_side += 1
        if ok_joint: correct_joint += 1
        
        clip_name = f"{vpath.parent.name}/{vpath.name}"
        status = "✅ JOINT" if ok_joint else ("⚡ STROKE" if ok_stroke else ("🔹 SIDE" if ok_side else "❌ WRONG"))
        print(f"{idx:<3} | {clip_name:<20} | {exp_side+'_'+exp_stroke:<18} | {p_side+'_'+p_stroke:<18} | {status}")
    
    print("-" * 75)
    print(f"Summary on {len(clips)} clips:")
    print(f"  Stroke Accuracy : {correct_stroke}/{len(clips)} ({correct_stroke/len(clips)*100:.1f}%)")
    print(f"  Side Accuracy   : {correct_side}/{len(clips)} ({correct_side/len(clips)*100:.1f}%)")
    print(f"  Joint Accuracy  : {correct_joint}/{len(clips)} ({correct_joint/len(clips)*100:.1f}%)")

if __name__ == "__main__":
    ckpt = Path("work_dirs/r2plus1d18_mixed_e21_e25_test/latest.pth")
    eval_ten_clips(ckpt)
