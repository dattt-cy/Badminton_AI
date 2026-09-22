import glob
import json
import subprocess
import time
import cv2

files = sorted(glob.glob('C:/Users/ADMIN/Downloads/archive/backhand_drive/*.mp4'))[:6]
for f in files:
    cap = cv2.VideoCapture(f)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    t0 = time.time()
    cmd = [
        'python', 'scripts/inference/classify_shuttleset_rgb_multitask.py',
        f, '--checkpoint', 'work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth',
        '--crop-hitter', '--player-side', 'top'
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    try:
        data = json.loads(res.stdout)
        pred = data['prediction']
        yc = 'N/A'
        if 'video_metadata' in data and data['video_metadata']['windows']:
            b = data['video_metadata']['windows'][0]['crop_box']
            yc = f'{(b[1]+b[3])/2/1080:.3f}'
        top3 = [f"{x['label']}:{x['probability']*100:.1f}%" for x in data['stroke_ranking'][:3]]
        fname = f.replace('\\', '/').split('/')[-1]
        print(f"{fname:12s} frames={n_frames} ({elapsed:.2f}s) yc={yc} -> {pred['combined_label']} | Top: {', '.join(top3)}")
    except Exception as e:
        fname = f.replace('\\', '/').split('/')[-1]
        print(f"{fname} error in {elapsed:.2f}s: {e}")

