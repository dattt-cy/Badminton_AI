import glob
import json
import subprocess

files = sorted(glob.glob('C:/Users/ADMIN/Downloads/archive/forehand_clear/*.mp4'))[:6]
for f in files:
    cmd = [
        'python', 'scripts/inference/classify_shuttleset_rgb_multitask.py',
        f, '--checkpoint', 'work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth',
        '--crop-hitter', '--player-side', 'top'
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    try:
        data = json.loads(res.stdout)
        pred = data['prediction']
        fname = f.replace('\\', '/').split('/')[-1]
        b = data['video_metadata']['windows'][0]['crop_box']
        yc = (b[1] + b[3]) / 2 / 1080
        top3 = [f"{x['label']}:{x['probability']*100:.1f}%" for x in data['stroke_ranking'][:3]]
        print(f"{fname:10s} yc={yc:.3f} -> {pred['combined_label']} | Top: {', '.join(top3)}")
    except Exception as e:
        print(f, 'error:', e)

