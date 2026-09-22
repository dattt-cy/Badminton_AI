import glob
import json
import subprocess

for pattern in ['C:/Users/ADMIN/Downloads/archive/backhand_drive/*.mp4', 'C:/Users/ADMIN/Downloads/archive/forehand_drive/*.mp4']:
    files = sorted(glob.glob(pattern))[:3]
    for f in files:
        cmd = ['python', 'scripts/inference/classify_shuttleset_rgb_multitask.py', f, '--checkpoint', 'work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth', '--crop-hitter', '--player-side', 'top']
        res = subprocess.run(cmd, capture_output=True, text=True)
        try:
            data = json.loads(res.stdout)
            pred = data['prediction']
            fname = f.replace('\\', '/').split('/')[-2:]
            top3 = [f"{x['label']}: {x['probability']*100:.1f}%" for x in data['stroke_ranking'][:3]]
            print(f"{'/'.join(fname)} -> Stroke: {pred['stroke']['label']} ({pred['stroke']['probability']*100:.1f}%), Side: {pred['stroke_side']['label']}")
            print(f"   Top 3: {', '.join(top3)}")
        except Exception as e:
            print(f, 'error:', e)

