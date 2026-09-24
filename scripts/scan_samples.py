import subprocess
from pathlib import Path

names = ['backhand_net_shot', 'forehand_lift', 'forehand_net_shot', 'backhand_drive']
for name in names:
    p = Path(f'C:/Users/ADMIN/Downloads/archive/{name}/001.mp4')
    out = Path(f'work_dirs/scan_{name}_001.json')
    cmd = [
        'python', 'scripts/inference/scan_video_hit_rgb.py',
        '--video', str(p),
        '--checkpoint', 'work_dirs/r2plus1d18_hit_full/best.pth',
        '--output', str(out),
    ]
    print(f'Scanning {name}...')
    subprocess.run(cmd, check=True)
print('Done scanning!')

