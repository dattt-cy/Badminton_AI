import json
import pandas as pd

df = pd.read_csv('data/manifests/shuttleset_rgb.csv')
m39 = df[df['video_path'].str.contains('39 ', na=False)].copy()
gt = m39[(m39['hit_frame'] >= 18000) & (m39['hit_frame'] <= 27000)].sort_values('hit_frame').copy()
gt['local_frame'] = gt['hit_frame'] - 18000

pred_json = json.load(open('work_dirs/test_match39_10m_15m_fast_pipeline/timeline.json', encoding='utf-8'))
events = pred_json.get('events', [])

matched = []
for ev in events:
    f_pred = ev['frame']
    diffs = (gt['local_frame'] - f_pred).abs()
    best_idx = diffs.idxmin()
    best_diff = diffs.loc[best_idx]
    if best_diff <= 15:
        gt_row = gt.loc[best_idx]
        ok_s = (ev['stroke']['label'] == gt_row['coarse_label'])
        ok_side = (ev['stroke_side']['label'] == gt_row['stroke_side'])
        matched.append({
            'pred_stroke': ev['stroke']['label'],
            'gt_stroke': gt_row['coarse_label'],
            'pred_side': ev['stroke_side']['label'],
            'gt_side': gt_row['stroke_side'],
            'ok_stroke': ok_s,
            'ok_side': ok_side,
            'ok_joint': ok_s and ok_side
        })

m_df = pd.DataFrame(matched)

n_matched = len(m_df)
c_stroke = int(m_df['ok_stroke'].sum())
c_side = int(m_df['ok_side'].sum())
c_joint = int(m_df['ok_joint'].sum())

stroke_acc = c_stroke / n_matched * 100 if n_matched else 0
side_acc = c_side / n_matched * 100 if n_matched else 0
joint_acc = c_joint / n_matched * 100 if n_matched else 0

print("=" * 60)
print("SO SANH: FAST MODE (SINGLE OFFSET) vs FULL ENSEMBLE TRÊN MATCH 39:")
print("=" * 60)
print(f"Thoi gian chay Fast Mode:       8 phut 53 giay (so voi 18 phut 40 giay)")
print(f"Toc do phan loai moi cu danh:   ~8.0 giay (so voi 14.5 giay)")
print("-" * 60)
print(f"Stroke Accuracy (Fast Mode):    {c_stroke} / {n_matched} ({stroke_acc:.1f}%) [Full: 82.0%]")
print(f"Side Accuracy (Fast Mode):      {c_side} / {n_matched} ({side_acc:.1f}%) [Full: 82.0%]")
print(f"Joint Accuracy (Fast Mode):     {c_joint} / {n_matched} ({joint_acc:.1f}%) [Full: 67.2%]")
print("=" * 60)
print("Chi tiet tung loai cu danh o Fast Mode:")
for stroke, grp in m_df.groupby('gt_stroke'):
    acc = grp['ok_stroke'].mean() * 100
    n_c = grp['ok_stroke'].sum()
    print(f"  {stroke:12s}: {n_c:2d} / {len(grp):2d} ({acc:5.1f}%)")

