import json
import pandas as pd

# Load GT
df = pd.read_csv('data/manifests/shuttleset_rgb.csv')
m39 = df[df['video_path'].str.contains('39 ', na=False)].copy()
gt = m39[(m39['hit_frame'] >= 18000) & (m39['hit_frame'] <= 27000)].sort_values('hit_frame').copy()
gt['local_frame'] = gt['hit_frame'] - 18000

pred_json = json.load(open('work_dirs/test_match39_10m_15m_full_pipeline_epoch20/timeline.json', encoding='utf-8'))
events = pred_json.get('events', [])

matched = []
gt_matched_indices = set()

for ev in events:
    f_pred = ev['frame']
    diffs = (gt['local_frame'] - f_pred).abs()
    best_idx = diffs.idxmin()
    best_diff = diffs.loc[best_idx]
    
    # Matching tolerance: +/- 15 frames (0.5s)
    if best_diff <= 15:
        gt_row = gt.loc[best_idx]
        gt_matched_indices.add(best_idx)
        
        pred_stroke = ev['stroke']['label']
        pred_side = ev['stroke_side']['label']
        exp_stroke = gt_row['coarse_label']
        exp_side = gt_row['stroke_side']
        
        ok_stroke = (pred_stroke == exp_stroke)
        ok_side = (pred_side == exp_side)
        ok_joint = (ok_stroke and ok_side)
        
        matched.append({
            'frame_pred': f_pred,
            'frame_gt': int(gt_row['local_frame']),
            'diff': int(best_diff),
            'pred_stroke': pred_stroke,
            'exp_stroke': exp_stroke,
            'pred_side': pred_side,
            'exp_side': exp_side,
            'ok_stroke': ok_stroke,
            'ok_side': ok_side,
            'ok_joint': ok_joint
        })

matched_df = pd.DataFrame(matched)

n_matched = len(matched_df)
c_stroke = int(matched_df['ok_stroke'].sum())
c_side = int(matched_df['ok_side'].sum())
c_joint = int(matched_df['ok_joint'].sum())

stroke_acc = c_stroke / n_matched * 100 if n_matched else 0
side_acc = c_side / n_matched * 100 if n_matched else 0
joint_acc = c_joint / n_matched * 100 if n_matched else 0

print("=" * 60)
print("KET QUA DOI CHIEU THUC TE TRUC TIEP MATCH 39 (5 PHUT):")
print("=" * 60)
print(f"Tong so cu danh Ground Truth (10m-15m)  : {len(gt)}")
print(f"So cu danh AI phat hien                 : {len(events)}")
print(f"So cu danh bat trung vi tri GT (<=15f)  : {n_matched} / {len(gt)} ({n_matched/len(gt)*100:.1f}%)")
print(f"Stroke Accuracy (Loai cu danh)          : {c_stroke} / {n_matched} ({stroke_acc:.1f}%)")
print(f"Side Accuracy (Huong tay)               : {c_side} / {n_matched} ({side_acc:.1f}%)")
print(f"JOINT ACCURACY (Dung ca hai)            : {c_joint} / {n_matched} ({joint_acc:.1f}%)")
print("-" * 60)
print("CHI TIET THEO TUNG LOAI CU DANH TREN MATCH 39:")
for stroke, grp in matched_df.groupby('exp_stroke'):
    acc = grp['ok_stroke'].mean() * 100
    n_c = grp['ok_stroke'].sum()
    print(f"  {stroke:12s}: {n_c:2d} / {len(grp):2d} ({acc:5.1f}%)")
print("=" * 60)
print("\nBang cac cu danh bi lech (Misclassified):")
wrong = matched_df[~matched_df['ok_stroke']]
for _, row in wrong.iterrows():
    print(f"  Frame {row['frame_pred']:4d} (GT {row['frame_gt']:4d}): GT={row['exp_stroke']:12s} -> AI={row['pred_stroke']:12s} ({row['pred_side']:8s} vs {row['exp_side']:8s})")

