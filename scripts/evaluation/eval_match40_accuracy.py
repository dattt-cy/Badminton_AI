import json, pandas as pd, numpy as np

# Load GT
df = pd.read_csv('data/manifests/shuttleset_rgb.csv')
m40 = df[df['video_path'].str.contains('40 ', na=False)].copy()
gt = m40[(m40['hit_frame'] >= 18000) & (m40['hit_frame'] <= 27000)].sort_values('hit_frame').copy()
gt['local_frame'] = gt['hit_frame'] - 18000

# Load Pipeline output
pred_json = json.load(open('work_dirs/test_match40_10m_15m_full_pipeline_epoch20/timeline.json'))
events = pred_json.get('events', [])

print(f"Ground Truth events in 5m window: {len(gt)}")
print(f"Pipeline detected events: {len(events)}")

matched = []
gt_matched_indices = set()

for ev in events:
    f_pred = ev['frame']
    diffs = (gt['local_frame'] - f_pred).abs()
    best_idx = diffs.idxmin()
    best_diff = diffs.loc[best_idx]
    
    # Matching tolerance: +/- 15 frames (0.5 second)
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

stroke_acc = c_stroke / n_matched * 100
side_acc = c_side / n_matched * 100
joint_acc = c_joint / n_matched * 100

print("\n" + "="*60)
print("KẾT QUẢ ĐỐI CHIẾU TRỰC TIẾP MATCH 40 (5 PHÚT):")
print("="*60)
print(f"  Số cú đánh bắt trúng vị trí GT : {n_matched} / {len(gt)} ({n_matched/len(gt)*100:.1f}%)")
print(f"  Stroke Accuracy (Loại cú đánh) : {c_stroke} / {n_matched} ({stroke_acc:.1f}%)")
print(f"  Side Accuracy (Hướng tay)      : {c_side} / {n_matched} ({side_acc:.1f}%)")
print(f"  JOINT ACCURACY (Đúng cả hai)   : {c_joint} / {n_matched} ({joint_acc:.1f}%)")
print("="*60)

# Check per-class stroke accuracy
print("\n--- CHI TIẾT THEO LOẠI CÚ ĐÁNH TRÊN MATCH 40 ---")
for cls, grp in matched_df.groupby('exp_stroke'):
    acc = grp['ok_stroke'].mean() * 100
    print(f"  {cls:<12}: {grp['ok_stroke'].sum():2d}/{len(grp):2d} ({acc:5.1f}%)")

