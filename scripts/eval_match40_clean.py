import json
import pandas as pd

df = pd.read_csv('data/manifests/shuttleset_rgb.csv')
m40 = df[df['video_path'].str.contains('40 ', na=False)].copy()
gt = m40[(m40['hit_frame'] >= 18000) & (m40['hit_frame'] <= 27000)].sort_values('hit_frame').copy()
gt['local_frame'] = gt['hit_frame'] - 18000

pred_json = json.load(open('work_dirs/test_match40_10m_15m_full_pipeline_epoch20/timeline.json'))
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
print("=" * 55)
print("MATCH 40 FULL PIPELINE (EPOCH 20 FUSION) EVALUATION:")
print("=" * 55)
print(f"Total Pipeline Detections: {len(events)}")
print(f"Matched Ground Truth:      {len(m_df)} / {len(gt)}")
print(f"Stroke Accuracy:           {m_df['ok_stroke'].sum()}/{len(m_df)} ({m_df['ok_stroke'].mean()*100:.1f}%)")
print(f"Side Accuracy:             {m_df['ok_side'].sum()}/{len(m_df)} ({m_df['ok_side'].mean()*100:.1f}%)")
print(f"Joint Accuracy:            {m_df['ok_joint'].sum()}/{len(m_df)} ({m_df['ok_joint'].mean()*100:.1f}%)")
print("-" * 55)
print("Per-class Accuracy on Match 40:")
for stroke, grp in m_df.groupby('gt_stroke'):
    acc = grp['ok_stroke'].mean() * 100
    n_correct = grp['ok_stroke'].sum()
    print(f"  {stroke:12s}: {n_correct:2d}/{len(grp):2d} ({acc:5.1f}%)")
print("=" * 55)
print("\nConfusion Matrix for Misclassified Events:")
wrong = m_df[~m_df['ok_stroke']]
for _, row in wrong.iterrows():
    print(f"  GT: {row['gt_stroke']:12s} -> AI predicted: {row['pred_stroke']:12s}")

