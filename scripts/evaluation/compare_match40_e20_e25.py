import json, pandas as pd

# Load GT
df = pd.read_csv('data/manifests/shuttleset_rgb.csv')
m40 = df[df['video_path'].str.contains('40 ', na=False)].copy()
gt = m40[(m40['hit_frame'] >= 18000) & (m40['hit_frame'] <= 27000)].sort_values('hit_frame').copy()
gt['local_frame'] = gt['hit_frame'] - 18000

def evaluate_run(timeline_path, label):
    pred_json = json.load(open(timeline_path))
    events = pred_json.get('events', [])
    matched = []
    for ev in events:
        f_pred = ev['frame']
        diffs = (gt['local_frame'] - f_pred).abs()
        best_idx = diffs.idxmin()
        best_diff = diffs.loc[best_idx]
        if best_diff <= 15:
            gt_row = gt.loc[best_idx]
            p_stroke = ev['stroke']['label']
            p_side = ev['stroke_side']['label']
            e_stroke = gt_row['coarse_label']
            e_side = gt_row['stroke_side']
            matched.append({
                'stroke_ok': p_stroke == e_stroke,
                'side_ok': p_side == e_side,
                'joint_ok': (p_stroke == e_stroke) and (p_side == e_side)
            })
    m_df = pd.DataFrame(matched)
    n = len(m_df)
    s_cnt = int(m_df['stroke_ok'].sum())
    side_cnt = int(m_df['side_ok'].sum())
    j_cnt = int(m_df['joint_ok'].sum())
    s_acc = s_cnt / n * 100
    side_acc = side_cnt / n * 100
    j_acc = j_cnt / n * 100
    print(f"{label:<22} | Stroke: {s_cnt:2d}/{n} ({s_acc:5.1f}%) | Side: {side_cnt:2d}/{n} ({side_acc:5.1f}%) | JOINT: {j_cnt:2d}/{n} ({j_acc:5.1f}%)")

print("=" * 80)
print("SO SÁNH ĐỐI ĐẦU TRỰC TIẾP MATCH 40 (5 PHÚT) TRÊN GROUND TRUTH BWF:")
print("=" * 80)
evaluate_run('work_dirs/test_match40_10m_15m_full_pipeline_epoch20/timeline.json', 'Epoch 20 (Chuẩn)')
evaluate_run('work_dirs/test_match40_10m_15m_full_pipeline_epoch25/timeline.json', 'Epoch 25 (Mới)')
print("=" * 80)

