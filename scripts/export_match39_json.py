import json
import pandas as pd

df = pd.read_csv('data/manifests/shuttleset_rgb.csv')
m39 = df[df['video_path'].str.contains('39 ', na=False)].copy()
gt = m39[(m39['hit_frame'] >= 18000) & (m39['hit_frame'] <= 27000)].sort_values('hit_frame').copy()
gt['local_frame'] = gt['hit_frame'] - 18000

pred_json = json.load(open('work_dirs/test_match39_10m_15m_full_pipeline_epoch20/timeline.json', encoding='utf-8'))
events = pred_json.get('events', [])

matched = []
for ev in events:
    f_pred = ev['frame']
    diffs = (gt['local_frame'] - f_pred).abs()
    best_idx = diffs.idxmin()
    best_diff = diffs.loc[best_idx]
    gt_row = gt.loc[best_idx] if best_diff <= 15 else None
    
    pred_stroke = ev['stroke']['label']
    pred_side = ev['stroke_side']['label']
    gt_stroke = str(gt_row['coarse_label']) if gt_row is not None else None
    gt_side = str(gt_row['stroke_side']) if gt_row is not None else None
    
    ok_stroke = (pred_stroke == gt_stroke) if gt_row is not None else None
    ok_side = (pred_side == gt_side) if gt_row is not None else None
    ok_joint = (ok_stroke and ok_side) if gt_row is not None else None
    
    matched.append({
        'id': ev.get('event_id', f"f{f_pred}"),
        'frame': f_pred,
        'time': round(ev.get('time_seconds', f_pred / 30.0), 2),
        'player': ev.get('court_player', 'upper'),
        'pred_stroke': pred_stroke,
        'pred_stroke_prob': round(float(ev['stroke']['probability']), 3),
        'pred_side': pred_side,
        'pred_side_prob': round(float(ev['stroke_side']['probability']), 3),
        'gt_stroke': gt_stroke,
        'gt_side': gt_side,
        'ok_stroke': ok_stroke,
        'ok_side': ok_side,
        'ok_joint': ok_joint,
        'is_hit_matched': bool(gt_row is not None)
    })

with open('work_dirs/match39_verified_eval.json', 'w', encoding='utf-8') as f:
    json.dump(matched, f, indent=2, ensure_ascii=False)

print(f"Exported match39_verified_eval.json with {len(matched)} events.")

