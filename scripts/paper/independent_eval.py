"""One-shot evaluation on ShuttleSet22 matches no model or decision has used (protocol in <output>/protocol.md).

Extraction uses the labeled hit frames, the deployed pipeline's pose, court and shuttle, and MATE on the
pipeline's crops. Scoring needs --score and refuses to run twice unless results are absent.

    python scripts/paper/independent_eval.py extract
    python scripts/paper/independent_eval.py score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts/training"), str(REPO_ROOT / "scripts/paper")]

from embed_pipeline_ownrgb import SS22, WORK, OwnRGBAdapter, save_atomic  # noqa: E402
from export_rgb_phases import PHASES  # noqa: E402
from reembed_rgb_phases import clip_from_video  # noqa: E402
from scripts.data.extract_shuttleset22_records import CKPTS, KEYS, ForwardCapture, court_corners, labeled_strokes  # noqa: E402
from scripts.inference.auto_court_detection import detect_court_line_corners  # noqa: E402
from scripts.inference.robust_aligned_video_pipeline import AlignedVideoPipeline  # noqa: E402
from summarize import macro_f1  # noqa: E402
from train_kap_ablation import KAPAblationModel  # noqa: E402
from train_rgb_scratch import MEAN, STD, RGBEncoder  # noqa: E402

MATCHES = {
    "pp134m934s8": "Lee_Zii_Jia_Li_Shi_Feng_GR_TOYOTA_GAZOO_RACING_Thailand_Open_2022_Finals",
    "s5ZlsOBrUfY": "LI_Shi_Feng_Kodai_NARAOKAGR_Thailand_Open_2022_Semi_finals",
    "T6evDKhWtgI": "CHEN_Yu_Fei_PUSARLA_V._Sindhu_Thailand_Open_2022_SemiFinals",
    "Da1aS0Kjqro": "CHEN_Yu_Fei_Ratchanok_INTANON_Denmark_Open_2022_SemiFinals",
    "RMPUSOPcACc": "SHI_Yu_Qi_Kodai_NARAOKA_Denmark_Open_2022_SemiFinals",
}
MINUTES = 10
CONTACT_RADIUS = 2  # Same as the deployed worker.
OUT = WORK / "paper_mate/independent"
ENCODER = WORK / "paper_ownrgb_ss22/x13/rgb_encoder_best.pth"


def load_encoder() -> RGBEncoder:
    encoder = RGBEncoder().cuda()
    encoder.load_state_dict(torch.load(ENCODER, weights_only=False)["model"])
    return encoder.eval()


def extract(youtube_id: str, folder: str, encoder: RGBEncoder, steps: list) -> None:
    target = OUT / f"{youtube_id}.pt"
    if target.exists():
        print(f"[skip] {folder}", flush=True)
        return
    video = SS22 / "videos" / f"{youtube_id}.mp4"
    cap = ForwardCapture(video)
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    strokes = [s for s in labeled_strokes(SS22, folder, court_corners(SS22, folder)) if 45 <= s["hit_frame"] < total - 60]
    end = strokes[0]["hit_frame"] + MINUTES * 60 * fps
    strokes = [s for s in strokes if s["hit_frame"] < end]
    corners = detect_court_line_corners(str(video))
    pipeline = AlignedVideoPipeline(
        ckpt_path=WORK / "kap_cross_v2_liftboost_seed20260926/kap_best.pth", corners=corners,
        orig_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), orig_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        pose_model_path=CKPTS / "yolov8n-pose.pt",
    )
    pipeline.rgb_model = OwnRGBAdapter(encoder, pipeline.rgb_model)
    pipeline.set_event_frames([s["hit_frame"] for s in strokes])
    rallies: dict = {}
    for stroke in strokes:
        rallies.setdefault((stroke["set"], stroke["rally"]), []).append(stroke)
    records = []
    for rally in sorted(rallies.values(), key=lambda group: group[0]["hit_frame"]):
        pipeline.prepare_event_cache(cap, [s["hit_frame"] for s in rally], total, contact_refine_radius=CONTACT_RADIUS)
        for stroke in rally:
            feat = pipeline.extract_features(cap, stroke["hit_frame"], total, hitter_side=stroke["side"], contact_refine_radius=CONTACT_RADIUS)
            if feat is not None:
                records.append({**stroke, **{key: feat[key].detach().cpu() for key in KEYS},
                                "crop_box": tuple(int(v) for v in feat["crop_box"]), "refined_hit_frame": int(feat["hit_frame"])})
    cap.release()

    cap = ForwardCapture(video)
    mean, std = MEAN.cuda(), STD.cuda()
    for record in sorted(records, key=lambda r: r["refined_hit_frame"]):
        clip = clip_from_video(cap, record, 0, total)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            steps.clear()
            logits = encoder((clip.cuda().permute(1, 0, 2, 3).unsqueeze(0).clamp(0, 1) - mean) / std)
            pooled = [steps[0][:, :, phase].mean(dim=-1) for phase in PHASES]
            record["emb"] = torch.stack([encoder._context, *pooled], dim=1).float().cpu()
            record["mate"] = logits.float().softmax(1).cpu()
    cap.release()
    save_atomic({"video": folder, "youtube_id": youtube_id, "court_corners": corners.tolist(), "strokes": len(strokes), "fps": fps, "records": records}, target)
    print(f"[+] {folder}: {len(records)}/{len(strokes)} records", flush=True)
    del pipeline
    torch.cuda.empty_cache()


def score() -> None:
    result_path = OUT / "results.json"
    if result_path.exists():
        raise SystemExit("already scored: this evaluation runs once")
    payloads = [torch.load(OUT / f"{i}.pt", weights_only=False) for i in MATCHES]
    records = [dict(r, video=p["video"]) for p in payloads for r in p["records"]]
    names = ("emb", "s_j", "s_p", "s_s", "s_c", "s_q")
    tensors = [torch.cat([r[n].float() for r in records]).cuda() for n in names]
    truth = np.array([r["gt_stroke_idx"] for r in records])
    videos = np.array([r["video"] for r in records])
    mate = torch.cat([r["mate"] for r in records]).numpy()
    print("shapes", [tuple(t.shape) for t in tensors], "events", len(truth), flush=True)

    def predict(family: str) -> list[np.ndarray]:
        out = []
        for run in sorted((WORK / "paper_mate/simple_final").glob(f"{family}_seed*/kap_best.pth")):
            model = KAPAblationModel(**json.loads((run.parent / "model_kwargs.json").read_text(encoding="utf-8")))
            model.load_state_dict(torch.load(run, map_location="cpu", weights_only=False)["model"])
            model = model.cuda().eval()
            with torch.inference_mode():
                out.append(torch.cat([model(*[t[i:i + 512] for t in tensors])[0].softmax(1).cpu() for i in range(0, len(truth), 512)]).numpy())
        return out

    def row(probabilities: list[np.ndarray], mask: np.ndarray | None = None) -> dict:
        mask = np.ones(len(truth), bool) if mask is None else mask
        acc = [float((p[mask].argmax(1) == truth[mask]).mean() * 100) for p in probabilities]
        f1 = [macro_f1(truth[mask], p[mask].argmax(1)) * 100 for p in probabilities]
        return {"acc": float(np.mean(acc)), "acc_std": float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0, "f1": float(np.mean(f1)), "per_seed": np.round(acc, 2).tolist()}

    no_rgb = predict("no_rgb")
    families = {"main": predict("main"), "full": predict("full"), "no_rgb": no_rgb, "full_no_rgb": predict("full_no_rgb"),
                "mate": [mate], "late_fusion": [0.5 * p + 0.5 * mate for p in no_rgb]}
    result = {"events": len(truth), "extracted": {p["video"]: f"{len(p['records'])}/{p['strokes']}" for p in payloads},
              "class_counts": np.bincount(truth, minlength=8).tolist(),
              "all": {k: row(v) for k, v in families.items()},
              "per_match": {str(v): {k: row(p, videos == v)["acc"] for k, p in families.items()} | {"n": int((videos == v).sum())} for v in sorted(set(videos))}}
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "score"))
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "score":
        score()
        return
    encoder = load_encoder()
    steps: list[torch.Tensor] = []
    encoder.net.temporal_tcn2.register_forward_hook(lambda _m, _i, out: steps.append(out))
    for youtube_id, folder in MATCHES.items():
        extract(youtube_id, folder, encoder, steps)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
