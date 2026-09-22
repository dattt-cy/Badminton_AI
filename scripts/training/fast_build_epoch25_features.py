"""Fast feature updater: reuse structured features from Epoch 20, re-extract RGB using Epoch 25."""
import sys, time
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_shuttleset_rgb_multitask import MultiTaskR2Plus1D
from scripts.training.train_shuttleset_feature_fusion import normalized_rgb

def build_epoch25_features():
    src_feat_path = Path("work_dirs/shuttleset_fusion_epoch20_full_b4/features.npz")
    rgb_cache = Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16")
    ckpt_path = Path("work_dirs/r2plus1d18_mixed_e21_e25_test/latest.pth")
    out_dir = Path("work_dirs/shuttleset_fusion_epoch25_full_b4")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_feat_path = out_dir / "features.npz"

    print("Loading source features.npz...")
    src = np.load(src_feat_path)
    sample_ids = src["sample_id"]
    total = len(sample_ids)
    print(f"Total samples to embed: {total}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading RGB Epoch 25 backbone on {device}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MultiTaskR2Plus1D()
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    rgb_embeddings = []
    batch = []
    batch_size = 32  # fast extraction

    t0 = time.time()
    with torch.inference_mode():
        for i, sid in enumerate(sample_ids):
            npy_path = rgb_cache / f"{sid}.npy"
            batch.append(normalized_rgb(npy_path))

            if len(batch) == batch_size or i == total - 1:
                tensor_batch = torch.stack(batch).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                    emb = model.backbone(tensor_batch).float().cpu().numpy()
                rgb_embeddings.extend(emb)
                batch = []

                if (i + 1) % 4000 == 0 or i == total - 1:
                    speed = (i + 1) / (time.time() - t0)
                    print(f"  [RGB Embedding] {i+1}/{total} ({speed:.1f} samples/s)", flush=True)

    rgb_arr = np.asarray(rgb_embeddings, dtype=np.float32)
    print(f"RGB extraction complete in {time.time()-t0:.1f}s. Saving new features.npz...")
    np.savez_compressed(
        out_feat_path,
        rgb=rgb_arr,
        structured=src["structured"],
        stroke=src["stroke"],
        side=src["side"],
        split=src["split"],
        sample_id=src["sample_id"]
    )
    print(f"Saved to {out_feat_path} successfully!")

if __name__ == "__main__":
    build_epoch25_features()

