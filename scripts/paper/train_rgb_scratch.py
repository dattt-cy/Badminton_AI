"""Train the team's own RGB stroke encoder from scratch and export its embeddings.

The encoder is BadmintonHitNetV2 (the team's spatio-temporal hit-spotting design:
RGB + frame difference, SE inverted-residual 2D encoder per frame, 4-scale dilated
TCN, temporal attention) with an 8-way stroke head.  It is randomly initialised and
trained only on the ShuttleSet train matches (selection on Match 41); no pretrained
weights and no external datasets are used.  The 224-d attention-pooled context
vector becomes the RGB token input of Robust KAP.

Outputs (work_dirs/paper_ownrgb/):
  rgb_encoder_best.pth, metrics.json, features_w15_30_ownrgb.npz (the KAP feature
  cache with the RGB embedding replaced).

Example:
    python scripts/paper/train_rgb_scratch.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT / "src", REPO_ROOT / "scripts" / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ai_classifier.models.badminton_hitnet import BadmintonHitNetV2  # noqa: E402
from train_kap_scratch import confusion_metrics, seed_everything  # noqa: E402

MEAN = torch.tensor([0.43, 0.43, 0.40]).view(3, 1, 1, 1)
STD = torch.tensor([0.23, 0.23, 0.23]).view(3, 1, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16"))
    parser.add_argument("--features", type=Path, default=REPO_ROOT / "work_dirs/context_window_pilot/features_w15_30.npz")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "work_dirs/paper_ownrgb")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


class ClipDataset(Dataset):
    def __init__(self, cache: Path, ids: list[str], labels: list[int], train: bool) -> None:
        self.cache, self.ids, self.labels, self.train = cache, ids, labels, train

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        path = self.cache / f"{self.ids[index]}.npy"
        if path.exists():
            clip = torch.from_numpy(np.load(path)).float().div_(255.0)  # T,C,H,W
        else:
            clip = torch.zeros(16, 3, 112, 112)
        clip = clip.permute(1, 0, 2, 3)  # C,T,H,W
        if self.train:
            # Photometric jitter and a small random shift; no horizontal flip so
            # the handedness of the stroke stays intact.
            clip = clip * (0.8 + 0.4 * torch.rand(1)) + (torch.rand(1) - 0.5) * 0.2
            dx, dy = np.random.randint(-8, 9, size=2)
            clip = torch.roll(clip, shifts=(int(dy), int(dx)), dims=(2, 3))
        clip = (clip.clamp(0, 1) - MEAN) / STD
        return clip, self.labels[index]


class RGBEncoder(nn.Module):
    """BadmintonHitNetV2 backbone with an 8-way stroke head; exposes the 224-d context vector."""

    def __init__(self) -> None:
        super().__init__()
        self.net = BadmintonHitNetV2(num_classes=8)
        self._context = None
        self.net.classifier.register_forward_hook(lambda _m, inputs, _o: setattr(self, "_context", inputs[0]))

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        return self.net(clips)

    def embed(self, clips: torch.Tensor) -> torch.Tensor:
        self.net(clips)
        return self._context


def evaluate(model, loader, device) -> dict:
    model.eval()
    matrix = torch.zeros(8, 8, dtype=torch.int64)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for clips, labels in loader:
            prediction = model(clips.to(device, non_blocking=True)).argmax(1).cpu()
            np.add.at(matrix.numpy(), (labels.numpy(), prediction.numpy()), 1)
    return confusion_metrics(matrix)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    raw = np.load(args.features)
    ids, split, stroke = raw["sample_id"], raw["split"], raw["stroke"]

    def subset(name: str, train: bool) -> ClipDataset:
        mask = split == name
        return ClipDataset(args.cache, [str(x) for x in ids[mask]], stroke[mask].tolist(), train)

    train_set, val_set = subset("train", True), subset("val", False)
    missing = sum(not (args.cache / f"{x}.npy").exists() for x in ids)
    print(f"[data] train={len(train_set)} val={len(val_set)} clips missing in cache={missing}", flush=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64, num_workers=args.workers, pin_memory=True, persistent_workers=True)

    model = RGBEncoder().to(device)
    print(f"[model] params={sum(p.numel() for p in model.parameters())}", flush=True)
    counts = np.bincount(stroke[split == "train"], minlength=8)
    weights = torch.tensor(len(train_set) / (8 * counts), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total, seen = time.time(), 0.0, 0
        for clips, labels in train_loader:
            clips, labels = clips.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(clips), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            total += float(loss) * len(labels); seen += len(labels)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "loss": total / seen, "val_acc": val["accuracy"], "val_f1": val["macro_f1"]})
        print(f"[rgb] {epoch:02d} loss={total / seen:.4f} val_acc={val['accuracy']:.4f} val_f1={val['macro_f1']:.4f} "
              f"({time.time() - start:.0f}s)", flush=True)
        if val["macro_f1"] > best:
            best = val["macro_f1"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output_dir / "rgb_encoder_best.pth")

    model.load_state_dict(torch.load(args.output_dir / "rgb_encoder_best.pth", weights_only=False)["model"])
    test_set = subset("test", False)
    test = evaluate(model, DataLoader(test_set, batch_size=64, num_workers=args.workers), device)
    print(f"[test] RGB alone acc={test['accuracy']:.4f} f1={test['macro_f1']:.4f}", flush=True)

    # Export embeddings for every event of the KAP cache.
    model.eval()
    everything = ClipDataset(args.cache, [str(x) for x in ids], stroke.tolist(), False)
    embeddings = []
    with torch.inference_mode():
        for clips, _ in DataLoader(everything, batch_size=64, num_workers=args.workers):
            embeddings.append(model.embed(clips.to(device)).float().cpu())
    rgb = torch.cat(embeddings).numpy().astype(np.float32)
    arrays = {k: raw[k] for k in raw.files}
    arrays["rgb"] = rgb
    np.savez(args.output_dir / "features_w15_30_ownrgb.npz", **arrays)
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "params": sum(p.numel() for p in model.parameters()), "best_val_f1": best, "history": history,
        "test_stroke": test, "embedding_dim": int(rgb.shape[1]), "missing_clips": int(missing),
        "training_data": "ShuttleSet train matches only, random initialisation",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"[export] {rgb.shape} -> features_w15_30_ownrgb.npz", flush=True)


if __name__ == "__main__":
    main()
