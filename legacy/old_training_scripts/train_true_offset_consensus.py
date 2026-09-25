"""Train an 8-class temporal head from real hit-centered video offsets."""

from __future__ import annotations

import argparse, csv, json, random, sys, time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_fine_badminton_rgb import confusion_metrics
from scripts.training.train_shuttleset_rgb_multitask import (
    MultiTaskR2Plus1D, STROKE_CLASSES, decode_clip, detect_hitter_crop, load_records,
)


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=Path("data/manifests/shuttleset_rgb.csv"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--offsets", type=int, nargs="+", default=[-4, 0, 4])
    p.add_argument("--train-per-class", type=int, default=100)
    p.add_argument("--val-per-class", type=int, default=50)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260921)
    return p.parse_args()


def balanced(records, count, seed):
    groups = defaultdict(list)
    for record in records: groups[record.coarse_label].append(record)
    rng, output = random.Random(seed), []
    for label in STROKE_CLASSES:
        candidates = sorted(groups[label], key=lambda item: item.sample_id)
        output.extend(rng.sample(candidates, min(count, len(candidates))))
    rng.shuffle(output)
    return output


def cache_path(root, split, sample_id, offset):
    sign = f"p{offset}" if offset >= 0 else f"m{abs(offset)}"
    return root / split / sign / f"{sample_id}.npy"


def materialize(records, split, offsets, root, pose):
    total, done, started = len(records) * len(offsets), 0, time.perf_counter()
    for index, record in enumerate(records, 1):
        paths = [cache_path(root, split, record.sample_id, offset) for offset in offsets]
        if all(path.is_file() for path in paths):
            done += len(paths)
        else:
            crop = detect_hitter_crop(record, pose, 0.55)
            for offset, path in zip(offsets, paths):
                if not path.is_file():
                    shifted = replace(
                        record, start_frame=max(0, record.start_frame + offset),
                        end_frame=max(0, record.end_frame + offset),
                        hit_frame=max(0, record.hit_frame + offset),
                    )
                    tensor = decode_clip(shifted, 16, crop, 112)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp.npy")
                    np.save(temporary, tensor.numpy()); temporary.replace(path)
                done += 1
        if index % 25 == 0 or index == len(records):
            rate = done / max(time.perf_counter() - started, 1e-6)
            print(f"cache split={split} samples={index}/{len(records)} clips={done}/{total} rate={rate:.2f}/s", flush=True)


def normalize(array):
    x = torch.from_numpy(array).float().div_(255.0)
    mean = x.new_tensor([0.43216, 0.394666, 0.37645]).view(1,3,1,1)
    std = x.new_tensor([0.22803, 0.22145, 0.216989]).view(1,3,1,1)
    return ((x - mean) / std).permute(1,0,2,3)


def embeddings(records, split, offsets, root, backbone, device, batch_size):
    output, labels, pending, started = [], [], [], time.perf_counter()
    for index, record in enumerate(records, 1):
        pending.extend(normalize(np.load(cache_path(root, split, record.sample_id, o))) for o in offsets)
        labels.append(STROKE_CLASSES.index(record.coarse_label))
        if len(labels) % batch_size == 0 or index == len(records):
            current = min(batch_size, len(labels) - sum(len(item) for item in output))
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
                values = backbone(torch.stack(pending).to(device)).float().cpu().reshape(current, len(offsets), -1)
            output.append(values); pending = []
        if index % 100 == 0 or index == len(records):
            print(f"embedding split={split} samples={index}/{len(records)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    return torch.cat(output), torch.tensor(labels)


class Head(nn.Module):
    def __init__(self, dim, views):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(dim*views),nn.Linear(dim*views,256),nn.GELU(),nn.Dropout(.25),nn.Linear(256,8))
    def forward(self,x): return self.net(x.flatten(1))


def score(logits, labels):
    c=torch.zeros(8,8,dtype=torch.int64)
    for t,p in zip(labels,logits.argmax(1).cpu()): c[int(t),int(p)]+=1
    return confusion_metrics(c)


def main():
    args=args_parser(); args.output_dir.mkdir(parents=True,exist_ok=True)
    train=balanced(load_records(args.manifest,"train"),args.train_per_class,args.seed)
    val=balanced(load_records(args.manifest,"val"),args.val_per_class,args.seed+1)
    pose=YOLO("models/checkpoints/pose/yolov8n-pose.pt")
    materialize(train,"train",args.offsets,args.cache_dir,pose); materialize(val,"val",args.offsets,args.cache_dir,pose)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False); base=MultiTaskR2Plus1D();base.load_state_dict(ck["model"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); backbone=base.backbone.to(device).eval()
    for parameter in backbone.parameters(): parameter.requires_grad=False
    feature_file=args.output_dir/"features.pt"
    if feature_file.is_file(): data=torch.load(feature_file,map_location="cpu",weights_only=False);print("embedding cache resumed",flush=True)
    else:
        tx,ty=embeddings(train,"train",args.offsets,args.cache_dir,backbone,device,args.batch_size)
        vx,vy=embeddings(val,"val",args.offsets,args.cache_dir,backbone,device,args.batch_size)
        data={"train_x":tx,"train_y":ty,"val_x":vx,"val_y":vy,"offsets":args.offsets};torch.save(data,feature_file);print(f"embedding cache saved={feature_file}",flush=True)
    tx,ty,vx,vy=data["train_x"],data["train_y"],data["val_x"],data["val_y"]
    with torch.inference_mode(): baseline=score(base.stroke_head(vx[:,args.offsets.index(0),:]),vy)
    print(f"baseline val_macro_f1={baseline['macro_f1']:.4f}",flush=True)
    head=Head(tx.shape[-1],len(args.offsets)).to(device);opt=torch.optim.AdamW(head.parameters(),lr=2e-4,weight_decay=1e-4);loss=nn.CrossEntropyLoss();rng=np.random.default_rng(args.seed);history=[];best=-1
    for epoch in range(1,args.epochs+1):
        head.train();order=rng.permutation(len(ty))
        for start in range(0,len(order),64):
            ids=torch.from_numpy(order[start:start+64]);opt.zero_grad(set_to_none=True);value=loss(head(tx[ids].to(device)),ty[ids].to(device));value.backward();opt.step()
        head.eval()
        with torch.inference_mode(): result=score(head(vx.to(device)),vy)
        history.append({"epoch":epoch,"val":result});print(f"epoch={epoch}/{args.epochs} val_macro_f1={result['macro_f1']:.4f} baseline={baseline['macro_f1']:.4f}",flush=True)
        if result["macro_f1"]>best: best=result["macro_f1"];torch.save({"model":head.state_dict(),"epoch":epoch,"offsets":args.offsets,"classes":STROKE_CLASSES},args.output_dir/"best.pth")
    (args.output_dir/"metrics.json").write_text(json.dumps({"baseline":baseline,"best_macro_f1":best,"history":history},indent=2)+"\n")

if __name__=="__main__": main()
