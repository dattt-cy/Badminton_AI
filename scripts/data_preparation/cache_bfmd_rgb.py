"""Cache BFMD hitter crops as 16-frame uint8 NPY tensors."""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.training.train_shuttleset_rgb_multitask import decode_clip, load_records, padded_square


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfmd-root",type=Path,required=True)
    p.add_argument("--manifest-dir",type=Path,default=Path("data/manifests/bfmd_splits"))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--splits",nargs="+",choices=("train","val","test"),default=["train","val"])
    p.add_argument("--frames",type=int,default=16)
    p.add_argument("--crop-size",type=int,default=112)
    p.add_argument("--padding",type=float,default=0.55)
    p.add_argument("--max-samples",type=int,help="Optional smoke-test limit after loading splits.")
    return p.parse_args()


def load_tracks(path: Path) -> dict[str, dict[int, tuple[float,float,float,float]]]:
    payload=json.loads(path.read_text(encoding="utf-8"));tracks={"top":{},"bottom":{}}
    for item in payload.get("annotations",[{}])[0].get("result",[]):
        value=item.get("value",{});labels=value.get("labels",[])
        if not labels: continue
        label=labels[0].lower();side="top" if "top" in label else "bottom" if "bottom" in label else None
        if side is None: continue
        for box in value.get("sequence",[]):
            if not box.get("enabled",True): continue
            tracks[side][int(box["frame"])]=(float(box["x"]),float(box["y"]),float(box["width"]),float(box["height"]))
    return tracks


def crop_from_percent(box, width=1280, height=720, padding=.55):
    x,y,w,h=box
    pixels=np.asarray([x*width/100,y*height/100,(x+w)*width/100,(y+h)*height/100],dtype=np.float32)
    return padded_square(pixels,width,height,padding)


def nearest_box(track, frame, max_distance=5):
    if frame in track:return track[frame]
    candidates=[key for key in track if abs(key-frame)<=max_distance]
    return track[min(candidates,key=lambda key:abs(key-frame))] if candidates else None


def main():
    args=parse_args();out=args.output_dir/(f"frames_{args.frames}" if args.crop_size==112 else f"frames_{args.frames}_crop_{args.crop_size}");out.mkdir(parents=True,exist_ok=True)
    bbox_dir=args.bfmd_root/"annotations"/"player_bbox";track_cache={};total=done=created=fallback=0;started=time.perf_counter()
    jobs=[]
    for split in args.splits:
        jobs.extend((split,record)for record in load_records(args.manifest_dir/f"{split}.csv",split))
    if args.max_samples is not None: jobs=jobs[:args.max_samples]
    total=len(jobs)
    for index,(split,record) in enumerate(jobs,1):
        target=out/f"{record.sample_id}.npy"
        if target.is_file():done+=1
        else:
            match=record.sample_id[len("bfmd_"):].rsplit("_",3)[0]
            if match not in track_cache:track_cache[match]=load_tracks(bbox_dir/f"{match}.json")
            box=nearest_box(track_cache[match][record.player_side],record.hit_frame)
            if box is None:
                fallback+=1
                # Conservative half-court fallback; reported in final summary.
                crop=(282,72,998,504) if record.player_side=="top" else (282,216,998,648)
            else:crop=crop_from_percent(box,padding=args.padding)
            tensor=decode_clip(record,args.frames,crop,args.crop_size);tmp=target.with_suffix(".tmp.npy");np.save(tmp,tensor.numpy());tmp.replace(target);done+=1;created+=1
        if index%50==0 or index==total:
            elapsed=time.perf_counter()-started;print(f"cache split={split} samples={index}/{total} done={done} created={created} fallback={fallback} rate={done/max(elapsed,1e-6):.2f}/s",flush=True)
    print(f"[DONE] total={total} created={created} fallback={fallback} output={out}",flush=True)


if __name__=="__main__":main()
