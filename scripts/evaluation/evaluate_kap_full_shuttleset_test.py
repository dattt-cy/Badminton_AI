"""Evaluate the frozen epoch-19 Robust KAP checkpoint on all five test matches.

This script never trains or selects a checkpoint. It constructs missing RGB
frame caches with the same hitter-crop protocol used by the frozen RGB model,
extracts frozen embeddings, and evaluates matches 39, 40, 42, 43, and 44.
"""
from __future__ import annotations

import argparse, csv, json, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r2plus1d_18
from torchvision.transforms import functional as VF
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from ai_classifier.models import KAPScratchModel, SIDE_CLASSES, STROKE_CLASSES
from ai_classifier.features import modality_quality_vector
from scripts.training.train_kap_scratch import confusion_metrics

TEST_MATCHES = {"39", "40", "42", "43", "44"}

@dataclass(frozen=True)
class Row:
    sample_id: str; match_id: str; video_path: Path; start: int; end: int
    hit: int; player_side: str; stroke: str; side: str
    set_id: str; rally: str; ball_round: int
    joints: str; position: str; shuttle: str

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=ROOT/"data/manifests/shuttleset_npy.csv")
    p.add_argument("--rgb-manifest",type=Path,default=ROOT/"data/manifests/shuttleset_rgb.csv")
    p.add_argument("--npy-root",type=Path,default=Path(r"C:\Users\ADMIN\Downloads\dataset_npy_between_2_hits_with_max_limits\dataset_npy_between_2_hits_with_max_limits"))
    p.add_argument("--rgb-cache",type=Path,default=Path(r"C:\Users\ADMIN\Downloads\AI_Classifier_cache\shuttleset_v3_full\frames_16"))
    p.add_argument("--rgb-checkpoint",type=Path,default=ROOT/"work_dirs/r2plus1d18_mixed_shuttleset_finebadminton/best.pth")
    p.add_argument("--kap-checkpoint",type=Path,default=ROOT/"work_dirs/kap_robust_seed20260926/kap_best.pth")
    p.add_argument("--pose-model",type=Path,default=ROOT/"models/checkpoints/pose/yolov8n-pose.pt")
    p.add_argument("--old-features",type=Path,default=ROOT/"work_dirs/cr_gated_fusion/sequence_features.npz")
    p.add_argument("--output",type=Path,default=ROOT/"work_dirs/kap_full_test5_evaluation.json")
    return p.parse_args()

def load_rows(a):
    with a.rgb_manifest.open(encoding="utf-8-sig",newline="") as f:
        rgb={r["sample_id"]:r for r in csv.DictReader(f)}
    with a.manifest.open(encoding="utf-8-sig",newline="") as f:
        source=list(csv.DictReader(f))
    # Contact indices must be computed from every event in the rally before
    # label filtering; otherwise a removed event changes the previous-hit bound.
    groups=defaultdict(list)
    for r in source:
        groups[(r["match_id"],r["set_id"],r["rally"])].append(r)
    targets={}
    for group in groups.values():
        ordered=sorted(group,key=lambda r:int(r["ball_round"]))
        for i,r in enumerate(ordered):
            hit=int(float(r["hit_frame"]));previous=int(float(ordered[i-1]["hit_frame"])) if i else hit-15
            targets[r["sample_id"]]=hit-max(previous,hit-45)
    rows=[]
    for r in source:
        if r["split"]!="test" or r["match_id"] not in TEST_MATCHES: continue
        if r["valid_coarse_label"]!="True" or r["valid_stroke_side"]!="True": continue
        v=rgb.get(r["sample_id"])
        if not v: continue
        rows.append(Row(r["sample_id"],r["match_id"],Path(v["video_path"]),int(v["start_frame"]),int(v["end_frame"]),int(v["hit_frame"]),v["player_side"],r["coarse_label"],r["stroke_side"],r["set_id"],r["rally"],int(r["ball_round"]),r["joints_path"],r["position_path"],r["shuttle_path"]))
    return sorted(rows,key=lambda x:(int(x.match_id),x.sample_id)),targets

def select_box(result, side, w, h):
    if result.boxes is None or result.keypoints is None:return None
    out=[]
    for box,cls,pose in zip(result.boxes.xyxy.cpu().numpy(),result.boxes.cls.cpu().numpy(),result.keypoints.data.cpu().numpy()):
        if int(cls)!=0:continue
        vis=pose[:,2]>=.25; c=pose[vis,:2].mean(0) if vis.any() else np.array([(box[0]+box[2])/2,(box[1]+box[3])/2]); x,y=c
        lo,hi=(.32,.68) if side=="top" else (.22,.78); miny=.25 if side=="top" else .42
        if not(lo*w<=x<=hi*w and miny*h<=y<=.92*h):continue
        if side=="top" and y>=.58*h:continue
        out.append((float(y),box))
    if not out:return None
    out.sort(key=lambda z:z[0]);return out[0][1] if side=="top" else out[-1][1]

def square(box,w,h,pad=.55):
    x1,y1,x2,y2=map(float,box);cx,cy=(x1+x2)/2,(y1+y2)/2;s=max(max(x2-x1,y2-y1)*(1+2*pad),min(w,h)*.22)
    return max(0,round(cx-s/2)),max(0,round(cy-s*.58)),min(w,round(cx+s/2)),min(h,round(cy+s*.42))

def make_cache(row, path, pose_model):
    cap=cv2.VideoCapture(str(row.video_path)); crop=None
    try:
        for off in (0,-2,2,-4,4,-6,6):
            cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,row.hit+off));ok,fr=cap.read()
            if not ok:continue
            res=pose_model.predict(fr,imgsz=960,conf=.20,verbose=False)[0];h,w=fr.shape[:2];box=select_box(res,row.player_side,w,h)
            if box is not None:crop=square(box,w,h);break
        wanted=np.rint(np.linspace(row.start,row.end,16)).astype(int); cap.set(cv2.CAP_PROP_POS_FRAMES,int(wanted[0])); got={}; ws=set(map(int,wanted))
        for i in range(int(wanted[0]),int(wanted[-1])+1):
            ok,fr=cap.read()
            if not ok:raise RuntimeError(f"decode stopped: {row.sample_id}@{i}")
            if i in ws:
                fr=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
                if crop:x1,y1,x2,y2=crop;fr=fr[y1:y2,x1:x2]
                got[i]=fr
        t=torch.from_numpy(np.stack([got[int(i)] for i in wanted]).copy()).permute(0,3,1,2)
        if crop:t=VF.resize(t,[112,112],antialias=False)
        else:t=VF.center_crop(VF.resize(t,[128,171],antialias=False),[112,112])
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp.npy");np.save(tmp,t.numpy().astype(np.uint8));tmp.replace(path)
    finally:cap.release()

class RGB(nn.Module):
    def __init__(self):
        super().__init__();self.backbone=r2plus1d_18(weights=None);self.backbone.fc=nn.Identity();self.stroke_head=nn.Linear(512,8);self.side_head=nn.Linear(512,3)

def norm_clip(path):
    t=torch.from_numpy(np.load(path)).float()/255;mean=t.new_tensor([.43216,.394666,.37645]).view(1,3,1,1);std=t.new_tensor([.22803,.22145,.216989]).view(1,3,1,1)
    return ((t-mean)/std).permute(1,0,2,3)

def resample(x,length=32):
    v=np.nan_to_num(x.astype(np.float32),nan=0).reshape(len(x),-1)
    if len(v)==length:return v
    source=np.linspace(0,1,len(v));target=np.linspace(0,1,length)
    return np.stack([np.interp(target,source,v[:,i]) for i in range(v.shape[1])],1).astype(np.float32)

def extract_sample_sequences(joints,position,shuttle,target_index,length=32,before=15,after=30):
    window=[];wl=before+1+after
    for array in (joints,position,shuttle):
        t=min(max(int(target_index),0),max(len(array)-1,0));out=np.zeros((wl,*array.shape[1:]),np.float32)
        ss=max(0,t-before);se=min(len(array),t+after+1);ds=before-min(before,t);span=min(se-ss,wl-ds)
        if span>0:out[ds:ds+span]=array[ss:ss+span]
        window.append(out)
    j,p,s=window;q=modality_quality_vector(j,s)
    # The frozen checkpoint was trained with the cache builder's historical
    # coordinate-scale convention for the two motion magnitudes.
    q=np.asarray(q,dtype=np.float32);q[[1,3]]/=100.0
    contact=np.linspace(-before,after,length,dtype=np.float32)[:,None]/before
    return resample(j,length).reshape(length,-1),resample(p,length).reshape(length,-1),resample(s,length).reshape(length,-1),contact,q

def main():
    a=args();rows,targets=load_rows(a);print(f"eligible={len(rows)} matches={sorted(TEST_MATCHES)}",flush=True)
    missing=[r for r in rows if not(a.rgb_cache/f"{r.sample_id}.npy").is_file()]
    if missing:
        pose=YOLO(str(a.pose_model));
        for i,r in enumerate(missing,1):
            make_cache(r,a.rgb_cache/f"{r.sample_id}.npy",pose)
            if i%25==0 or i==len(missing):print(f"cache={i}/{len(missing)}",flush=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu");print(f"device={device}",flush=True)
    rgb_model=RGB();rgb_model.load_state_dict(torch.load(a.rgb_checkpoint,map_location="cpu",weights_only=False)["model"]);rgb_model.to(device).eval()
    old=np.load(a.old_features);old_ids=old["sample_id"];old_rgb=old["rgb"]
    oldmap={str(x):old_rgb[i] for i,x in enumerate(old_ids)}
    emb=[]
    with torch.inference_mode():
        for start in range(0,len(rows),2):
            batch=rows[start:start+2]; vals=[]; todo=[]
            for i,r in enumerate(batch):
                if r.sample_id in oldmap:vals.append(oldmap[r.sample_id])
                else:vals.append(None);todo.append(i)
            if todo:
                clips=torch.stack([norm_clip(a.rgb_cache/f"{batch[i].sample_id}.npy") for i in todo]).to(device)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):new=rgb_model.backbone(clips).float().cpu().numpy()
                for i,v in zip(todo,new):vals[i]=v
            emb.extend(vals)
            if start%100==0:print(f"embedding={min(start+2,len(rows))}/{len(rows)}",flush=True)
    posev=[];courtv=[];shuttlev=[];contactv=[];qualityv=[]
    for r in rows:
        j=np.load(a.npy_root/r.joints);p=np.load(a.npy_root/r.position);s=np.load(a.npy_root/r.shuttle)
        x=extract_sample_sequences(j,p,s,targets[r.sample_id]);posev.append(x[0]);courtv.append(x[1]);shuttlev.append(x[2]);contactv.append(x[3]);qualityv.append(x[4])
    kap=KAPScratchModel();payload=torch.load(a.kap_checkpoint,map_location=device,weights_only=False);kap.load_state_dict(payload["model"]);kap.to(device).eval()
    arrays=[np.asarray(x,dtype=np.float32) for x in (emb,posev,courtv,shuttlev,contactv,qualityv)];ys=np.array([STROKE_CLASSES.index(r.stroke) for r in rows]);yd=np.array([SIDE_CLASSES.index(r.side) for r in rows])
    cm_s=torch.zeros(8,8,dtype=torch.int64);cm_d=torch.zeros(3,3,dtype=torch.int64);per={m:{"stroke":torch.zeros(8,8,dtype=torch.int64),"side":torch.zeros(3,3,dtype=torch.int64)} for m in TEST_MATCHES}
    with torch.inference_mode():
        for start in range(0,len(rows),128):
            ids=range(start,min(start+128,len(rows)));inp=[torch.from_numpy(x[start:start+128]).to(device) for x in arrays];os,od,_=kap(*inp);ps=os.argmax(1).cpu();pd=od.argmax(1).cpu()
            for k,i in enumerate(ids):
                cm_s[ys[i],ps[k]]+=1;cm_d[yd[i],pd[k]]+=1;m=rows[i].match_id;per[m]["stroke"][ys[i],ps[k]]+=1;per[m]["side"][yd[i],pd[k]]+=1
    result={"protocol":"frozen_epoch19_full_five_match_test","checkpoint_epoch":payload["epoch"],"test_matches":sorted(TEST_MATCHES),"samples":len(rows),"stroke":confusion_metrics(cm_s),"side":confusion_metrics(cm_d),"per_match":{m:{"samples":sum(r.match_id==m for r in rows),"stroke":confusion_metrics(v["stroke"]),"side":confusion_metrics(v["side"])} for m,v in per.items()}}
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps({"samples":result["samples"],"stroke":{k:result["stroke"][k] for k in ("accuracy","balanced_accuracy","macro_f1")},"side":{k:result["side"][k] for k in ("accuracy","balanced_accuracy","macro_f1")}},indent=2),flush=True)
if __name__=="__main__":main()
