"""Head-only X3D-S pilot on ShuttleSet, Fine-Badminton, and BFMD caches."""
import argparse,json,random,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,ConcatDataset,WeightedRandomSampler
from torchvision.transforms import functional as VF
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.training.train_fine_badminton_rgb import confusion_metrics
from ai_classifier.datasets import load_fine_badminton_manifest as load_fine
from scripts.training.train_shuttleset_rgb_multitask import STROKE_CLASSES,load_records

def select(records,count,seed):
 g=defaultdict(list)
 for r in records:g[r.coarse_label].append(r)
 rng=random.Random(seed);out=[]
 for c in STROKE_CLASSES:
  x=sorted(g[c],key=lambda r:r.sample_id);out+=rng.sample(x,min(count,len(x)))
 rng.shuffle(out);return out

class Cache(Dataset):
 def __init__(self,records,root,prefix,training):self.records=records;self.root=root/'frames_16';self.prefix=prefix;self.training=training
 def __len__(self):return len(self.records)
 def __getitem__(self,i):
  r=self.records[i];x=torch.from_numpy(np.load(self.root/f'{r.sample_id}.npy')).float()/255;x=x[torch.linspace(0,15,13).round().long()];x=VF.resize(x,[182,182],antialias=False);x=(x-.45)/.225;x=x.permute(1,0,2,3)
  if self.training and torch.rand(())<.5:x=torch.flip(x,(-1,))
  return x,STROKE_CLASSES.index(r.coarse_label),f'{self.prefix}:{r.sample_id}'

def eval_model(model,loader,device):
 c=torch.zeros(8,8,dtype=torch.int64);model.eval()
 with torch.inference_mode():
  for x,y,_ in loader:
   with torch.autocast(device_type=device.type,dtype=torch.float16):p=model(x.to(device)).argmax(1).cpu()
   for a,b in zip(y,p):c[int(a),int(b)]+=1
 return confusion_metrics(c)

def main():
 p=argparse.ArgumentParser();p.add_argument('--shuttle-cache',type=Path,required=True);p.add_argument('--fine-cache',type=Path,required=True);p.add_argument('--bfmd-cache',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--train-per-class',type=int,default=200);p.add_argument('--val-per-class',type=int,default=50);p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=16);a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 sources=[]
 for name,tr,va,cache in [('shuttle',load_records(Path('data/manifests/shuttleset_rgb.csv'),'train'),load_records(Path('data/manifests/shuttleset_rgb.csv'),'val'),a.shuttle_cache),('fine',load_fine(Path('data/manifests/fine_badminton_splits/train.csv')),load_fine(Path('data/manifests/fine_badminton_splits/val.csv')),a.fine_cache),('bfmd',load_records(Path('data/manifests/bfmd_splits/train.csv'),'train'),load_records(Path('data/manifests/bfmd_splits/val.csv'),'val'),a.bfmd_cache)]:sources.append((name,select(tr,a.train_per_class,20260922),select(va,a.val_per_class,20260923),cache))
 trainsets=[Cache(tr,c,n,True)for n,tr,_,c in sources];valsets=[Cache(va,c,n,False)for n,_,va,c in sources];weights=[]
 for _,tr,_,_ in sources:
  counts=Counter(r.coarse_label for r in tr);weights += [1/(3*8*counts[r.coarse_label])for r in tr]
 train=ConcatDataset(trainsets);val=ConcatDataset(valsets);sampler=WeightedRandomSampler(weights,len(train),replacement=True,generator=torch.Generator().manual_seed(20260922));tl=DataLoader(train,batch_size=a.batch_size,sampler=sampler,num_workers=0);vl=DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0)
 model=torch.hub.load('facebookresearch/pytorchvideo','x3d_s',pretrained=True);model.blocks[-1].proj=nn.Linear(2048,8)
 for p0 in model.parameters():p0.requires_grad=False
 for p0 in model.blocks[-1].proj.parameters():p0.requires_grad=True
 device=torch.device('cuda');model.to(device);opt=torch.optim.AdamW(model.blocks[-1].proj.parameters(),lr=3e-4,weight_decay=1e-4);lossfn=nn.CrossEntropyLoss();scaler=torch.amp.GradScaler('cuda');history=[];best=-1
 print('samples',{'train':len(train),'val':len(val),'sources':{n:(len(tr),len(va))for n,tr,va,_ in sources}},flush=True)
 for epoch in range(1,a.epochs+1):
  model.train();
  for block in model.blocks[:-1]: block.eval()
  total=0
  for step,(x,y,_) in enumerate(tl,1):
   opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16):loss=lossfn(model(x.to(device)),y.to(device))
   scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=float(loss)
   if step%100==0:print(f'epoch={epoch}/{a.epochs} step={step}/{len(tl)} loss={total/step:.4f}',flush=True)
  m=eval_model(model,vl,device);history.append({'epoch':epoch,'loss':total/len(tl),'val':m});print(f"epoch={epoch}/{a.epochs} val_macro_f1={m['macro_f1']:.4f} val_accuracy={m['accuracy']:.4f}",flush=True)
  if m['macro_f1']>best:best=m['macro_f1'];torch.save({'model':model.state_dict(),'epoch':epoch,'classes':STROKE_CLASSES,'architecture':'x3d_s','frames':13,'crop_size':182},a.output_dir/'best.pth')
 (a.output_dir/'metrics.json').write_text(json.dumps({'best_macro_f1':best,'history':history},indent=2)+'\n')
if __name__=='__main__':main()
