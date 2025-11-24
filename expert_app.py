from utils.logger import log
import os, json, torch, torch.nn as nn
from fastapi import FastAPI, Request, Response
from shared import dumps, loads, tensor_to_pack, pack_to_tensor

device='cuda' if torch.cuda.is_available() else 'cpu'
DIM=int(os.getenv('EMB_DIM','256')); EXPERT_ID=os.getenv('EXPERT_ID','0')
class ExpertMLP(nn.Module):
  def __init__(self,dim):
    super().__init__(); self.fc1=nn.Linear(dim,4*dim); self.fc2=nn.Linear(4*dim,dim); self.act=nn.GELU()
  def forward(self,x): return self.fc2(self.act(self.fc1(x)))
expert=ExpertMLP(DIM).to(device); optim=torch.optim.AdamW(expert.parameters(), lr=float(os.getenv('LR','1e-3')))
app=FastAPI()

@app.post('/fwd')
async def fwd(req: Request):
  p=loads(await req.body()); x=pack_to_tensor(p['x'],device); y=expert(x)
  return Response(content=dumps({'y': tensor_to_pack(y)}), media_type='application/msgpack')
  log(f"expert-{EXPERT_ID}", f"Received /fwd: input shape={x.shape}")

@app.post("/grad/apply")
async def grad_apply(req: Request):
    p = loads(await req.body())
    grads = p["grads"]   # {name: bytes}
    log(f"expert-{EXPERT_ID}", f"Received /grad/apply with {len(grads)} tensors")
    # 这里你可以选择真正应用梯度，或者先 no-op
    return {"ok": True}

@app.post('/step')
def step():
  torch.nn.utils.clip_grad_norm_(expert.parameters(), float(os.getenv('CLIP','1.0'))); optim.step(); optim.zero_grad(set_to_none=True); return {'ok':True}
  log(f"expert-{EXPERT_ID}", "Optimizer step() executed")

