import os, asyncio, httpx, torch, torch.nn as nn, torch.nn.functional as F, json
from fastapi import FastAPI, Response, Request
from shared import dumps, loads, tensor_to_pack, pack_to_tensor, route_pack
from hotcold import HotColdManager
from scheduler import LGBMScheduler
from utils.logger import log
from scheduler_lgbm import lgb_select_instance
import json


device='cuda' if torch.cuda.is_available() else 'cpu'
TOP_K=int(os.getenv('TOP_K','2')); HC=HotColdManager(); SCHED=LGBMScheduler()
EXPERT_INSTANCE_TABLE=json.loads(os.getenv('EXP_INSTANCES_JSON','{}'))

class PreModel(nn.Module):
  def __init__(self,vocab,dim,L,nE):
    super().__init__(); self.embed=nn.Embedding(vocab,dim); self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(dim,8,4*dim) for _ in range(L)]); self.router=nn.Linear(dim, nE)
  def forward_until_router(self,x_ids):
    x=self.embed(x_ids); x=x.transpose(0,1)
    for b in self.blocks: x=b(x)
    x=x.transpose(0,1); logits=self.router(x); return x, logits

pre=None; optim=None
def init_model():
  global pre,optim
  vocab=int(os.getenv('VOCAB_SIZE','2000')); dim=int(os.getenv('EMB_DIM','256')); L=int(os.getenv('N_LAYERS_PRE','2'))
  nE=max(1, len(EXPERT_INSTANCE_TABLE)); pre=PreModel(vocab, dim, L, nE).to(device); optim=torch.optim.AdamW(pre.parameters(), lr=float(os.getenv('LR','1e-3')))
app=FastAPI(on_startup=[init_model])


async def call_experts(xt, idx, gate, emb_dim: int):
  """
  xt: [N, D] 展平后的 token 表示
  idx: [N, K] 每个 token 的专家编号
  gate: [N, K] 每个专家的 gate 权重
  """
  import httpx

  N, D = xt.shape
  K = idx.size(-1)

  # 收集每个 expert_id 对应的 token 子集
  # route_map[eid] = list of (token_idx, k_slot, gate_value)
  route_map = {}
  for t in range(N):
    for k in range(K):
      eid = int(idx[t, k].item())
      g = gate[t, k].item()
      if g <= 0:
        continue
      route_map.setdefault(eid, []).append((t, k, g))

  out = torch.zeros_like(xt, device=xt.device)

  async with httpx.AsyncClient(timeout=None) as client:
    tasks = []

    for eid, items in route_map.items():
      # 取出候选实例列表
      inst_list = EXPERT_INSTANCES.get(str(eid), [])
      if not inst_list:
        log("pre-fn", f"No instances for expert {eid}, skip")
        continue

      # 构造要发给该 expert 的子 batch
      token_indices = [t for (t, k, g) in items]
      sub_x = xt[token_indices]  # [n_e, D]

      # 使用 LightGBM 在多个实例中选一个
      if USE_LGBM and len(inst_list) > 1:
        req = {"tokens": len(token_indices)}
        best_inst, score = lgb_select_instance(eid, inst_list, req)
      else:
        best_inst = inst_list[0]
        score = 0.0

      url = best_inst.get("url")
      log(
        "pre-fn",
        f"expert={eid} send {len(token_indices)} tokens to inst={best_inst.get('id')} "
        f"url={url}, lgb_score={score:.4f}",
      )

      # 异步 HTTP 请求
      async def _call_one(sub_x=sub_x, token_indices=token_indices, eid=eid, url=url):
        from shared import tensor_to_pack, pack_to_tensor, dumps, loads
        r = await client.post(
          url + "/fwd",
          content=dumps({"x": tensor_to_pack(sub_x)}),
          headers={"Content-Type": "application/msgpack"},
        )
        resp = loads(r.content)
        y_sub = pack_to_tensor(resp["y"], xt.device)  # [n_e, D]
        # 写回到 out
        out[token_indices] = y_sub

      tasks.append(_call_one())

    if tasks:
      await asyncio.gather(*tasks)

  return out


@app.post('/fwd')
async def fwd(req: Request):
  pay=loads(await req.body()); x_ids=pack_to_tensor(pay['x_ids'],device).long(); pre.train(); x,logits=pre.forward_until_router(x_ids)
  probs=torch.softmax(logits,dim=-1); topk=torch.topk(probs,k=min(TOP_K, probs.size(-1)),dim=-1); idx=topk.indices.reshape(-1,topk.indices.size(-1)); gate=topk.values.reshape(-1,topk.values.size(-1))
  HC.observe_batch(idx.view(-1).tolist()); xt=x.reshape(-1,x.size(-1)); yt=await call_experts(xt,idx,gate,emb_dim=x.size(-1)); y=yt.reshape(x.shape)
  return Response(content=dumps({'y':tensor_to_pack(y),'route':route_pack(idx,gate),'micro_id':pay.get('micro_id',0)}), media_type='application/msgpack')
  log("pre-fn", f"Received /fwd batch size={len(tokens)}")
  log("pre-fn", f"Selected experts={selected_experts}")
  log("pre-fn", "Dispatching to expert instances...")

@app.post('/bwd')
async def bwd(req: Request):
  p=loads(await req.body()); x_ids=pack_to_tensor(p['x_ids'],device); dy=pack_to_tensor(p['dy'],device); x,logits=pre.forward_until_router(x_ids); y=x; y.backward(dy); return Response(content=dumps({'ok':True}), media_type='application/msgpack')
  log("pre-fn", "Received /bwd request")
  log("pre-fn", f"Gradient from post_fn: shape={dy.shape}")


@app.post('/step')
def step():
  torch.nn.utils.clip_grad_norm_(pre.parameters(), float(os.getenv('CLIP','1.0'))); optim.step(); optim.zero_grad(set_to_none=True); return {'ok':True}

CONFIG_PATH = os.getenv("EXP_INSTANCES_FILE", "experts.json")

def load_expert_instances(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log("pre-fn", f"Loaded expert instances from {path}, keys={list(data.keys())}")
        return data
    except FileNotFoundError:
        log("pre-fn", f"WARNING: config file {path} not found, using empty EXP_INSTANCES")
        return {}
    except Exception as e:
        log("pre-fn", f"ERROR loading {path}: {e}")
        return {}

EXPERT_INSTANCES = load_expert_instances(CONFIG_PATH)
USE_LGBM = os.getenv("USE_LGBM", "1") == "1"
