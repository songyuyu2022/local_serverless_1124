import os, asyncio, httpx, torch, torch.nn as nn, torch.nn.functional as F, json
from fastapi import FastAPI, Response, Request
from shared import dumps, loads, tensor_to_pack, pack_to_tensor, route_pack
from hotcold import HotColdManager
from scheduler import LGBMScheduler
from utils.logger import log
from scheduler_lgbm import lgb_select_instance, select_instance_generic
from scheduler_hybrid import HYBRID_SCHED
from scheduler_nn import NN_SCHED  # 目前没用到，只是保留接口

device = 'cuda' if torch.cuda.is_available() else 'cpu'

TOP_K = int(os.getenv('TOP_K', '2'))               # MoE expert top-k
HC = HotColdManager()
SCHED = LGBMScheduler()
EXPERT_INSTANCE_TABLE = json.loads(os.getenv('EXP_INSTANCES_JSON', '{}'))

class PreModel(nn.Module):
    def __init__(self, vocab, dim, L, nE):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [nn.TransformerEncoderLayer(dim, 8, 4 * dim) for _ in range(L)]
        )
        self.router = nn.Linear(dim, nE)

    def forward_until_router(self, x_ids):
        x = self.embed(x_ids)          # [B, T, D]
        x = x.transpose(0, 1)          # [T, B, D]
        for b in self.blocks:
            x = b(x)
        x = x.transpose(0, 1)          # [B, T, D]
        logits = self.router(x)        # [B, T, nE]
        return x, logits


pre = None
optim = None


def init_model():
    global pre, optim
    vocab = int(os.getenv('VOCAB_SIZE', '2000'))
    dim = int(os.getenv('EMB_DIM', '256'))
    L = int(os.getenv('N_LAYERS_PRE', '2'))
    nE = max(1, len(EXPERT_INSTANCE_TABLE))
    pre = PreModel(vocab, dim, L, nE).to(device)
    optim = torch.optim.AdamW(pre.parameters(), lr=float(os.getenv('LR', '1e-3')))


app = FastAPI(on_startup=[init_model])

# ----------------- Expert 实例配置 & 调度开关 -----------------

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
USE_NN_SCHED = os.getenv("USE_NN_SCHED", "0") == "1"   # 现在没用到
USE_HYBRID = os.getenv("USE_HYBRID", "1") == "1"
INSTANCE_TOP_K = int(os.getenv("INSTANCE_TOP_K", "2"))  # ★ 实例层面的 top-k


# ----------------- 调 Expert（支持实例 top-k） -----------------

async def call_experts(xt, idx, gate, emb_dim: int):
    """
    xt:   [N, D] 展平后的 token 表示（B*T 展平后的 N 个 token）
    idx:  [N, K] 每个 token 的专家编号（MoE 层的 top-k）
    gate: [N, K] 每个专家的 gate 权重
    """
    import time

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
            inst_list = EXPERT_INSTANCES.get(str(eid), [])
            if not inst_list:
                log("pre-fn", f"No instances for expert {eid}, skip")
                continue

            # 该 expert 负责的所有 token 下标
            token_indices_all = [t for (t, k, g) in items]

            # 基础请求特征（总负载），给调度器用
            base_req = {
                "tokens": len(token_indices_all),
                "emb_dim": emb_dim,
            }

            # ===== 选择实例：支持实例 top-k =====
            if USE_HYBRID and len(inst_list) > 1 and INSTANCE_TOP_K > 1:
                # Hybrid + 实例 top-k
                chosen_insts, scores = HYBRID_SCHED.select_instances(
                    func_type="expert",
                    logical_id=eid,
                    instances=inst_list,
                    req=base_req,
                    top_k=min(INSTANCE_TOP_K, len(inst_list)),
                )
            elif USE_HYBRID and len(inst_list) > 1:
                # Hybrid 但只选 1 个
                inst, score = HYBRID_SCHED.select_instance(
                    func_type="expert",
                    logical_id=eid,
                    instances=inst_list,
                    req=base_req,
                )
                chosen_insts, scores = [inst], [score]
            elif USE_LGBM and len(inst_list) > 1:
                # 纯 LGBM（选 1 个）
                inst, score = lgb_select_instance(eid, inst_list, base_req)
                chosen_insts, scores = [inst], [score]
            else:
                # 简单 fallback：就用第一个实例
                chosen_insts, scores = [inst_list[0]], [0.0]

            M = len(chosen_insts)  # 实际选到的实例数量

            # 按实例数量，把 token roughly 均匀切成 M 份
            # 例如 token_indices_all = [0,1,2,3,4,5], M=2 → [0,2,4], [1,3,5]
            chunks = [[] for _ in range(M)]
            for i, tid in enumerate(token_indices_all):
                chunks[i % M].append(tid)

            for j, (inst, sc) in enumerate(zip(chosen_insts, scores)):
                sub_token_indices = chunks[j]
                if not sub_token_indices:
                    continue

                sub_x = xt[sub_token_indices]

                # 对于这个子 batch 的请求特征（用于在线更新）
                sub_req = {
                    "tokens": len(sub_token_indices),
                    "emb_dim": emb_dim,
                }

                url = inst.get("url")
                log(
                    "pre-fn",
                    f"expert={eid} send {len(sub_token_indices)} tokens to inst={inst.get('id')} "
                    f"url={url}, score={sc:.4f}, top_k_inst={M}",
                )

                async def _call_one(
                    sub_x=sub_x,
                    sub_token_indices=sub_token_indices,
                    eid=eid,
                    url=url,
                    inst=inst,
                    sub_req=sub_req,
                ):
                    from shared import tensor_to_pack, pack_to_tensor, dumps, loads

                    t0 = time.time()
                    r = await client.post(
                        url + "/fwd",
                        content=dumps({"x": tensor_to_pack(sub_x)}),
                        headers={"Content-Type": "application/msgpack"},
                    )
                    latency_ms = (time.time() - t0) * 1000.0

                    resp = loads(r.content)
                    y_sub = pack_to_tensor(resp["y"], xt.device)
                    out[sub_token_indices] = y_sub

                    # Hybrid 在线更新（expert）
                    if USE_HYBRID:
                        try:
                            HYBRID_SCHED.online_update(
                                func_type="expert",
                                logical_id=eid,
                                inst=inst,
                                req=sub_req,
                                latency_ms=latency_ms,
                            )
                        except Exception as e:
                            log("pre-fn", f"HYBRID_SCHED update error (expert): {e}")

                tasks.append(_call_one())

        if tasks:
            await asyncio.gather(*tasks)

    return out


# ----------------- FastAPI 路由 -----------------

@app.post('/fwd')
async def fwd(req: Request):
    """
    前向：
      1) PreModel 做 embedding + Transformer + router
      2) MoE top-k 选 expert
      3) call_experts 再在每个 expert 内部做实例 top-k 调度
    """
    pay = loads(await req.body())
    x_ids = pack_to_tensor(pay['x_ids'], device).long()

    pre.train()
    x, logits = pre.forward_until_router(x_ids)  # x: [B, T, D], logits: [B, T, nE]

    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=min(TOP_K, probs.size(-1)), dim=-1)
    idx = topk.indices.reshape(-1, topk.indices.size(-1))   # [N, K]
    gate = topk.values.reshape(-1, topk.values.size(-1))    # [N, K]

    # 记录 router 选择的专家，用于热/冷统计
    HC.observe_batch(idx.view(-1).tolist())

    xt = x.reshape(-1, x.size(-1))                          # [N, D]
    yt = await call_experts(xt, idx, gate, emb_dim=x.size(-1))
    y = yt.reshape(x.shape)                                 # [B, T, D]

    return Response(
        content=dumps({
            'y': tensor_to_pack(y),
            'route': route_pack(idx, gate),
            'micro_id': pay.get('micro_id', 0),
        }),
        media_type='application/msgpack',
    )


@app.post('/bwd')
async def bwd(req: Request):
    """
    反向：目前只对 PreModel 做 backward（不含 expert 内部的梯度传递）
    """
    p = loads(await req.body())
    x_ids = pack_to_tensor(p['x_ids'], device)
    dy = pack_to_tensor(p['dy'], device)

    x, logits = pre.forward_until_router(x_ids)
    y = x
    y.backward(dy)

    return Response(content=dumps({'ok': True}), media_type='application/msgpack')

@app.post('/step')
def step():
    """
    优化器 step（仅 PreModel）
    """
    torch.nn.utils.clip_grad_norm_(pre.parameters(), float(os.getenv('CLIP', '1.0')))
    optim.step()
    optim.zero_grad(set_to_none=True)
    return {'ok': True}
