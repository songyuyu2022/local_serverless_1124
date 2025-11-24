import os, torch, torch.nn as nn, torch.nn.functional as F
from fastapi import FastAPI, Request, Response
import torch
from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from utils.logger import log

device='cuda' if torch.cuda.is_available() else 'cpu'
EXTRACT=os.getenv('EXPERT_BWD_EXTRACT','1')=='1'
class PostModel(nn.Module):
  def __init__(self,vocab,dim,L):
    super().__init__(); self.blocks=nn.ModuleList([nn.TransformerEncoderLayer(dim,8,4*dim) for _ in range(L)]); self.ln=nn.LayerNorm(dim); self.head=nn.Linear(dim,vocab,bias=False)
  def forward_from_boundary(self,y):
    x=y.transpose(0,1)
    for b in self.blocks: x=b(x)
    x=x.transpose(0,1); x=self.ln(x); return self.head(x)
post=None; optim=None
def init_model():
  global post,optim
  vocab=int(os.getenv('VOCAB_SIZE','2000')); dim=int(os.getenv('EMB_DIM','256')); L=int(os.getenv('N_LAYERS_POST','2'))
  post=PostModel(vocab,dim,L).to(device); optim=torch.optim.AdamW(post.parameters(), lr=float(os.getenv('LR','1e-3')))
app=FastAPI(on_startup=[init_model])

@app.post("/fwd")
async def fwd(req: Request):
    p = loads(await req.body())
    y = pack_to_tensor(p["y"], device)
    targets = pack_to_tensor(p["targets"], device)

    log(
        "post-fn",
        f"Received /fwd, y shape={tuple(y.shape)}, targets shape={tuple(targets.shape)}",
    )

    post.train()
    logits = post.forward_from_boundary(y)  # 假设输出形状和之前一致

    # 展平成 [N, V] 以适配 cross_entropy
    if logits.dim() > 2:
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
    else:
        logits_flat = logits
        targets_flat = targets.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

    # ===== 计算 top-1 / top-5 准确率 =====
    with torch.no_grad():
        probs = torch.softmax(logits_flat, dim=-1)
        pred_top1 = probs.argmax(dim=-1)
        correct_top1 = (pred_top1 == targets_flat).float().mean().item()

        num_classes = probs.size(-1)
        k = min(5, num_classes)
        topk_vals, topk_idx = probs.topk(k, dim=-1)
        correct_topk = (
            (topk_idx == targets_flat.unsqueeze(-1))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )

    metrics = {
        "loss": float(loss.item()),
        "acc_top1": float(correct_top1),
        "acc_top5": float(correct_topk),
    }
    log(
        "post-fn",
        f"fwd: loss={metrics['loss']:.4f}, "
        f"acc1={metrics['acc_top1']:.4f}, acc5={metrics['acc_top5']:.4f}",
    )

    return Response(
        content=dumps(
            {
                "loss": float(loss.item()),
                "metrics": metrics,
                "stash": {
                    "y": tensor_to_pack(y),
                    "targets": tensor_to_pack(targets),
                    "micro_id": p.get("micro_id", 0),
                },
            }
        ),
        media_type="application/msgpack",
    )



@app.post('/bwd')
async def bwd(req: Request):
    p = loads(await req.body())
    st = p['stash']

    # 1. 还原边界张量 & 标签
    y = pack_to_tensor(st['y'], device)
    targets = pack_to_tensor(st['targets'], device)

    log("post-fn", f"Received /bwd, y shape={tuple(y.shape)}, targets shape={tuple(targets.shape)}")

    # 让 y 参与梯度计算（关键！）
    # detach 防止之前 graph 干扰，本次反向只关心当前 step
    y = y.detach().requires_grad_(True)

    # 2. 前向一遍，得到 logits 和 loss
    logits = post.forward_from_boundary(y)

    if logits.dim() > 2:
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
    else:
        logits_flat = logits
        targets_flat = targets.view(-1)

    loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')

    # 3. 反向传播：同时得到参数梯度 + y 的梯度
    loss.backward()
    log("post-fn", f"Backward finished, loss={loss.item():.6f}")

    # 4. 收集 post 模块的参数梯度，作为 expert_grads 返回
    grad_dict = {}
    for name, param in post.named_parameters():
        if param.grad is not None:
            grad_dict[name] = tensor_to_pack(param.grad.detach())
    log("post-fn", f"Collected {len(grad_dict)} parameter gradients")

    # 5. 取 y 的梯度，用于回传给 pre_fn
    if y.grad is None:
        # 理论上在 requires_grad_(True) + loss.backward() 后一定有 grad
        raise RuntimeError("y.grad is None after backward, check requires_grad_ usage")
    dy = y.grad.detach()
    log("post-fn", f"Computed dy for pre-fn, dy shape={tuple(dy.shape)}")

    return Response(
        content=dumps({
            "dy": tensor_to_pack(dy),
            "expert_grads": grad_dict,
            "micro_id": st["micro_id"]
        }),
        media_type="application/msgpack"
    )



@app.post('/step')
def step():
  torch.nn.utils.clip_grad_norm_(post.parameters(), float(os.getenv('CLIP','1.0'))); optim.step(); optim.zero_grad(set_to_none=True); return {'ok':True}
