# scheduler_hybrid.py
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.logger import log
from scheduler_lgbm import (
    get_lgb_model,
    build_instance_feature_generic,
    predict_cost_generic,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


class ResidualNN(nn.Module):
    """
    在线学习 LightGBM 残差的小 MLP：
      输入: [通用特征(7维)] = [role_id, logical_id, tokens, emb_dim, rtt, price, avg_q]
            + LGBM cost (1维) = 总共 8 维
      输出: residual ≈ true_latency - lgb_cost
    """
    def __init__(self, in_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class HybridScheduler:
    """
    通用 Hybrid 调度器：
      score = LGBM_cost(func_type, logical_id, inst, req)
            + NN_residual(func_type, logical_id, inst, req, lgb_cost)
    """

    def __init__(self, lr: float = 1e-3, warmup: int = 100):
        self.in_dim = 8  # 通用特征7 + lgb_cost 1
        self.model = ResidualNN(self.in_dim).to(device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.warmup = warmup     # 前 warmup 次更新，以 LGBM 为主
        self.num_updates = 0

    def build_nn_feature(
        self,
        func_type: str,
        logical_id: int,
        inst: Dict[str, Any],
        req: Dict[str, Any],
        lgb_cost: float,
    ) -> torch.Tensor:
        base_feat = build_instance_feature_generic(func_type, logical_id, inst, req)  # [7]
        feat = np.concatenate([base_feat, np.array([lgb_cost], dtype=np.float32)], axis=0)
        return torch.from_numpy(feat).to(device)  # [8]

    @torch.no_grad()
    def select_instances(
        self,
        func_type: str,
        logical_id: int,
        instances: List[Dict[str, Any]],
        req: Dict[str, Any],
        top_k: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        选多个实例：
          返回 ([inst1, inst2, ...], [score1, score2, ...])，按 score 从小到大排序。

        - func_type: "pre" / "expert" / "post"
        - logical_id: expert_id 或 0
        - top_k: 要选多少个实例（会自动 clip 到实例数量上限）
        """
        if not instances:
            raise ValueError(f"No instances for func_type={func_type}, id={logical_id}")

        top_k = max(1, min(top_k, len(instances)))

        model_lgb = get_lgb_model()
        if model_lgb is None:
            # 没有 baseline，退回到最简单的 min-rtt
            sorted_insts = sorted(
                instances,
                key=lambda inst: float(inst.get("meta", {}).get("rtt_ms", 0.0)),
            )
            chosen = sorted_insts[:top_k]
            scores = [0.0] * len(chosen)
            log(
                "hybrid-scheduler",
                f"[no-LGBM] func={func_type} id={logical_id}, "
                f"fallback to min-rtt top_k={top_k} insts={[i.get('id') for i in chosen]}",
            )
            return chosen, scores

        # LGBM baseline
        lgb_costs = []
        for inst in instances:
            c = predict_cost_generic(func_type, logical_id, inst, req)
            lgb_costs.append(c)
        lgb_costs = np.asarray(lgb_costs, dtype=np.float32)  # [N]

        # warmup 阶段：只用 LGBM 排序取前 top_k
        if self.num_updates < self.warmup:
            order = np.argsort(lgb_costs)
            chosen_idx = order[:top_k]
            chosen_insts = [instances[i] for i in chosen_idx]
            chosen_scores = [float(lgb_costs[i]) for i in chosen_idx]
            log(
                "hybrid-scheduler",
                f"[warmup] func={func_type} id={logical_id}, "
                f"choose by LGBM top_k={top_k}, insts={[i.get('id') for i in chosen_insts]}",
            )
            return chosen_insts, chosen_scores

        # 正常：LGBM + residual NN
        self.model.eval()
        scores = []
        for inst, base_c in zip(instances, lgb_costs):
            feat = self.build_nn_feature(func_type, logical_id, inst, req, base_c)
            x = feat.unsqueeze(0)  # [1, 8]
            residual = self.model(x).squeeze().item()
            total = float(base_c + residual)
            scores.append(total)

        scores = np.asarray(scores, dtype=np.float32)
        order = np.argsort(scores)
        chosen_idx = order[:top_k]
        chosen_insts = [instances[i] for i in chosen_idx]
        chosen_scores = [float(scores[i]) for i in chosen_idx]

        log(
            "hybrid-scheduler",
            f"func={func_type} id={logical_id}, candidates={len(instances)}, "
            f"top_k={top_k}, chosen={[(i.get('id'), chosen_scores[j]) for j, i in enumerate(chosen_insts)]}",
        )
        return chosen_insts, chosen_scores

    @torch.no_grad()
    def select_instance(
        self,
        func_type: str,
        logical_id: int,
        instances: List[Dict[str, Any]],
        req: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], float]:
        """
        兼容旧接口：只选一个实例
        """
        insts, scores = self.select_instances(func_type, logical_id, instances, req, top_k=1)
        return insts[0], scores[0]

    def online_update(
        self,
        func_type: str,
        logical_id: int,
        inst: Dict[str, Any],
        req: Dict[str, Any],
        latency_ms: float,
    ):
        """
        在线更新 residual NN：
          target_residual = true_latency - lgb_cost
        """
        model_lgb = get_lgb_model()
        if model_lgb is None:
            return

        lgb_cost = predict_cost_generic(func_type, logical_id, inst, req)
        target_residual = float(latency_ms) - float(lgb_cost)

        self.model.train()
        feat = self.build_nn_feature(func_type, logical_id, inst, req, lgb_cost)
        x = feat.unsqueeze(0)  # [1, 8]
        y = torch.tensor([[target_residual]], dtype=torch.float32, device=device)

        self.opt.zero_grad()
        pred = self.model(x)
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        self.opt.step()

        self.num_updates += 1
        if self.num_updates % 50 == 0:
            log(
                "hybrid-scheduler",
                f"updates={self.num_updates}, last_residual_loss={loss.item():.4f}",
            )


HYBRID_SCHED = HybridScheduler(
    lr=float(os.getenv("HYBRID_SCHED_LR", "1e-3")),
    warmup=int(os.getenv("HYBRID_SCHED_WARMUP", "100")),
)
