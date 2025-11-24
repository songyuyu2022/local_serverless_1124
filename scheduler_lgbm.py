# scheduler_lgbm.py
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import lightgbm as lgb

from utils.logger import log

_LGB_MODEL = None


def get_lgb_model():
    """懒加载 LightGBM 模型，避免每次 forward 都重新加载文件。"""
    global _LGB_MODEL
    if _LGB_MODEL is None:
        path = os.getenv("LGB_MODEL_PATH", "lgb_instance_selector.txt")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"LGB model file {path} not found, please train LightGBM and export Booster."
            )
        _LGB_MODEL = lgb.Booster(model_file=path)
        log("lgb-scheduler", f"Loaded LightGBM model from {path}")
    return _LGB_MODEL


def build_instance_feature(
    expert_id: int,
    inst: Dict[str, Any],
    req: Dict[str, Any],
) -> np.ndarray:
    """
    构造实例的特征向量，用于 Feed 给 LightGBM。
    你可以按自己的建模文档扩展这里的特征。
    当前示例特征：
      - expert_id
      - grad_bytes 或 batch_tokens (请求规模)
      - meta.rtt_ms
      - meta.price_cents_s
      - dyn.avg_q_ms
    """
    meta = inst.get("meta", {})
    dyn = inst.get("dyn", {})

    rtt_ms = float(meta.get("rtt_ms", 0.0))
    price = float(meta.get("price_cents_s", 0.0))
    avg_q = float(dyn.get("avg_q_ms", 0.0))
    # 对于前向，我们可以用请求 token 数；对于反向你可以传梯度大小等
    load = float(req.get("tokens", 0.0))  # 前向可填 tokens，反向可填 grad_bytes

    feats = np.array(
        [
            float(expert_id),
            load,
            rtt_ms,
            price,
            avg_q,
        ],
        dtype=np.float32,
    )
    return feats


def lgb_select_instance(
    expert_id: int,
    instances: List[Dict[str, Any]],
    req: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """
    使用 LightGBM 从多个实例中选择一个：
    返回 (被选中的 inst, 预测代价 score)，score 越小越好（假设模型是回归延迟）。
    """
    model = get_lgb_model()
    if not instances:
        raise ValueError(f"No candidate instances for expert {expert_id}")

    feats = np.stack(
        [build_instance_feature(expert_id, inst, req) for inst in instances], axis=0
    )  # [N, F]
    # LightGBM Booster.predict 返回 [N] 或 [N,1]
    scores = model.predict(feats)  # 越小越好（假设是延迟或 cost）
    scores = np.asarray(scores).reshape(-1)

    best_idx = int(np.argmin(scores))
    best_inst = instances[best_idx]
    best_score = float(scores[best_idx])

    log(
        "lgb-scheduler",
        f"expert={expert_id} candidates={len(instances)}, best={best_inst.get('id')} "
        f"score={best_score:.4f}",
    )
    return best_inst, best_score
