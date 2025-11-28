# scheduler_lgbm.py
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import lightgbm as lgb

from utils.logger import log

# 全局模型 & 模型文件修改时间
_LGB_MODEL = None
_LGB_MODEL_MTIME = None  # 记录当前已加载模型文件的修改时间
LGB_MODEL_PATH = os.getenv("LGB_MODEL_PATH", "lgb_instance_selector.txt")

# 函数类型编码：pre / expert / post
ROLE_TO_ID = {
    "pre": 0,
    "expert": 1,
    "post": 2,
}


def _load_lgb_model(path: str):
    """
    内部工具：真正执行 LightGBM 模型加载，并记录文件修改时间。
    """
    global _LGB_MODEL, _LGB_MODEL_MTIME
    _LGB_MODEL = lgb.Booster(model_file=path)
    try:
        _LGB_MODEL_MTIME = os.path.getmtime(path)
    except OSError:
        _LGB_MODEL_MTIME = None
    log("lgb-scheduler", f"Loaded LightGBM model from {path}, mtime={_LGB_MODEL_MTIME}")


def get_lgb_model():
    """
    懒加载 + 热重载 LightGBM 模型：

    - 如果从未加载过，且文件存在 → 加载
    - 如果已经加载过，但文件修改时间变大 → 重新加载（热更新）
    - 如果文件不存在 → 返回当前内存中的模型（可能是 None）
    """
    global _LGB_MODEL, _LGB_MODEL_MTIME

    # 模型文件不存在：保持现状（可能是第一次 or 临时没有）
    if not os.path.exists(LGB_MODEL_PATH):
        if _LGB_MODEL is None:
            log(
                "lgb-scheduler",
                f"LGB model file {LGB_MODEL_PATH} not found, "
                f"hybrid scheduler will fallback or only use NN."
            )
        return _LGB_MODEL  # 可能是 None（外层需处理 fallback）

    # 获取当前文件修改时间
    try:
        mtime = os.path.getmtime(LGB_MODEL_PATH)
    except OSError:
        mtime = None

    # 尚未加载过模型 → 首次加载
    if _LGB_MODEL is None:
        _load_lgb_model(LGB_MODEL_PATH)
        return _LGB_MODEL

    # 已加载过模型 & 文件修改时间更新 → 热重载
    if (_LGB_MODEL_MTIME is not None and mtime is not None and mtime > _LGB_MODEL_MTIME):
        log(
            "lgb-scheduler",
            f"Detected newer LGB model file at {LGB_MODEL_PATH}, "
            f"old_mtime={_LGB_MODEL_MTIME}, new_mtime={mtime}, reloading..."
        )
        _load_lgb_model(LGB_MODEL_PATH)

    return _LGB_MODEL


def build_instance_feature_generic(
    func_type: str,
    logical_id: int,
    inst: Dict[str, Any],
    req: Dict[str, Any],
) -> np.ndarray:
    """
    通用特征构造：
      func_type: "pre" / "expert" / "post"
      logical_id: 对于 expert = expert_id；对于 pre/post 可用 0/1 等

    特征向量：
      0: role_id          (0: pre, 1: expert, 2: post)
      1: logical_id
      2: load_tokens      （本次请求 token 数）
      3: emb_dim          （embedding 维度）
      4: rtt_ms           （实例静态 RTT 估计）
      5: price_cents_s    （单价）
      6: avg_q_ms         （动态队列延迟估计）
    """
    meta = inst.get("meta", {})
    dyn = inst.get("dyn", {})

    role_id = float(ROLE_TO_ID.get(func_type, 1))
    lid = float(logical_id)

    tokens = float(req.get("tokens", 0.0))
    emb_dim = float(req.get("emb_dim", 0.0))

    rtt_ms = float(meta.get("rtt_ms", 0.0))
    price = float(meta.get("price_cents_s", 0.0))
    avg_q = float(dyn.get("avg_q_ms", 0.0))

    feats = np.array(
        [
            role_id,
            lid,
            tokens,
            emb_dim,
            rtt_ms,
            price,
            avg_q,
        ],
        dtype=np.float32,
    )
    return feats


# ======= 为了兼容旧接口，保留 expert 专用封装 =======
def build_instance_feature(
    expert_id: int,
    inst: Dict[str, Any],
    req: Dict[str, Any],
) -> np.ndarray:
    """
    旧接口：仅针对 expert 的特征构造（内部调用通用版本）
    """
    return build_instance_feature_generic("expert", expert_id, inst, req)


def predict_cost_generic(
    func_type: str,
    logical_id: int,
    inst: Dict[str, Any],
    req: Dict[str, Any],
) -> float:
    """
    通用 LGBM 预测：
      func_type: "pre" / "expert" / "post"
      logical_id: expert_id 或 0

    返回该实例在本次请求下的 baseline 代价预测（例如延迟 / 综合 cost）。
    """
    model = get_lgb_model()
    if model is None:
        return 0.0

    feats = build_instance_feature_generic(func_type, logical_id, inst, req)[None, :]  # [1, F]
    score = model.predict(feats)
    return float(np.asarray(score).reshape(-1)[0])


def select_instance_generic(
    func_type: str,
    logical_id: int,
    instances: List[Dict[str, Any]],
    req: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """
    通用 LGBM 选择函数实例：
      func_type: "pre" / "expert" / "post"
      logical_id: expert_id 或 0

    返回 (被选中的 inst, 预测代价 score)，score 越小越好。
    """
    if not instances:
        raise ValueError(f"No candidate instances for func_type={func_type}, id={logical_id}")

    model = get_lgb_model()
    if model is None:
        # 退回到简单 min-rtt
        best_inst = min(
            instances,
            key=lambda inst: float(inst.get("meta", {}).get("rtt_ms", 0.0)),
        )
        log(
            "lgb-scheduler",
            f"[no-model] func={func_type} id={logical_id}, fallback to min-rtt "
            f"inst={best_inst.get('id')}",
        )
        return best_inst, 0.0

    feat_mat = np.stack(
        [
            build_instance_feature_generic(func_type, logical_id, inst, req)
            for inst in instances
        ],
        axis=0,
    )  # [N, F]

    scores = model.predict(feat_mat)
    scores = np.asarray(scores).reshape(-1)

    best_idx = int(np.argmin(scores))
    best_inst = instances[best_idx]
    best_score = float(scores[best_idx])

    log(
        "lgb-scheduler",
        f"func={func_type} id={logical_id}, candidates={len(instances)}, "
        f"best={best_inst.get('id')} score={best_score:.3f}",
    )
    return best_inst, best_score


# ======= 旧 expert 接口的轻量封装 =======
def predict_cost(
    expert_id: int,
    inst: Dict[str, Any],
    req: Dict[str, Any],
) -> float:
    """
    旧接口：只针对 expert 的代价预测（兼容已有代码）
    """
    return predict_cost_generic("expert", expert_id, inst, req)


def lgb_select_instance(
    expert_id: int,
    instances: List[Dict[str, Any]],
    req: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    """
    旧接口：只针对 expert 的实例选择（兼容已有代码）
    """
    return select_instance_generic("expert", expert_id, instances, req)
