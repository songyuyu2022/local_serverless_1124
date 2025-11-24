# utils/metrics.py
import csv
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class StepMetrics:
    # 训练身份信息
    epoch: int                 # 第几个 epoch
    step: int                  # 全局 step（从 0 开始计）
    step_in_epoch: int         # 当前 epoch 内的 step 序号
    phase: str                 # "train" or "val"

    # 损失 & 精度
    loss: Optional[float] = None
    acc_top1: Optional[float] = None
    acc_top5: Optional[float] = None

    # 规模相关
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None
    tokens: Optional[int] = None

    # 时间相关（毫秒）
    step_time_ms: Optional[float] = None
    pre_fwd_ms: Optional[float] = None
    post_fwd_ms: Optional[float] = None
    post_bwd_ms: Optional[float] = None
    pre_bwd_ms: Optional[float] = None
    expert_comm_ms: Optional[float] = None

    # 吞吐
    samples_per_s: Optional[float] = None
    tokens_per_s: Optional[float] = None

    # 通信相关
    expert_grad_bytes: Optional[float] = None
    expert_instances: Optional[int] = None


class MetricsLogger:
    """
    简单 CSV 记录器：第一次写入会写 header，后面 append。
    """
    def __init__(self, path: str = "metrics.csv"):
        self.path = path
        self._initialized = os.path.exists(path)

    def log(self, m: StepMetrics):
        row = asdict(m)
        fieldnames = list(row.keys())
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)
