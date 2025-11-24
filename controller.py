# controller.py
import os
import asyncio
import httpx
import torch
import json
import time

from utils.metrics import MetricsLogger, StepMetrics
from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from nsga2_bw import nsga2_select
from utils.logger import log
from dataset import LMTextBatcher, DATA_PATH_DEFAULT

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 基本配置
# ------------------------------------------------------------------
PRE_URL = os.getenv("PRE_URL", "http://127.0.0.1:9000")
POST_URL = os.getenv("POST_URL", "http://127.0.0.1:9001")

USE_NSGA2 = os.getenv("USE_NSGA2", "1") == "1"

# 优先从 JSON 文件读专家实例配置，默认文件名 experts.json
CONFIG_PATH = os.getenv("EXP_INSTANCES_FILE", "experts.json")


def load_expert_instances(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log("controller", f"Loaded expert instances from {path}, keys={list(data.keys())}")
        return data
    except FileNotFoundError:
        log("controller", f"WARNING: config file {path} not found, using empty EXP_INSTANCES")
        return {}
    except Exception as e:
        log("controller", f"ERROR loading {path}: {e}")
        return {}


EXPERT_INSTANCES = load_expert_instances(CONFIG_PATH)

MICRO_BATCHES = int(os.getenv("MICRO_BATCHES", "4"))  # 暂时未用
BLOCK = int(os.getenv("BLOCK_SIZE", "8"))
BATCH = int(os.getenv("BATCH_SIZE", "4"))
STEP_PERIOD_MS = float(os.getenv("STEP_PERIOD_MS", "50"))

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "1"))          # 总 epoch 数
VAL_SPLIT_RATIO = float(os.getenv("VAL_SPLIT_RATIO", "0.9"))  # 训练/验证切分比例

METRICS_PATH = os.getenv("METRICS_PATH", "metrics.csv")
metrics_logger = MetricsLogger(METRICS_PATH)

# ------------------------------------------------------------------
# 数据集：基于 input.txt 的 LM 训练 & 验证
# ------------------------------------------------------------------
DATA_PATH = os.getenv("DATA_PATH", DATA_PATH_DEFAULT)  # 默认 input.txt

train_batcher = LMTextBatcher(
    DATA_PATH, batch_size=BATCH, block_size=BLOCK, device=device,
    split="train", split_ratio=VAL_SPLIT_RATIO
)
val_batcher = LMTextBatcher(
    DATA_PATH, batch_size=BATCH, block_size=BLOCK, device=device,
    split="val", split_ratio=VAL_SPLIT_RATIO
)

# 估算每个 epoch 的 step 数（保证至少 1 步）
def calc_steps_per_epoch(batcher: LMTextBatcher) -> int:
    need = BATCH * (BLOCK + 1)
    return max(1, batcher.data_len // need)

TRAIN_STEPS_PER_EPOCH = calc_steps_per_epoch(train_batcher)
VAL_STEPS_PER_EPOCH = calc_steps_per_epoch(val_batcher)

log(
    "controller",
    f"TRAIN_STEPS_PER_EPOCH={TRAIN_STEPS_PER_EPOCH}, "
    f"VAL_STEPS_PER_EPOCH={VAL_STEPS_PER_EPOCH}"
)

# ------------------------------------------------------------------
# 工具函数：估算梯度大小（用于 NSGA-II 多目标调度）
# ------------------------------------------------------------------
def est_grad_bytes(grads: dict) -> float:
    total = 0.0
    for g in grads.values():
        t = pack_to_tensor(g, "cpu") if isinstance(g, dict) else g
        total += t.numel() * t.element_size()

    topk = float(os.getenv("GRAD_TOPK", "0.0"))
    if topk > 0:
        total *= topk
    if os.getenv("GRAD_FP16", "1") == "1":
        total *= 0.5
    return float(total)


# ------------------------------------------------------------------
# 单次 step：既可以是 train，也可以是 val
# ------------------------------------------------------------------
async def run_step(phase: str, epoch_idx: int, step_in_epoch: int, global_step: int):
    """
    phase: "train" or "val"
    """
    assert phase in ("train", "val")
    log(
        "controller",
        f"=== {phase.upper()} epoch={epoch_idx} step={step_in_epoch} (global_step={global_step}) ==="
    )

    # 选用对应的数据源
    batcher = train_batcher if phase == "train" else val_batcher

    tokens = BATCH * BLOCK
    x_ids, target_ids = batcher.get_batch()
    log(
        "controller",
        f"[{phase}] Loaded batch from dataset, x_ids shape={tuple(x_ids.shape)}, "
        f"targets shape={tuple(target_ids.shape)}, device={x_ids.device}",
    )

    # 时间指标初始化
    t_step_start = time.perf_counter()
    t_pre_fwd_start = t_pre_fwd_end = 0.0
    t_post_fwd_start = t_post_fwd_end = 0.0
    t_post_bwd_start = t_post_bwd_end = 0.0
    t_pre_bwd_start = t_pre_bwd_end = 0.0
    t_expert_comm_start = t_expert_comm_end = 0.0

    loss_val = None
    acc1 = None
    acc5 = None
    grad_bytes = None
    expert_inst_cnt = 0
    expert_comm_ms = 0.0

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            # ------------------ 前向：pre_fn ------------------
            t_pre_fwd_start = time.perf_counter()
            log("controller", f"[{phase}] → Call pre_fn /fwd")
            rpre = await client.post(
                PRE_URL + "/fwd",
                content=dumps({"x_ids": tensor_to_pack(x_ids), "micro_id": global_step}),
                headers={"Content-Type": "application/msgpack"},
            )
            t_pre_fwd_end = time.perf_counter()

            pre_out = loads(rpre.content)
            y = pack_to_tensor(pre_out["y"], device)
            log("controller", f"[{phase}] pre_fn /fwd done, y shape={tuple(y.shape)}")

            # ------------------ 前向：post_fn ------------------
            t_post_fwd_start = time.perf_counter()
            log("controller", f"[{phase}] → Call post_fn /fwd")
            rpost_fwd = await client.post(
                POST_URL + "/fwd",
                content=dumps({
                    "y": tensor_to_pack(y),
                    "targets": tensor_to_pack(target_ids),
                    "micro_id": global_step,
                }),
                headers={"Content-Type": "application/msgpack"},
            )
            t_post_fwd_end = time.perf_counter()

            post_fwd_out = loads(rpost_fwd.content)
            stash = post_fwd_out["stash"]

            # 读取 post_fn 计算的指标
            m = post_fwd_out.get("metrics", {})
            loss_val = m.get("loss", post_fwd_out.get("loss"))
            acc1 = m.get("acc_top1")
            acc5 = m.get("acc_top5")
            log(
                "controller",
                f"[{phase}] post_fn /fwd metrics: loss={loss_val}, acc1={acc1}, acc5={acc5}",
            )

            # ------------------ 仅 train 才做反向和更新 ------------------
            if phase == "train":
                # 反向：post_fn
                t_post_bwd_start = time.perf_counter()
                log("controller", "[train] ← Call post_fn /bwd")
                rb = await client.post(
                    POST_URL + "/bwd",
                    content=dumps({"stash": stash}),
                    headers={"Content-Type": "application/msgpack"},
                )
                t_post_bwd_end = time.perf_counter()

                rb = loads(rb.content)
                dy = pack_to_tensor(rb["dy"], device)
                log("controller", f"[train] post_fn /bwd done, dy shape={tuple(dy.shape)}")

                # 反向：pre_fn
                t_pre_bwd_start = time.perf_counter()
                log("controller", "[train] ← Call pre_fn /bwd")
                await client.post(
                    PRE_URL + "/bwd",
                    content=dumps(
                        {"x_ids": tensor_to_pack(x_ids), "dy": tensor_to_pack(dy)}
                    ),
                    headers={"Content-Type": "application/msgpack"},
                )
                t_pre_bwd_end = time.perf_counter()
                log("controller", "[train] pre_fn /bwd done")

                # 专家梯度调度（NSGA-II）
                if "expert_grads" in rb and USE_NSGA2:
                    grads = rb["expert_grads"]
                    grad_bytes = est_grad_bytes(grads)
                    log(
                        "controller",
                        f"[train] expert_grads present, estimated size={grad_bytes/1e6:.3f} MB",
                    )

                    eid = "0"
                    inst_list = EXPERT_INSTANCES.get(eid, [])
                    expert_inst_cnt = len(inst_list)

                    if not inst_list:
                        log(
                            "controller",
                            f"[train] No expert instances configured for id={eid}, "
                            f"skip NSGA-II scheduling",
                        )
                    else:
                        req = {
                            "grad_bytes": grad_bytes,
                            "price_cents_s": float(
                                os.getenv("DEFAULT_PRICE_CENTS_S", "0.0")
                            ),
                        }
                        log(
                            "controller",
                            f"[train] Run NSGA-II for expert {eid}, instances={len(inst_list)}",
                        )
                        choice = nsga2_select(
                            inst_list,
                            req,
                            STEP_PERIOD_MS,
                            pop_size=8,
                            generations=3,
                            seed=42,
                        )
                        log("controller", f"[train] NSGA-II result={choice}")

                        if choice is not None:
                            inst, mode = choice
                            url = inst.get("url")
                            log(
                                "controller",
                                f"[train] → Send /grad/apply to expert instance "
                                f"id={inst.get('id')} mode={mode} url={url}",
                            )
                            t_expert_comm_start = time.perf_counter()
                            await client.post(
                                url + "/grad/apply",
                                content=dumps({"grads": grads}),
                                headers={"Content-Type": "application/msgpack"},
                            )
                            t_expert_comm_end = time.perf_counter()
                            expert_comm_ms = (t_expert_comm_end - t_expert_comm_start) * 1000.0
                        else:
                            log("controller", "[train] NSGA-II returned None, skip grad/apply")
                else:
                    log("controller", "[train] No expert_grads returned from post_fn, skip experts")

                # step 所有模块
                log("controller", "[train] → Call pre_fn /step")
                await client.post(PRE_URL + "/step")

                log("controller", "[train] → Call post_fn /step")
                await client.post(POST_URL + "/step")

                for eid, inst_list in EXPERT_INSTANCES.items():
                    for inst in inst_list:
                        url = inst.get("url")
                        log("controller", f"[train] → Call expert {eid} /step url={url}")
                        try:
                            await client.post(url + "/step")
                        except Exception as e:
                            log(
                                "controller",
                                f"[train] WARNING: call expert /step failed for {url}: {e}",
                            )

        # step 结束时间与吞吐计算
        t_step_end = time.perf_counter()
        step_time_ms = (t_step_end - t_step_start) * 1000.0

        pre_fwd_ms = (t_pre_fwd_end - t_pre_fwd_start) * 1000.0
        post_fwd_ms = (t_post_fwd_end - t_post_fwd_start) * 1000.0
        post_bwd_ms = (t_post_bwd_end - t_post_bwd_start) * 1000.0 if phase == "train" else 0.0
        pre_bwd_ms = (t_pre_bwd_end - t_pre_bwd_start) * 1000.0 if phase == "train" else 0.0

        samples_per_s = BATCH / (step_time_ms / 1000.0)
        tokens_per_s = tokens / (step_time_ms / 1000.0)

        log(
            "controller",
            f"[{phase}] epoch={epoch_idx} step={step_in_epoch} finished: "
            f"loss={loss_val}, acc1={acc1}, acc5={acc5}, "
            f"step_time_ms={step_time_ms:.2f}, tokens/s={tokens_per_s:.1f}",
        )

        # 记录到 CSV（train / val 都记录）
        metrics_logger.log(
            StepMetrics(
                # use_lgbm: Optional[int] = None,
                # use_nsga2: Optional[int] = None,
                epoch=epoch_idx,
                step=global_step,
                step_in_epoch=step_in_epoch,
                phase=phase,
                loss=loss_val,
                acc_top1=acc1,
                acc_top5=acc5,
                batch_size=BATCH,
                seq_len=BLOCK,
                tokens=tokens,
                step_time_ms=step_time_ms,
                pre_fwd_ms=pre_fwd_ms,
                post_fwd_ms=post_fwd_ms,
                post_bwd_ms=post_bwd_ms,
                pre_bwd_ms=pre_bwd_ms,
                expert_comm_ms=expert_comm_ms,
                samples_per_s=samples_per_s,
                tokens_per_s=tokens_per_s,
                expert_grad_bytes=grad_bytes,
                expert_instances=expert_inst_cnt,
            )
        )

    except Exception as e:
        log("controller", f"ERROR in run_step(phase={phase}, epoch={epoch_idx}, step={step_in_epoch}): {e}")
        raise


# ------------------------------------------------------------------
# 主训练循环：NUM_EPOCHS × (train + val)
# ------------------------------------------------------------------
if __name__ == "__main__":
    log(
        "controller",
        f"Controller started, NUM_EPOCHS={NUM_EPOCHS}, "
        f"PRE_URL={PRE_URL}, POST_URL={POST_URL}, "
        f"BATCH={BATCH}, BLOCK={BLOCK}",
    )

    loop = asyncio.get_event_loop()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        log("controller", f"=== EPOCH {epoch} TRAIN ===")
        for step_in_epoch in range(TRAIN_STEPS_PER_EPOCH):
            loop.run_until_complete(run_step("train", epoch, step_in_epoch, global_step))
            global_step += 1

        log("controller", f"=== EPOCH {epoch} VAL ===")
        for step_in_epoch in range(VAL_STEPS_PER_EPOCH):
            loop.run_until_complete(run_step("val", epoch, step_in_epoch, global_step))
            global_step += 1
