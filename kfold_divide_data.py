import json
import os
import random
from typing import List, Dict


"""
基于原始 log_data.json 生成五折交叉验证的数据（**日志级 K 折**）：

目标：
    - **测试集中的学生在训练集中都出现过**（即每个学生的一部分做训练、另一部分做测试）
    - 更接近你之前单次划分（train/val/test 都包含同一批学生）的设定

输出目录结构：
    data/kfold/fold0/train_set.json
                       val_set.json
                       test_set.json
    ...
    data/kfold/fold4/...

其中：
    - train_set.json : 扁平日志列表（给 TrainDataLoader 用），来自“除当前折外”的所有日志
    - val_set.json   : 按学生分组的日志（给 ValTestDataLoader('validation') 用），与 test_set 相同
    - test_set.json  : 按学生分组的日志（给 ValTestDataLoader('test') 用），为当前折的日志

划分策略（log-level K-fold with per-student cycling）：
    1. 从 log_data.json 读取所有学生日志，并删除 log_num < MIN_LOG 的学生
    2. 对于每个学生：
         - 将该学生的 logs 随机打乱
         - 按顺序轮流分配到 fold0..fold4（j % N_FOLDS）
       这样可以保证：
         - 每个学生在大多数折里，既出现在 train，也出现在 test
         - 尤其当 log_num >= N_FOLDS 时，几乎一定满足“test 集学生在 train 集都见过”
    3. 第 k 折：
         - 当前折的所有日志构成 test_set_k
         - 其它折的所有日志合并构成 train_set_k
         - val_set_k 直接复用 test_set_k（存在测试数据参与模型选择的问题，见 EXPERIMENTS.md）
"""


MIN_LOG = 15
N_FOLDS = 5
RANDOM_SEED = 2025


def _load_and_filter_log_data(path: str) -> List[Dict]:
    """读取 log_data.json 并过滤掉答题次数过少的学生"""
    with open(path, encoding="utf8") as f:
        stus = json.load(f)

    filtered = [stu for stu in stus if stu.get("log_num", 0) >= MIN_LOG]
    print(f"[INFO] Loaded {len(stus)} students from {path}")
    print(f"[INFO] Kept {len(filtered)} students with log_num >= {MIN_LOG}")
    return filtered


def _assign_logs_to_folds(stus: List[Dict], n_folds: int):
    """
    将每个学生的日志按“轮流分配”的方式分到 K 个折中。
    返回：
        fold_logs: 长度为 n_folds 的列表，每个元素是该折的扁平日志列表：
            [{user_id, exer_id, score, knowledge_code}, ...]
    """
    fold_logs: List[List[Dict]] = [[] for _ in range(n_folds)]

    for stu in stus:
        uid = stu["user_id"]
        logs = list(stu.get("logs", []))
        random.shuffle(logs)
        for j, log in enumerate(logs):
            fold_id = j % n_folds
            fold_logs[fold_id].append(
                {
                    "user_id": uid,
                    "exer_id": log["exer_id"],
                    "score": log["score"],
                    "knowledge_code": log["knowledge_code"],
                }
            )

    for k in range(n_folds):
        print(f"[INFO] Fold {k}: logs={len(fold_logs[k])}")

    return fold_logs


def _group_logs_by_student(flat_logs: List[Dict]) -> List[Dict]:
    """
    根据扁平日志列表构建按学生分组的列表：
    [
        {
            "user_id": ...,
            "log_num": ...,
            "logs": [ {exer_id, score, knowledge_code}, ... ]
        },
        ...
    ]
    """
    by_user = {}
    for log in flat_logs:
        uid = log["user_id"]
        by_user.setdefault(uid, []).append(
            {
                "exer_id": log["exer_id"],
                "score": log["score"],
                "knowledge_code": log["knowledge_code"],
            }
        )

    grouped = []
    for uid, logs in by_user.items():
        grouped.append(
            {
                "user_id": uid,
                "log_num": len(logs),
                "logs": logs,
            }
        )
    return grouped


def build_kfold_data(
    log_path: str = "data/log_data.json",
    out_root: str = "data/kfold",
    n_folds: int = N_FOLDS,
) -> None:
    random.seed(RANDOM_SEED)

    stus = _load_and_filter_log_data(log_path)
    fold_logs = _assign_logs_to_folds(stus, n_folds)

    os.makedirs(out_root, exist_ok=True)

    for fold_id in range(n_folds):
        print(f"\n===== Building Fold {fold_id} =====")

        # 当前折的日志作为 test，其余折的日志作为 train
        test_flat = fold_logs[fold_id]
        train_flat = []
        for k in range(n_folds):
            if k == fold_id:
                continue
            train_flat.extend(fold_logs[k])

        print(
            f"[INFO] Fold {fold_id}: train_logs={len(train_flat)}, "
            f"test_logs={len(test_flat)}"
        )

        # 将 test_flat 按学生分组，得到 test_set / val_set
        test_grouped = _group_logs_by_student(test_flat)
        val_grouped = test_grouped  # 当前验证与测试相同，正式实验前需划分独立验证集

        fold_dir = os.path.join(out_root, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        train_path = os.path.join(fold_dir, "train_set.json")
        val_path = os.path.join(fold_dir, "val_set.json")
        test_path = os.path.join(fold_dir, "test_set.json")

        with open(train_path, "w", encoding="utf8") as f:
            json.dump(train_flat, f, indent=4, ensure_ascii=False)
        with open(val_path, "w", encoding="utf8") as f:
            json.dump(val_grouped, f, indent=4, ensure_ascii=False)
        with open(test_path, "w", encoding="utf8") as f:
            json.dump(test_grouped, f, indent=4, ensure_ascii=False)

        print(
            f"[SAVE] Fold {fold_id}: "
            f"train_logs={len(train_flat)}, val_users={len(val_grouped)}, test_users={len(test_grouped)}"
        )

    print(f"\n[DONE] K-fold data generated under: {out_root}")


if __name__ == "__main__":
    build_kfold_data()


