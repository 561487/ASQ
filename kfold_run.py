import os
import shutil
import argparse
from typing import List, Tuple


"""
五折交叉验证运行脚本。

依赖：
    - 先运行 kfold_divide_data.py 生成 data/kfold/fold0..4 下的 train/val/test json
    - 使用现有的 train.py 和 predict.py 进行训练与测试

用法示例：
    # 当前修改骨干 + 静态 Q 五折
    python kfold_run.py cuda:0 5

    # DynamicQ 五折
    python kfold_run.py cuda:0 5 --use_dynamic_q --d_model 128

    # DynamicQ + 专家 Q 先验融合 五折
    python kfold_run.py cuda:0 5 --use_dynamic_q --d_model 128 --use_q_init

说明：
    - 本脚本会在每一折：
        1) 覆盖 data/train_set.json, data/val_set.json, data/test_set.json
           为当前折的对应文件
        2) 调用一次 train.py
        3) 调用一次 predict.py
    - 所有测试结果会依次追加写入 result/model_test.txt
      建议在运行前手动清空该文件，避免旧结果干扰。
"""


def _copy_fold_data(fold_id: int) -> None:
    """将 data/kfold/fold{fold_id} 下的 json 复制到 data/ 根目录"""
    fold_dir = os.path.join("data", "kfold", f"fold{fold_id}")
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    src_train = os.path.join(fold_dir, "train_set.json")
    src_val = os.path.join(fold_dir, "val_set.json")
    src_test = os.path.join(fold_dir, "test_set.json")

    dst_train = os.path.join("data", "train_set.json")
    dst_val = os.path.join("data", "val_set.json")
    dst_test = os.path.join("data", "test_set.json")

    shutil.copyfile(src_train, dst_train)
    shutil.copyfile(src_val, dst_val)
    shutil.copyfile(src_test, dst_test)


def _build_train_cmd(
    device: str,
    epoch: int,
    use_dynamic_q: bool,
    d_model: int,
    use_q_init: bool,
    patience: int = 0,
) -> str:
    cmd = f"python train.py {device} {epoch}"
    if use_dynamic_q:
        cmd += f" --use_dynamic_q --d_model {d_model}"
        if use_q_init:
            cmd += " --use_q_init"
    if patience > 0:
        cmd += f" --patience {patience}"
    return cmd


def _build_predict_cmd(
    epoch: int,
    use_dynamic_q: bool,
    d_model: int,
    use_q_init: bool,
    use_best: bool = True,  # 默认使用最佳 epoch
) -> str:
    if use_best:
        cmd = "python predict.py --use_best"
    else:
        cmd = f"python predict.py {epoch}"
    if use_dynamic_q:
        cmd += f" --use_dynamic_q --d_model {d_model}"
        if use_q_init:
            cmd += " --use_q_init"
    return cmd


def run_kfold(
    device: str,
    epoch: int,
    n_folds: int = 5,
    use_dynamic_q: bool = False,
    d_model: int = 128,
    use_q_init: bool = False,
    patience: int = 0,
) -> None:
    print(
        f"[K-FOLD] device={device}, epoch={epoch}, n_folds={n_folds}, "
        f"use_dynamic_q={use_dynamic_q}, d_model={d_model}, use_q_init={use_q_init}"
    )

    # 提示清理旧的测试结果
    test_log = os.path.join("result", "model_test.txt")
    if os.path.exists(test_log):
        print(f"[WARN] {test_log} 已存在，建议在运行前手动清空以避免旧结果干扰。")

    for fold in range(n_folds):
        print(f"\n================ Fold {fold} ================")

        # 1) 覆盖当前折的数据到 data/
        _copy_fold_data(fold)
        print(f"[INFO] Copied data for fold {fold} into data/train/val/test_set.json")

        # 2) 训练
        train_cmd = _build_train_cmd(device, epoch, use_dynamic_q, d_model, use_q_init, patience)
        print(f"[RUN] {train_cmd}")
        ret = os.system(train_cmd)
        if ret != 0:
            print(f"[ERROR] Train command failed on fold {fold}, exit code={ret}")
            break

        # 3) 测试（使用验证集上最好的 epoch）
        predict_cmd = _build_predict_cmd(epoch, use_dynamic_q, d_model, use_q_init, use_best=True)
        print(f"[RUN] {predict_cmd}")
        ret = os.system(predict_cmd)
        if ret != 0:
            print(f"[ERROR] Predict command failed on fold {fold}, exit code={ret}")
            break

    print("\n[K-FOLD] Finished. Please check result/model_test.txt for per-fold metrics.")


def main():
    parser = argparse.ArgumentParser(description="Run K-fold cross-validation for NCDM / DynamicQ")
    parser.add_argument("device", type=str, help='Device string, e.g. "cpu" or "cuda:0"')
    parser.add_argument("epoch", type=int, help="Number of training epochs per fold")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument(
        "--use_dynamic_q",
        action="store_true",
        help="Use DynamicQ (sparse Q) instead of static expert Q",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Embedding dimension for DynamicQ (default: 128)",
    )
    parser.add_argument(
        "--use_q_init",
        action="store_true",
        help="Fuse expert Q prior on every forward pass",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (default: 0 = no early stopping)",
    )

    args = parser.parse_args()

    if (args.device != "cpu") and ("cuda:" not in args.device):
        print('Error: device must be "cpu" or "cuda:X"')
        raise SystemExit(1)

    run_kfold(
        device=args.device,
        epoch=args.epoch,
        n_folds=args.n_folds,
        use_dynamic_q=args.use_dynamic_q,
        d_model=args.d_model,
        use_q_init=args.use_q_init,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()


