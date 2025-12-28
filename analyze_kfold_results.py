import re
import numpy as np

"""
读取 result/model_test.txt，自动计算多次/多折测试结果的 mean ± std。

默认假设每一行格式类似：
    epoch= 5, accuracy= 0.725126, rmse= 0.441644, auc= 0.750386
"""


LOG_PATH = "result/model_test.txt"


def parse_line(line: str):
    """
    从一行文本中解析 accuracy / rmse / auc
    返回 (acc, rmse, auc) 或 None
    """
    # 使用正则提取浮点数
    pattern = (
        r"accuracy=\s*([0-9.]+)\s*,\s*rmse=\s*([0-9.]+)\s*,\s*auc=\s*([0-9.]+)"
    )
    m = re.search(pattern, line)
    if not m:
        return None
    acc = float(m.group(1))
    rmse = float(m.group(2))
    auc = float(m.group(3))
    return acc, rmse, auc


def analyze(path: str = LOG_PATH):
    acc_list = []
    rmse_list = []
    auc_list = []

    with open(path, encoding="utf8") as f:
        for line in f:
            parsed = parse_line(line)
            if parsed is None:
                continue
            acc, rmse, auc = parsed
            acc_list.append(acc)
            rmse_list.append(rmse)
            auc_list.append(auc)

    if not acc_list:
        print(f"[WARN] No valid lines found in {path}")
        return

    acc_arr = np.array(acc_list)
    rmse_arr = np.array(rmse_list)
    auc_arr = np.array(auc_list)

    def stats(arr):
        return arr.mean(), arr.std()

    acc_mean, acc_std = stats(acc_arr)
    rmse_mean, rmse_std = stats(rmse_arr)
    auc_mean, auc_std = stats(auc_arr)

    print(f"Read {len(acc_list)} runs from {path}")
    print(f"Accuracy: mean={acc_mean:.6f}, std={acc_std:.6f}")
    print(f"RMSE    : mean={rmse_mean:.66f}, std={rmse_std:.6f}")
    print(f"AUC     : mean={auc_mean:.6f}, std={auc_std:.6f}")


if __name__ == "__main__":
    analyze()


