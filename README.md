# Adaptive Sparse Q-Matrix Learning for Cognitive Diagnosis

**ASQ：面向认知诊断的自适应稀疏 Q 矩阵学习**

本项目对应论文 **Adaptive Sparse Q-Matrix Learning for Cognitive Diagnosis**，当前仓库主要提供 ASQ 在 NCDM（NeuralCDM）中的接入代码。ASQ 通过试题与知识点的低维嵌入和线性投影计算连续关联，再经 Entmax₁.₅ 与数值归一化生成自适应稀疏 Q 矩阵，与下游作答预测任务端到端联合优化。代码中的 `DynamicQ` 是 ASQ 核心模块的现有类名。

论文覆盖 ASSIST09、ASSIST17、Junyi 三个数据集及 NCDM、RCD、CDMFKC、KaNCD 四种下游模型；当前仓库仅包含 NCDM 接入及一份 ASSIST09 来源数据，不代表完整论文实验已发布。论文设置与代码差异见 [论文对应说明](PAPER_ALIGNMENT.md)。

本仓库基于 NeuralCD/NCDM 相关代码开展修改。当前诊断骨干还包含注意力、LayerNorm、GELU 等调整；关闭 DynamicQ 得到的是当前修改骨干的静态 Q 模式，不能直接称为原版 NCDM。基础工作引用见文末。

## 方法概览

```text
题目嵌入 + 知识点嵌入
          ↓
线性投影与相似度计算
          ↓
Entmax₁.₅ → 稀疏 Q
          ↓
数值归一化 → 自适应 Q
          ↓
按题目取行 → 知识点注意力加权
          ↓
NCDM 诊断交互与预测网络 → 答对概率
```

论文方法由连续值 Q 计算和稀疏 Q 生成两部分组成，以二元交叉熵联合训练；Q 随训练更新，同一模型状态下所有学生共享同一矩阵。论文最终设置 τ=1，而当前代码默认固定为 0.7，尚无命令行调节入口。`--use_q_init` 是论文主方法之外的专家先验融合扩展，不应在复现 ASQ 主方法时启用。完整细节见 [DYNAMICQ_USAGE.md](DYNAMICQ_USAGE.md)。

## 已实现模式

| 模式 | 开关 | Q 来源 |
| --- | --- | --- |
| 当前修改骨干 + 静态 Q | 不传动态开关 | 数据中知识点标注构造的多热向量 |
| 当前修改骨干 + ASQ 核心模块 | `--use_dynamic_q` | 题目与知识点嵌入生成 |
| 额外扩展：ASQ 核心模块 + 专家先验 | `--use_dynamic_q --use_q_init` | 学习 Q 与训练标注构造的专家 Q 融合 |

三种模式共用当前诊断骨干；仓库尚未提供独立的原版 NCDM 基线实现。实验设计与限制见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 环境

使用 Python、PyTorch、NumPy、scikit-learn；默认稀疏映射另需 `entmax`。`json`、`argparse` 等为标准库。当前未锁定依赖版本，也未验证完整的版本兼容范围，不沿用上游早期 Python/PyTorch 最低版本声明。

```bash
python -m pip install torch numpy scikit-learn entmax
```

CUDA 环境需安装与设备匹配的 PyTorch。未安装 `entmax` 时默认映射回退为 softmax，不能保证精确零值稀疏性。模块支持 `sparsemax`，但训练和预测命令行尚未提供映射类型参数。

## 数据与单次实验

数据沿用上游 ASSIST2009–2010 预处理数据。上游说明的筛选规则为排除开放作答与空技能标注，同一学生重复作答同题仅保留首次；当前仓库不包含从原始 CSV 完整复现这些处理的脚本。

`config.txt` 当前配置为 4163 名学生、17746 道题、123 个知识点。JSON 中 ID 从 1 开始，加载器内部转为从 0 开始。

- `data/log_data.json`：按学生分组，每条作答含 `exer_id`、`score`、`knowledge_code`。
- `data/train_set.json`：扁平作答列表，另含 `user_id`。
- `data/val_set.json`、`data/test_set.json`：按学生分组的作答列表。

**仓库自带的根目录验证集与测试集文件完全相同，不应直接用于独立测试结论。** `divide_data.py` 的实际逻辑会按学生分别生成约 7:1:2 的独立划分，并过滤少于 15 条作答的学生。正式实验应重新生成并固定划分；下列命令会覆盖现有划分文件。该脚本未固定随机种子，各模型应复用同一次生成的划分。

```bash
python divide_data.py
python -c "from pathlib import Path; Path('model').mkdir(exist_ok=True); Path('result').mkdir(exist_ok=True)"
```

以下是当前代码的 NCDM + ASQ 核心模块运行示例（τ 仍为代码默认 0.7，不是论文 τ=1 的完整复现），使用验证 AUC 最优的检查点评估：

```bash
python train.py cpu 70 --use_dynamic_q --d_model 128 --patience 5
python predict.py --use_best --use_dynamic_q --d_model 128
```

训练设备可改成 `cuda:0`，预测当前在 CPU 上执行。两条命令同时去掉 `--use_dynamic_q --d_model 128` 即为静态模式；同时加上 `--use_q_init` 即为专家先验融合模式。

训练命令默认不早停（`--patience 0`）；正数启用基于验证 AUC 的早停。也可用 `python predict.py 70` 加上匹配的模式参数加载指定轮次，但该检查点必须存在。训练与测试配置需一致。

## 输出与当前实验状态

- 训练使用 Adam、学习率 0.002、二元交叉熵，批大小 32；不足完整批次的尾部样本被跳过。
- 检查点为 `model/model_epoch{N}`、`model/best_model.pt` 和 `model/best_epoch.txt`。
- AUC、RMSE、Accuracy 分别追加到 `result/model_val.txt` 和 `result/model_test.txt`。

不同配置会覆盖相同检查点并共用日志，切换实验前应归档结果与配置。当前仓库提交的两个指标文件为空，无法用这些日志核验论文结果。论文表 5-1 中的 NCDM 结果已按稿件整理到 [实验说明](EXPERIMENTS.md)，明确标为论文报告值，而非本仓库重跑结果。

五折脚本目前复用测试集作为验证集，再按验证 AUC 选择模型，存在测试数据参与模型选择的问题。修正为独立验证集之前，不应将其输出作为论文的独立测试结果，详见 [实验说明](EXPERIMENTS.md)。

## 文件导航

| 文件 | 作用 |
| --- | --- |
| `dynamic_q.py` | 学习 Q 生成、稀疏映射、专家先验融合 |
| `build_q_matrix.py` | 从训练标注构造并归一化专家 Q |
| `model.py` | 支持静态/学习 Q 的当前 NCDM 骨干 |
| `data_loader.py` | 数据加载 |
| `divide_data.py` | 单次划分 |
| `train.py` / `predict.py` | 训练、模型选择与测试 |
| `kfold_divide_data.py` / `kfold_run.py` | 存在验证/测试复用问题的五折流程 |
| `analyze_kfold_results.py` | 汇总全部匹配日志行，不自动区分配置或折 |
| [DYNAMICQ_USAGE.md](DYNAMICQ_USAGE.md) | 算法细节和参数 |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 论文 RQ1–RQ4、NCDM 报告结果与复现状态 |
| [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) | 论文公式、设置与代码逐项对应 |
| [认知诊断模型演化总结.md](认知诊断模型演化总结.md) | 项目定位与论文表述边界 |

## 基础工作与引用

以下引用属于基础模型论文，不代表本项目新增 Q 方法发表于这些论文。上游的 AAAI 公式勘误图片保留在 [equation.JPG](equation.JPG)，不作为本项目 Q 方法公式。

```
@article{wang2020neural,
  title={Neural Cognitive Diagnosis for Intelligent Education Systems},
  author={Wang, Fei and Liu, Qi and Chen, Enhong and Huang, Zhenya and Chen, Yuying and Yin, Yu and Huang, Zai and Wang, Shijin},
  booktitle={Thirty-Fourth AAAI Conference on Artificial Intelligence},
  year={2020}
}
```

or

```
@article{wang2022neuralcd,
  title={NeuralCD: A General Framework for Cognitive Diagnosis},
  author={Wang, Fei and Liu, Qi and Chen, Enhong and Huang, Zhenya and Yin, Yu and Wang, Shijin and Su, Yu},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2022},
  publisher={IEEE}
}
```
