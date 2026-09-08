# DynamicQ：NCDM 中的可学习 Q 矩阵

本文件以当前代码为准。数据准备与基础命令见 [README](README.md)，实验限制见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 方法定义

研究对象是 Q 矩阵生成方法，NCDM 是下游诊断模型。`DynamicQ` 是代码模块名。输出是非负连续权重矩阵，不是传统二值 Q。它随训练更新，但同一模型状态下同一道题的 Q 行对所有学生相同，不属于学生个性化 Q 或时序 Q。

## 计算流程

设题目数 I、知识点数 K、嵌入维度 d。

1. `model.py` 定义可学习题目嵌入 E（I × d）和知识点嵌入 C（K × d）。
2. 分别做带偏置的线性投影，计算 `S = item_proj(E) @ skill_proj(C).T / tau`，其中 `tau=0.7`。
3. 沿知识点维度映射得到学习 Q。默认使用 entmax15；模块支持 sparsemax，但命令行没有对应参数。未安装 entmax 时默认分支回退为 softmax。
4. 启用专家先验时计算 `Q_mix = 0.9 * Q_learned + 0.1 * Q_expert`，否则使用学习 Q；随后按行归一化得到 `Q_star`。
5. 取当前题目行，经 `KnowledgeAttention` 得到 `q_effective = q * softmax(attn_layer(q))`。这一步之后不再归一化，实际参与诊断交互的向量不保证行和为 1。
6. 构造 `x = e_disc * (stu_emb - k_diff) * q_effective`，经预测网络输出答对概率，使用答题标签的二元交叉熵联合训练。

当前无额外 Q 重构损失、专家对齐损失或显式稀疏正则项。稀疏性来自映射，需要实际测量；使用稀疏映射不意味着每行必然出现零值，softmax 回退也不能当作稀疏实验。

## 专家先验不是参数初始化

`build_expert_q_matrix()` 从当前训练集的 `knowledge_code` 合并同题标注并按行归一化。训练中未出现的题目对应全零行。

`--use_q_init` 保留原参数名，准确含义是**每次前向计算持续融合专家 Q 先验**。专家 Q 作为 buffer 随模型保存，不用于初始化题目或知识点嵌入。融合之后还会归一化，全零专家行不能解释为最终固定占比的 10% 专家贡献。

这里“专家 Q”指数据集知识点标注构造的矩阵，代码没有额外专家标注流程。无先验模式仍使用配置中的知识点数量；预测指标本身不能证明学到的各列与原知识点语义对齐。

## 参数与命令

| 参数 | 实际含义 |
| --- | --- |
| `device` | 训练设备；预测固定使用 CPU |
| `epoch` | 训练最大轮数或预测指定检查点轮次 |
| `--use_dynamic_q` | 启用学习 Q |
| `--d_model` | 嵌入维度，默认 128 |
| `--use_q_init` | 动态模式下持续融合专家先验 |
| `--patience` | 训练命令默认 0，不早停；正数按验证 AUC 早停 |
| `--use_best` | 预测时加载验证 AUC 最优的检查点 |

先按 README 准备独立数据划分和输出目录，再选择一个配置运行：

```bash
# 当前修改骨干 + 静态 Q
python train.py cpu 70 --patience 5
python predict.py --use_best
```

```bash
# 当前修改骨干 + 学习 Q
python train.py cpu 70 --use_dynamic_q --d_model 128 --patience 5
python predict.py --use_best --use_dynamic_q --d_model 128
```

```bash
# 当前修改骨干 + 学习 Q + 专家先验
python train.py cpu 70 --use_dynamic_q --d_model 128 --use_q_init --patience 5
python predict.py --use_best --use_dynamic_q --d_model 128 --use_q_init
```

不同配置共用输出路径，切换前需归档检查点与日志。训练和预测的开关、嵌入维度、映射实现和依赖环境应一致。检查点仅保存 `state_dict`，不会自动恢复全部实验设置。

## 诊断骨干与实现边界

静态和动态模式共用知识点注意力、512/256 隐藏层、LayerNorm、GELU、Dropout（0.5）及可学习 sigmoid 温度，均对预测全连接层权重进行非负裁剪。因此静态模式不能称为严格的原版 NCDM 复现。

LayerNorm、GELU 和没有正值约束的输出温度意味着，不能仅凭非负裁剪就声称保留了原版 NCDM 的严格单调性保证。

每次前向计算完整 I × K 的 Q 矩阵后再取批次题目行，大规模实验需关注计算和显存开销。

## 输出与结论

模型保存在 `model/model_epoch{N}`、`model/best_model.pt`，最佳轮次记录于 `model/best_epoch.txt`。指标追加到 `result/model_val.txt` 和 `result/model_test.txt`，日志不记录模式、种子或折号。

当前仓库没有已记录的有效性能结果，本说明不提供示例指标。Q 的结构质量、稀疏度和解释性需要对应的额外验证，不能仅由预测效果推断。
