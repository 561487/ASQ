# DynamicQ（稀疏 Q）模块使用说明

## 概述

本项目已集成 DynamicQ（稀疏 Q）模块，可以用可学习的稀疏 Q 矩阵替换 NCDM 中的静态专家 Q 矩阵。

## 功能说明

### 1. 原始 NCDM（静态专家 Q）
- 使用从数据中提取的静态专家 Q 矩阵
- Q 矩阵是固定的，不参与训练

### 2. NCDM + DynamicQ（动态稀疏 Q）
- 使用 DynamicQ 模块生成可学习的稀疏 Q 矩阵
- Q 矩阵通过题目和技能的嵌入向量动态生成
- 支持使用专家 Q 矩阵进行初始化（可选）

## 使用方法

### 训练模型

#### 1. 原始 NCDM（静态专家 Q）
```bash
python train.py cuda:0 70
```

#### 2. NCDM + DynamicQ（不使用专家 Q 初始化）
```bash
python train.py cuda:0 70 --use_dynamic_q --d_model 128
```

#### 3. NCDM + DynamicQ（使用专家 Q 初始化）
```bash
python train.py cuda:0 70 --use_dynamic_q --d_model 128 --use_q_init
```

**参数说明：**
- `cuda:0`: 设备（可以是 `cpu` 或 `cuda:0`, `cuda:1` 等）
- `70`: 训练轮数
- `--use_dynamic_q`: 启用 DynamicQ 模块
- `--d_model`: DynamicQ 的嵌入维度（默认：128）
- `--use_q_init`: 使用专家 Q 矩阵初始化 DynamicQ

### 测试模型

#### 1. 原始 NCDM
```bash
python predict.py 70
```

#### 2. NCDM + DynamicQ（不使用专家 Q 初始化）
```bash
python predict.py 70 --use_dynamic_q --d_model 128
```

#### 3. NCDM + DynamicQ（使用专家 Q 初始化）
```bash
python predict.py 70 --use_dynamic_q --d_model 128 --use_q_init
```

**重要提示：**
- 测试时的参数（`--use_dynamic_q`, `--d_model`, `--use_q_init`）必须与训练时完全一致！
- `epoch` 参数指定要加载的模型 epoch 编号

## 实验对比

为了验证 DynamicQ 的效果，建议进行以下对比实验：

### 实验 1：原始 NCDM vs NCDM + DynamicQ（无初始化）
```bash
# 基线：原始 NCDM
python train.py cuda:0 70

# 实验：NCDM + DynamicQ
python train.py cuda:0 70 --use_dynamic_q --d_model 128
```

### 实验 2：原始 NCDM vs NCDM + DynamicQ（有初始化）
```bash
# 基线：原始 NCDM
python train.py cuda:0 70

# 实验：NCDM + DynamicQ（使用专家 Q 初始化）
python train.py cuda:0 70 --use_dynamic_q --d_model 128 --use_q_init
```

### 评估结果
训练完成后，查看 `result/model_val.txt` 和 `result/model_test.txt` 文件，对比以下指标：
- **AUC**: 曲线下面积（越高越好）
- **RMSE**: 均方根误差（越低越好）
- **Accuracy**: 准确率（越高越好）

## 技术细节

### DynamicQ 模块
- **稀疏注意力机制**: 使用 `entmax15` 或 `sparsemax` 生成稀疏 Q 矩阵
- **可学习参数**: 
  - `item_embed`: 题目嵌入向量 `[exer_n, d_model]`
  - `skill_embed`: 技能嵌入向量 `[knowledge_n, d_model]`
  - `item_proj` 和 `skill_proj`: 线性投影层

### 专家 Q 矩阵构建
- 从训练数据 `data/train_set.json` 中提取
- 每个题目对应的知识点被标记为 1，其他为 0
- 行归一化后作为初始化（如果使用 `--use_q_init`）

### 模型架构变化
- **原始模式**: `kn_emb` 直接从数据加载器输入（静态专家 Q）
- **DynamicQ 模式**: `kn_emb` 由 DynamicQ 模块动态生成（可学习稀疏 Q）

## 注意事项

1. **参数一致性**: 训练和测试时的 DynamicQ 相关参数必须完全一致
2. **模型保存**: 模型保存在 `model/model_epoch{N}`，确保训练和测试使用相同的 epoch
3. **依赖库**: 如果安装了 `entmax` 库，DynamicQ 会使用 `entmax15`；否则会退化为 `softmax`
4. **内存占用**: DynamicQ 模式会增加模型参数量（主要是 `item_embed` 和 `skill_embed`）

## 文件说明

- `dynamic_q.py`: DynamicQ 模块实现
- `build_q_matrix.py`: 从数据构建专家 Q 矩阵的工具函数
- `model.py`: 修改后的 NCDM 模型（支持 DynamicQ）
- `train.py`: 修改后的训练脚本（支持 DynamicQ）
- `predict.py`: 修改后的预测脚本（支持 DynamicQ）

## 示例输出

训练时会显示：
```
✅ Using DynamicQ (sparse Q) with d_model=128
✅ Using expert Q matrix for initialization
Building expert Q matrix from training data...
Expert Q matrix shape: torch.Size([17746, 123])
training model...
```

测试时会显示：
```
✅ Using DynamicQ (sparse Q) with d_model=128
testing model...
epoch= 70, accuracy= 0.7234, rmse= 0.3456, auc= 0.7890
```



