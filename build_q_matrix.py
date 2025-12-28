"""
从训练数据构建专家 Q 矩阵的工具函数
"""
import json
import numpy as np
import torch


def build_expert_q_matrix(data_file: str, exer_n: int, knowledge_n: int) -> torch.Tensor:
    """
    从训练数据构建专家 Q 矩阵
    
    参数:
        data_file: 训练数据文件路径（JSON 格式）
        exer_n: 题目数量
        knowledge_n: 知识点数量
    
    返回:
        Q_matrix: torch.Tensor, shape [exer_n, knowledge_n]，值为 0/1
    """
    Q_matrix = np.zeros((exer_n, knowledge_n), dtype=np.float32)
    
    with open(data_file, encoding='utf8') as f:
        data = json.load(f)
    
    # 统计每个题目对应的知识点
    for log in data:
        exer_id = log['exer_id'] - 1  # 转换为 0-based 索引
        for knowledge_code in log['knowledge_code']:
            knowledge_idx = knowledge_code - 1  # 转换为 0-based 索引
            if 0 <= exer_id < exer_n and 0 <= knowledge_idx < knowledge_n:
                Q_matrix[exer_id, knowledge_idx] = 1.0
    
    # 行归一化（保证每题技能权重和为1）
    row_sums = Q_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # 避免除零
    Q_matrix = Q_matrix / row_sums
    
    return torch.from_numpy(Q_matrix)


if __name__ == '__main__':
    # 测试：从训练数据构建 Q 矩阵
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: python build_q_matrix.py <data_file> <exer_n> <knowledge_n>")
        print("Example: python build_q_matrix.py data/train_set.json 17746 123")
        exit(1)
    
    data_file = sys.argv[1]
    exer_n = int(sys.argv[2])
    knowledge_n = int(sys.argv[3])
    
    Q_matrix = build_expert_q_matrix(data_file, exer_n, knowledge_n)
    print(f"Q matrix shape: {Q_matrix.shape}")
    print(f"Q matrix dtype: {Q_matrix.dtype}")
    print(f"Non-zero entries: {(Q_matrix > 0).sum().item()}")
    print(f"First 5 rows:\n{Q_matrix[:5]}")



