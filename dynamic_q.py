import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def sparsemax(logits: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    z_sorted, _ = torch.sort(logits, descending=True, dim=dim)
    z_cumsum = torch.cumsum(z_sorted, dim)
    k = torch.arange(1, logits.size(dim) + 1, device=logits.device)
    support = 1 + k * z_sorted > z_cumsum
    k_z = support.sum(dim=dim, keepdim=True)
    tau = (z_cumsum.gather(dim, k_z - 1) - 1) / k_z
    p = torch.clamp(logits - tau, min=0)
    return p / (p.sum(dim=dim, keepdim=True) + eps)


class DynamicQ(nn.Module):
    def __init__(self, d_model: int = 128, num_skills: int = 50,
                 sparse_type: str = "entmax15", tau: float = 0.7):
        super().__init__()
        self.d_model = d_model
        self.num_skills = num_skills
        self.sparse_type = sparse_type.lower()
        self.tau = tau

        # 题目与技能线性映射
        self.item_proj = nn.Linear(d_model, d_model)
        self.skill_proj = nn.Linear(d_model, d_model)

    def forward(self, item_embed: torch.Tensor, skill_embed: torch.Tensor,
                q_init: Optional[torch.Tensor] = None) -> torch.Tensor:
        # (1) 计算题目与技能间的相似度
        i_proj = self.item_proj(item_embed)       # [I, d_model]
        s_proj = self.skill_proj(skill_embed)     # [K, d_model]
        logits = torch.matmul(i_proj, s_proj.T) / self.tau  # [I, K]

        # (2) 稀疏注意力
        if self.sparse_type == "sparsemax":
            Q_star = sparsemax(logits, dim=-1)
        else:
            # 默认使用 entmax15
            try:
                from entmax import entmax15
                Q_star = entmax15(logits, dim=-1)
            except ImportError:
                Q_star = F.softmax(logits, dim=-1)  # 退化为 softmax
                print("⚠️ 未安装 entmax 库, 使用 softmax 近似。")

        # (3) 若提供 q_init，则向其靠拢
        if q_init is not None:
            Q_star = 0.9 * Q_star + 0.1 * q_init.to(Q_star.device)

        # (4) 行归一化 (保证每题技能权重和为1)
        Q_star = Q_star / (Q_star.sum(dim=-1, keepdim=True) + 1e-8)

        return Q_star



