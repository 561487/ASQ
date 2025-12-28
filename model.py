import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_q import DynamicQ
from typing import Optional


class KnowledgeAttention(nn.Module):
    """
    注意力机制：对知识嵌入向量 kn_emb 进行加权
    输入： [batch_size, knowledge_dim]
    输出： [batch_size, knowledge_dim]
    """
    def __init__(self, knowledge_dim):
        super(KnowledgeAttention, self).__init__()
        self.attn_layer = nn.Sequential(
            nn.Linear(knowledge_dim, knowledge_dim),
            nn.Tanh(),
            nn.Linear(knowledge_dim, knowledge_dim)
        )

    def forward(self, kn_emb):  # [B, K]
        attn_scores = self.attn_layer(kn_emb)               # [B, K]
        attn_weights = torch.softmax(attn_scores, dim=1)    # 按知识维度加权
        weighted_kn_emb = kn_emb * attn_weights
        return weighted_kn_emb


class Net(nn.Module):
    """
    优化后的 NeuralCDM 模型（注意力 + LayerNorm + GELU + Dropout + 温度 sigmoid + 非负裁剪）
    支持 DynamicQ（稀疏 Q）模块替换静态专家 Q 矩阵
    """
    def __init__(self, student_n, exer_n, knowledge_n, 
                 use_dynamic_q: bool = False,
                 d_model: int = 128,
                 q_init: Optional[torch.Tensor] = None,
                 sparse_type: str = "entmax15"):
        super(Net, self).__init__()

        self.knowledge_dim = knowledge_n
        self.emb_num = student_n
        self.exer_n = exer_n
        self.use_dynamic_q = use_dynamic_q
        self.d_model = d_model

        self.prednet_len1, self.prednet_len2 = 512, 256

        # 嵌入层
        self.student_emb = nn.Embedding(self.emb_num, knowledge_n)
        self.k_difficulty = nn.Embedding(self.exer_n, knowledge_n)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        # 初始化 student_emb（xavier_normal）
        nn.init.xavier_normal_(self.student_emb.weight)

        # DynamicQ 相关组件（仅在 use_dynamic_q=True 时使用）
        if self.use_dynamic_q:
            # 题目和技能的嵌入向量（可学习）
            self.item_embed = nn.Parameter(torch.randn(exer_n, d_model) * 0.02)
            self.skill_embed = nn.Parameter(torch.randn(knowledge_n, d_model) * 0.02)
            
            # DynamicQ 模块
            self.dynamic_q = DynamicQ(
                d_model=d_model, 
                num_skills=knowledge_n,
                sparse_type=sparse_type
            )
            
            # 专家 Q 矩阵（可选，用于初始化）
            if q_init is not None:
                self.register_buffer('q_init', q_init.float())
            else:
                self.register_buffer('q_init', None)
        else:
            # 传统模式：使用静态专家 Q（从 kn_emb 输入）
            self.item_embed = None
            self.skill_embed = None
            self.dynamic_q = None
            self.q_init = None

        # 注意力机制（仅在非 DynamicQ 模式下使用，或作为后处理）
        self.kn_attention = KnowledgeAttention(knowledge_n)

        # 前馈神经网络
        self.prednet_full1 = nn.Linear(knowledge_n, self.prednet_len1)
        self.bn1 = nn.LayerNorm(self.prednet_len1)

        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.bn2 = nn.LayerNorm(self.prednet_len2)

        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)
        self.drop = nn.Dropout(p=0.5)

        # 可学习的 sigmoid 温度参数
        self.sigmoid_temp = nn.Parameter(torch.tensor(1.0))

        # 参数初始化（除 student_emb 外）
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2 and 'student_emb' not in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name and param is not None:
                nn.init.constant_(param, 0.01)

    def forward(self, stu_id, exer_id, kn_emb):
        # 获取学生和试题嵌入
        stu_emb = torch.sigmoid(self.student_emb(stu_id))               # [B, K]
        k_diff = torch.sigmoid(self.k_difficulty(exer_id))              # [B, K]
        e_disc = torch.sigmoid(self.e_discrimination(exer_id)) * 10     # [B, 1]

        # Q 矩阵处理：使用 DynamicQ 或静态专家 Q
        if self.use_dynamic_q:
            # 使用 DynamicQ 生成稀疏 Q*
            Q_star = self.dynamic_q(
                self.item_embed, 
                self.skill_embed, 
                q_init=self.q_init
            )  # [exer_n, knowledge_n]
            
            # 根据 exer_id 取对应的 Q* 行
            kn_emb = Q_star[exer_id]  # [B, K]
            
            # 仍然使用注意力机制作为后处理（可选）
            kn_emb = self.kn_attention(kn_emb)  # [B, K]
        else:
            # 传统模式：使用静态专家 Q（从 kn_emb 输入）
            kn_emb = self.kn_attention(kn_emb)  # [B, K]

        # 认知诊断表示构建
        x = e_disc * (stu_emb - k_diff) * kn_emb                        # [B, K]

        # 前馈神经网络
        x = F.gelu(self.bn1(self.prednet_full1(x)))
        x = self.drop(x)

        x = F.gelu(self.bn2(self.prednet_full2(x)))
        x = self.drop(x)

        out = torch.sigmoid(self.prednet_full3(x) / self.sigmoid_temp).clamp(1e-4, 1 - 1e-4)
        return out

    def apply_clipper(self):
        clipper = NoneNegClipper(min_val=0.0)
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        return torch.sigmoid(self.student_emb(stu_id)).data

    def get_exer_params(self, exer_id):
        k_diff = torch.sigmoid(self.k_difficulty(exer_id))
        e_disc = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_diff.data, e_disc.data


class NoneNegClipper(object):
    def __init__(self, min_val=0.0):
        super(NoneNegClipper, self).__init__()
        self.min_val = min_val

    def __call__(self, module):
        if hasattr(module, 'weight'):
            module.weight.data.clamp_(min=self.min_val)
