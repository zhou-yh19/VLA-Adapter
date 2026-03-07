import torch.nn as nn
import torch



class StageClassifier_0(nn.Module):
    """Stage classifier: softmax[W @ stage^T @ w + b].
    Input: stage (B, 8, llm_dim) = 8 tokens of dim llm_dim.
    Output: logits (B, 4). W: (4, 896), w: (8, 1), b: (4,).
    """
    def __init__(self, llm_dim: int, num_stage_classes: int = 4, num_stage_tokens: int = 8):
        super().__init__()
        self.W = nn.Parameter(torch.empty(num_stage_classes, llm_dim))
        self.w = nn.Parameter(torch.empty(num_stage_tokens, 1))
        self.b = nn.Parameter(torch.empty(num_stage_classes))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)

    def forward(self, stage: torch.Tensor) -> torch.Tensor:
        # stage (B, 8, llm_dim) -> logits (B, 4)
        # (4, llm_dim) @ (B, llm_dim, 8) -> (B, 4, 8); then @ (8, 1) -> (B, 4, 1); + b -> (B, 4)
        x = torch.einsum("ij,bjk->bik", self.W, stage.transpose(1, 2))  # (B, 4, 8)
        logits = (x @ self.w).squeeze(-1) + self.b  # (B, 4)
        return logits



class StageClassifier_1(nn.Module):
    """Stage classifier: Mean Pooling + MLP (Linear -> Activation -> Dropout -> Linear).
    Input: stage (B, 8, llm_dim) = 8 tokens of dim llm_dim.
    Output: logits (B, 4).
    """
    def __init__(self, 
        llm_dim: int = 896, hidden_dim: int = 128, 
        num_stage_classes: int = 4, dropout_rate: float = 0.5
        ):
        super().__init__()
        
        # 1. 第一层线性网络（特征压缩/瓶颈层）
        # 将 896 维的高维特征压缩提炼为 hidden_dim 维的核心特征
        self.fc1 = nn.Linear(llm_dim, hidden_dim)
        
        # 2. 非线性激活函数
        # 推荐使用 GELU (LLM 领域最常用) 或 ReLU，赋予模型非线性表达能力
        self.activation = nn.GELU() 
        
        # 3. Dropout 层（非常关键！）
        # 放在非线性激活之后，第二层线性网络之前，打断死记硬背的神经元连接
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 4. 第二层线性网络（最终分类输出层）
        # 从隐藏层维度映射到 4 个类别
        self.fc2 = nn.Linear(hidden_dim, num_stage_classes)

    def forward(self, stage: torch.Tensor) -> torch.Tensor:
        # 输入维度: (B, 8, llm_dim)
        
        # 第一步：序列维度求平均 (Mean Pooling) -> 破除位置依赖
        # 维度: (B, 8, llm_dim) -> (B, llm_dim)
        x = torch.mean(stage, dim=1) 
        
        # 第二步：第一层线性映射 (提炼特征)
        # 维度: (B, llm_dim) -> (B, hidden_dim)
        x = self.fc1(x)
        
        # 第三步：非线性激活 (增加表达能力)
        x = self.activation(x)
        
        # 第四步：Dropout (强制泛化)
        x = self.dropout(x)
        
        # 第五步：第二层线性映射 (输出结果)
        # 维度: (B, hidden_dim) -> (B, num_stage_classes)
        logits = self.fc2(x)
        
        return logits