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

        # 1. First linear layer (feature compression/bottleneck layer)
        # Compresses high-dimensional features from llm_dim to hidden_dim
        self.fc1 = nn.Linear(llm_dim, hidden_dim)

        # 2. Non-linear activation function
        # Using GELU (most common in LLMs) or ReLU to give the model non-linear expressive power
        self.activation = nn.GELU()

        # 3. Dropout layer (very important!)
        # Placed after non-linear activation and before second linear layer to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_rate)

        # 4. Second linear layer (final classification output layer)
        # Maps from hidden dimension to num_stage_classes categories
        self.fc2 = nn.Linear(hidden_dim, num_stage_classes)

    def forward(self, stage: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 8, llm_dim)

        # Step 1: Mean pooling over sequence dimension -> remove position dependency
        # Shape: (B, 8, llm_dim) -> (B, llm_dim)
        x = torch.mean(stage, dim=1)

        # Step 2: First linear projection (feature extraction)
        # Shape: (B, llm_dim) -> (B, hidden_dim)
        x = self.fc1(x)

        # Step 3: Non-linear activation (add expressive power)
        x = self.activation(x)

        # Step 4: Dropout (enforce generalization)
        x = self.dropout(x)

        # Step 5: Second linear projection (output)
        # Shape: (B, hidden_dim) -> (B, num_stage_classes)
        logits = self.fc2(x)

        return logits