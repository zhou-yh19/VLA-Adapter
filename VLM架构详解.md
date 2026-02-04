# VLA-Adapter 中 VLM 的实现架构详解

## 一、整体架构概览

VLA-Adapter 中的 VLM（Vision-Language Model）采用**多模态融合架构**，主要由以下三个核心组件组成：

```
输入: [RGB图像] + [文本指令] + [ActionQuery]
         ↓
    ┌─────────────────────────────────────┐
    │  1. Vision Backbone (视觉编码器)     │
    │     - SigLIP ViT                     │
    │     - DINOv2 ViT (可选，融合模式)    │
    └─────────────────────────────────────┘
         ↓ [patch_features: (B, num_patches, vision_dim)]
    ┌─────────────────────────────────────┐
    │  2. Projector (投影层)               │
    │     - MLPProjector / LinearProjector │
    │     - 将视觉特征投影到LLM空间         │
    └─────────────────────────────────────┘
         ↓ [projected_patches: (B, num_patches, llm_dim)]
    ┌─────────────────────────────────────┐
    │  3. Multimodal Fusion (多模态融合)   │
    │     [BOS] + [Vision Patches] + [Text]│
    └─────────────────────────────────────┘
         ↓ [multimodal_embeddings]
    ┌─────────────────────────────────────┐
    │  4. LLM Backbone (语言模型)          │
    │     -  Qwen2.5                      │
    │     - 多层Transformer                │
    └─────────────────────────────────────┘
         ↓ [hidden_states: 所有层的输出]
    ┌─────────────────────────────────────┐
    │  5. Action Head (动作头)             │
    │     - 接收多层hidden states          │
    │     - 预测连续动作值                  │
    └─────────────────────────────────────┘
```

---

## 二、核心组件详解

### 2.1 Vision Backbone（视觉编码器）

**位置**：`prismatic/extern/hf/modeling_prismatic.py` → `PrismaticVisionBackbone`

**功能**：将输入图像编码为视觉特征patch

**架构特点**：

1. **单backbone模式**（默认）：
   - 使用 **SigLIP ViT** 作为主要视觉编码器
   - 输出形状：`(batch_size, num_patches, vision_dim)`
   - 例如：`(1, 257, 1024)` - 256个patch + 1个CLS token

2. **融合backbone模式**（可选）：
   - 同时使用 **SigLIP ViT** 和 **DINOv2 ViT**
   - 两个编码器的特征在特征维度上拼接
   - 输出形状：`(batch_size, num_patches, vision_dim_siglip + vision_dim_dinov2)`
   - 例如：`(1, 257, 1024 + 768) = (1, 257, 1792)`

**代码实现**：

```python
# prismatic/extern/hf/modeling_prismatic.py 第196-237行
def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
    if self.num_images_in_input == 1:
        if not self.use_fused_vision_backbone:
            # 单backbone：只使用SigLIP
            return self.featurizer(pixel_values)
        else:
            # 融合backbone：SigLIP + DINOv2
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches = self.featurizer(img)  # SigLIP
            patches_fused = self.fused_featurizer(img_fused)  # DINOv2
            return torch.cat([patches, patches_fused], dim=2)  # 在特征维度拼接
```

**关键点**：
- 使用TIMM库加载预训练的ViT模型
- 提取**倒数第二层**的特征（`get_intermediate_layers`）
- 支持多图像输入（1-3张图像）

---

### 2.2 Projector（投影层）

**位置**：`prismatic/util/nn_utils.py` → `MLPProjector` / `LinearProjector`

**功能**：将视觉特征从视觉空间投影到LLM的embedding空间

**架构类型**：

#### 类型1：LinearProjector（线性投影）

```python
# 简单的线性变换
projected_features = Linear(vision_dim → llm_dim)
```

#### 类型2：MLPProjector（MLP投影，默认）

```python
# prismatic/util/nn_utils.py 第21-34行
class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int):
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim, bias=True),  # 第一层
            nn.GELU(),                                   # 激活函数
            nn.Linear(llm_dim, llm_dim, bias=True),     # 第二层
        )
```

**输入输出**：
- 输入：`(batch_size, num_patches, vision_dim)`
- 输出：`(batch_size, num_patches, llm_dim)`
- 例如：`(1, 257, 1024)` → `(1, 257, 4096)`（Llama-2 7B）

#### 类型3：FusedMLPProjector（融合MLP投影）

用于融合backbone模式，包含3层：

```python
# prismatic/util/nn_utils.py 第37-53行
class FusedMLPProjector(nn.Module):
    def __init__(self, fused_vision_dim: int, llm_dim: int):
        initial_projection_dim = fused_vision_dim * 4
        self.projector = nn.Sequential(
            nn.Linear(fused_vision_dim, initial_projection_dim, bias=True),
            nn.GELU(),
            nn.Linear(initial_projection_dim, llm_dim, bias=True),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim, bias=True),
        )
```

---

### 2.3 Multimodal Fusion（多模态融合）

**位置**：`prismatic/models/vlms/prismatic.py` → `PrismaticVLM.forward()`

**功能**：将视觉patch embeddings和文本embeddings拼接成统一的多模态序列

**融合方式**：

```python
# prismatic/models/vlms/prismatic.py 第388-396行
multimodal_embeddings = torch.cat(
    [
        input_embeddings[multimodal_indices, :1, :],  # BOS token
        projected_patch_embeddings,                   # Vision patches
        input_embeddings[multimodal_indices, 1:, :],  # Text tokens
    ],
    dim=1,  # 在序列维度拼接
)
```

**序列结构**：

```
[BOS] + [P1] [P2] ... [P257] + [Text Token 1] [Text Token 2] ... [ActionQuery Tokens] + [STOP]
  ↓         ↓      ↓      ↓           ↓              ↓                    ↓                ↓
[BOS]   Vision Patches (257个)      Text Tokens      ActionQuery (64个)              STOP Token
```

**Attention关系**（双向，所有位置可以互相看到）：
- ✅ Vision Patches ↔ Text Tokens（可以互相attention）
- ✅ Vision Patches ↔ ActionQuery Tokens（可以互相attention）
- ✅ Text Tokens ↔ ActionQuery Tokens（可以互相attention）
- ✅ 所有位置 ↔ 所有位置（完全双向）

**形状**：
- BOS token: `(batch_size, 1, llm_dim)`
- Vision patches: `(batch_size, num_patches, llm_dim)`，例如 `(1, 257, 4096)`
- Text tokens: `(batch_size, text_len, llm_dim)`
- **最终multimodal_embeddings**: `(batch_size, 1 + num_patches + text_len, llm_dim)`

**Attention Mask**：

```python
# 为所有vision patches创建attention mask（全部为True）
projected_patch_attention_mask = torch.full(
    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
    True,  # 所有vision patches都参与attention
    dtype=attention_mask.dtype,
    device=attention_mask.device,
)

multimodal_attention_mask = torch.cat(
    [
        attention_mask[multimodal_indices, :1],      # BOS
        projected_patch_attention_mask,               # Vision patches
        attention_mask[multimodal_indices, 1:],       # Text tokens
    ],
    dim=1,
)
```

---

### 2.4 LLM Backbone（语言模型）

**位置**：`prismatic/models/backbones/llm/base_llm.py` → `HFCausalLLMBackbone`

**功能**：处理多模态序列，生成所有层的hidden states

**支持的模型**：
- **Llama-2**（7B, 13B等）
- **Mistral**
- 其他HuggingFace Causal LM

**架构特点**：

1. **多层Transformer**：
   - 通常包含32层（Llama-2 7B）或更多
   - 每层包含Self-Attention和FFN

2. **输出所有层的hidden states**：
   ```python
   # prismatic/models/vlms/prismatic.py 第469行
   llm_output = self.llm_backbone(
       inputs_embeds=multimodal_embeddings,
       output_hidden_states=True,  # ✅ 返回所有层的hidden states
       ...
   )
   ```

3. **Hidden States形状**：
   - 每一层：`(batch_size, seq_len, llm_dim)`
   - 所有层：`tuple of (batch_size, seq_len, llm_dim)`，长度为层数
   - 例如：32层 × `(1, 1+257+text_len, 4096)`

**关键点**：
- 使用**双向注意力**（bi-directional self-attention），不是因果注意力
- 所有位置的token可以互相看到，支持并行计算
- 输出包含：`logits`、`hidden_states`（所有层）、`past_key_values`（不使用）

**重要：Attention机制详解**

VLA-Adapter使用的是**非因果双向自注意力**（non-causal bi-directional self-attention），这意味着：

1. **所有token可以互相看到**：
   - ✅ Vision Patches 可以看到 Text Tokens 和 ActionQuery Tokens
   - ✅ Text Tokens 可以看到 Vision Patches 和 ActionQuery Tokens
   - ✅ ActionQuery Tokens 可以看到 Vision Patches 和 Text Tokens

2. **序列结构**：
   ```
   [BOS] + [Vision Patches] + [Text Tokens] + [ActionQuery Tokens] + [STOP]
   ```

3. **Attention Mask设置**：
   ```python
   # 所有位置的attention mask都是True
   attention_mask = [True, True, ..., True]  # 所有位置都可以互相attention
   ```

4. **代码证据**：
   ```python
   # modeling_prismatic.py 第756行注释
   # "needed in non-causal bi-directional self-attention, as it appears at train time"
   
   # 第767行：action_query的attention mask也是True
   mask_extension = torch.ones(...)  # action_query的mask都是True
   attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)
   ```

**因此，图片和文本的hidden states可以看到action_query的hidden states！**

---

### 2.5 ActionQuery（动作查询）

**位置**：`prismatic/extern/hf/modeling_prismatic.py` → `PrismaticForConditionalGeneration`

**功能**：作为可学习的embeddings，直接输入到LLM中

**实现方式**：

```python
# 第374-376行：定义可学习的ActionQuery Embeddings
self.action_queries = nn.Embedding(NUM_TOKENS, self.llm_dim)
# NUM_TOKENS = 64，llm_dim = 4096（Qwen2.5）
# 形状: (64, 4096) - 64个可学习的action query embeddings
self.action_queries.weight.data.zero_()  # 初始化为全零
```

**处理流程**：

```python
# 第629-633行：获取并替换ActionQuery Embeddings
action_queries = self.action_queries.weight  # 可学习的embeddings
action_queries = action_queries.repeat(batch_size, 1, 1)  # (batch, 64, 4096)

# 替换input_embeddings中action token位置的embeddings
input_embeddings = self._replace_input_embeddings(
    input_embeddings, 
    all_actions_mask,  # 标记action token位置
    action_queries     # 可学习的action query embeddings
)
```

**关键点**：
- ✅ **ActionQuery是直接输入到LLM的embeddings**，不是token IDs
- ✅ 使用可学习的`nn.Embedding`层，初始化为全零
- ✅ 被替换到`input_embeddings`中action token的位置
- ✅ 然后和Vision Patches一起拼接，作为`multimodal_embeddings`输入到LLM
- ✅ 在LLM中，ActionQuery通过双向attention可以看到Vision和Text的信息

### 2.6 Action Head（动作头）

**位置**：`prismatic/models/action_heads.py` → `L1RegressionActionHead`

**功能**：从VLM的多层hidden states中提取信息，预测连续动作值

**输入**：
- `multi_layer_hidden_states`: `(batch, num_layers, num_tokens, hidden_dim)`
- 包含所有层的hidden states（包括ActionQuery的hidden states）

**处理流程**：
1. 分离task和action的hidden states
2. 创建Initial Action（全零张量）
3. 通过多层Bridge Attention处理
4. 输出连续动作值：`(NUM_ACTIONS_CHUNK, ACTION_DIM) = (30, 16)`

---

## 三、完整Forward流程

### 3.1 输入准备

```python
# 输入
input_ids: (batch_size, text_len)           # 文本token IDs
pixel_values: (batch_size, 3, H, W)        # RGB图像
attention_mask: (batch_size, text_len)     # 文本attention mask
```

### 3.2 视觉特征提取

```python
# Step 1: Vision Backbone
patch_features = vision_backbone(pixel_values)
# 形状: (batch_size, num_patches, vision_dim)
# 例如: (1, 257, 1024)

# Step 2: Projector
projected_patch_embeddings = projector(patch_features)
# 形状: (batch_size, num_patches, llm_dim)
# 例如: (1, 257, 4096)
```

### 3.3 文本Embedding和ActionQuery处理

```python
# Step 3a: 文本Embedding
input_embeddings = self.get_input_embeddings()(input_ids)
# 形状: (batch_size, seq_len, llm_dim)
# 例如: (1, seq_len, 4096)
# 注意：seq_len包含文本tokens和action token占位符

# Step 3b: 获取ActionQuery Embeddings（可学习的embedding）
action_queries = self.action_queries.weight  # (NUM_TOKENS, llm_dim)
# 例如: (64, 4096) - 64个可学习的action query embeddings
action_queries = action_queries.repeat(batch_size, 1, 1)  # (batch_size, NUM_TOKENS, llm_dim)

# Step 3c: 替换Action Token位置的Embeddings
all_actions_mask = self._process_action_masks(labels)  # 标记action token位置
input_embeddings = self._replace_input_embeddings(
    input_embeddings, 
    all_actions_mask,  # 布尔mask，标记哪些位置是action tokens
    action_queries      # 可学习的action query embeddings
)
# 结果：action token位置的embeddings被替换为action_query embeddings
```

### 3.4 多模态融合

```python
# Step 4: 拼接多模态序列
multimodal_embeddings = torch.cat([
    input_embeddings[:, :1, :],      # BOS token
    projected_patch_embeddings,        # Vision patches
    input_embeddings[:, 1:, :],       # Text tokens + ActionQuery embeddings
], dim=1)
# 形状: (batch_size, 1 + num_patches + seq_len, llm_dim)
# 例如: (1, 1 + 257 + (text_len + NUM_TOKENS), 4096)
# 注意：input_embeddings[:, 1:, :] 已经包含了ActionQuery的embeddings！
```

**关键点**：
- ✅ **ActionQuery是作为embeddings直接输入到LLM的**
- ✅ ActionQuery不是token IDs，而是**可学习的embeddings**（`nn.Embedding`）
- ✅ ActionQuery embeddings被替换到`input_embeddings`中action token的位置
- ✅ 然后和Vision Patches一起拼接，作为`multimodal_embeddings`输入到LLM

### 3.5 LLM Forward Pass

```python
# Step 5: LLM处理（Qwen2.5等）
llm_output = self.language_model(
    input_ids=None,  # ❌ 不使用token IDs
    inputs_embeds=multimodal_embeddings,  # ✅ 直接使用embeddings
    attention_mask=multimodal_attention_mask,
    output_hidden_states=True,  # ✅ 返回所有层
    ...
)

# 输出
logits = llm_output.logits  # (batch, seq_len, vocab_size)
hidden_states = llm_output.hidden_states  # tuple of (batch, seq_len, llm_dim) × num_layers
```

**关键点**：
- ✅ **ActionQuery作为embeddings的一部分，直接输入到LLM（Qwen2.5）中**
- ✅ 使用`inputs_embeds`而不是`input_ids`，说明是直接使用embeddings
- ✅ ActionQuery、Vision Patches、Text Tokens都在同一个序列中，通过双向attention互相看到

### 3.6 提取多层Hidden States

```python
# Step 6: 提取并组织多层hidden states
multi_layer_hidden_states = []
for item in hidden_states[0:]:  # 遍历所有层
    task_hidden_states = item[:, :num_patches]  # Vision patches
    action_hidden_states = item[:, num_patches + prompt_len:num_patches + prompt_len + NUM_TOKENS]  # ActionQuery
    all_hidden_states = torch.cat((task_hidden_states, action_hidden_states), dim=2)
    multi_layer_hidden_states.append(all_hidden_states)

multi_layer_hidden_states = torch.cat(multi_layer_hidden_states, dim=1)
# 形状: (batch, num_layers, num_tokens, hidden_dim)
```

### 3.7 Action Head预测

```python
# Step 7: Action Head预测动作
predicted_actions = action_head.predict_action(
    multi_layer_hidden_states,
    proprio=proprio,
    proprio_projector=proprio_projector,
)
# 输出形状: (NUM_ACTIONS_CHUNK, ACTION_DIM) = (30, 16)
```

---

## 四、架构特点总结

### 4.1 多模态融合方式

- **拼接式融合**：将视觉patches和文本tokens在序列维度拼接
- **位置**：`[BOS] + [Vision Patches] + [Text Tokens]`
- **优势**：简单高效，所有模态共享同一个LLM处理

### 4.2 视觉编码器

- **单backbone**：SigLIP ViT（默认）
- **融合backbone**：SigLIP + DINOv2（可选）
- **特征提取**：倒数第二层特征（避免过度抽象）

### 4.3 投影层

- **MLPProjector**：2层MLP + GELU激活（默认）
- **LinearProjector**：单层线性变换（简单场景）
- **FusedMLPProjector**：3层MLP（融合backbone）

### 4.4 LLM处理

- **双向注意力**（non-causal bi-directional self-attention）：
  - ✅ **所有位置的token可以互相看到**
  - ✅ Vision Patches 可以看到 ActionQuery Tokens
  - ✅ Text Tokens 可以看到 ActionQuery Tokens
  - ✅ ActionQuery Tokens 可以看到 Vision Patches 和 Text Tokens
  - 支持并行计算所有位置的hidden states
- **多层输出**：返回所有层的hidden states，供action head使用
- **不使用KV-cache**：并行生成，不需要缓存

### 4.5 层级对齐

- VLM的每一层hidden states → Action Head对应层的Bridge Attention
- 实现**层级对齐**的cross-attention机制

---

## 五、代码位置总结

| 组件 | 文件路径 | 类名 |
|------|---------|------|
| **Vision Backbone** | `prismatic/extern/hf/modeling_prismatic.py` | `PrismaticVisionBackbone` |
| **Projector** | `prismatic/util/nn_utils.py` | `MLPProjector` / `LinearProjector` |
| **VLM主类** | `prismatic/models/vlms/prismatic.py` | `PrismaticVLM` |
| **LLM Backbone** | `prismatic/models/backbones/llm/base_llm.py` | `HFCausalLLMBackbone` |
| **Action Head** | `prismatic/models/action_heads.py` | `L1RegressionActionHead` |
| **多模态融合** | `prismatic/models/vlms/prismatic.py` | `PrismaticVLM.forward()` |

---

## 六、Attention机制与可见性详解

### 6.1 关键问题：图片和文本的hidden states能否看到action_query的hidden states？

**答案：✅ 可以！**

### 6.2 原因分析

#### 6.2.1 使用双向注意力（非因果）

VLA-Adapter使用的是**非因果双向自注意力**（non-causal bi-directional self-attention），而不是因果注意力（causal attention）。

**代码证据**：

```python
# modeling_prismatic.py 第756行
# 注释明确说明："needed in non-causal bi-directional self-attention, as it appears at train time"
stop_token_id = torch.ones((input_ids.shape[0], 1)) * STOP_INDEX
input_ids = torch.cat([input_ids, stop_token_id], dim=-1)
```

#### 6.2.2 Attention Mask设置

所有位置的attention mask都是`True`，包括action_query：

```python
# modeling_prismatic.py 第762-767行
mask_extension = torch.ones(
    (attention_mask.shape[0], input_ids.shape[-1] - attention_mask.shape[-1])
)  # ✅ action_query的mask都是True
attention_mask = torch.cat([attention_mask, mask_extension], dim=-1)
```

#### 6.2.3 序列结构

```
序列: [BOS] + [Vision Patches] + [Text Tokens] + [ActionQuery Tokens] + [STOP]
位置:   0        1-257           258-...          ...-N                 N+1
```

**Attention关系矩阵**（所有位置都可以互相看到）：

```
        BOS  Vision  Text  ActionQuery  STOP
BOS       ✅    ✅     ✅       ✅        ✅
Vision    ✅    ✅     ✅       ✅        ✅
Text      ✅    ✅     ✅       ✅        ✅
ActionQuery ✅  ✅     ✅       ✅        ✅
STOP      ✅    ✅     ✅       ✅        ✅
```

### 6.3 实际影响

1. **Vision Patches的hidden states**：
   - ✅ 可以看到ActionQuery Tokens的hidden states
   - ✅ 可以通过attention机制获取action_query的信息
   - ✅ 在计算hidden states时，会考虑action_query的上下文

2. **Text Tokens的hidden states**：
   - ✅ 可以看到ActionQuery Tokens的hidden states
   - ✅ 可以通过attention机制获取action_query的信息
   - ✅ 在计算hidden states时，会考虑action_query的上下文

3. **ActionQuery Tokens的hidden states**：
   - ✅ 可以看到Vision Patches的hidden states
   - ✅ 可以看到Text Tokens的hidden states
   - ✅ 在计算hidden states时，会融合视觉和文本信息

### 6.4 与因果注意力的对比

| 特性 | 因果注意力（Causal） | 双向注意力（Bi-directional） |
|------|---------------------|---------------------------|
| **可见性** | ❌ 只能看到前面的token | ✅ 可以看到所有位置的token |
| **Vision → ActionQuery** | ❌ 看不到 | ✅ 可以看到 |
| **Text → ActionQuery** | ❌ 看不到 | ✅ 可以看到 |
| **ActionQuery → Vision/Text** | ❌ 看不到 | ✅ 可以看到 |
| **生成方式** | 自回归（逐个生成） | 并行（一次性计算） |
| **VLA使用** | ❌ 不使用 | ✅ 使用 |

### 6.5 为什么使用双向注意力？

1. **训练时所有token都在序列中**：
   - 训练时，action_query tokens已经在输入序列中
   - 使用双向注意力可以让所有token互相看到，学习更好的表示

2. **并行计算**：
   - 双向注意力支持并行计算所有位置的hidden states
   - 不需要自回归生成，提高效率

3. **更好的多模态融合**：
   - Vision、Text、ActionQuery可以充分交互
   - 学习更丰富的多模态表示

---

## 七、ActionQuery输入机制详解

### 7.1 ActionQuery是否直接输入到LLM？

**答案：✅ 是的！**

ActionQuery是作为**embeddings直接输入到LLM（Qwen2.5）中**的，具体流程如下：

### 7.2 完整流程

#### Step 1: 定义可学习的ActionQuery Embeddings

```python
# modeling_prismatic.py 第374-376行
self.action_queries = nn.Embedding(NUM_TOKENS, self.llm_dim)
# NUM_TOKENS = 64（action query的数量）
# llm_dim = 4096（Qwen2.5的hidden dimension）
# 形状: (64, 4096) - 64个可学习的action query embeddings
self.action_queries.weight.data.zero_()  # 初始化为全零
```

#### Step 2: 获取ActionQuery Embeddings

```python
# modeling_prismatic.py 第629-630行
action_queries = self.action_queries.weight  # (NUM_TOKENS, llm_dim)
action_queries = action_queries.repeat(batch_size, 1, 1)  # (batch_size, NUM_TOKENS, llm_dim)
```

#### Step 3: 替换到input_embeddings中

```python
# modeling_prismatic.py 第631-633行
all_actions_mask = self._process_action_masks(labels)  # 标记action token位置
input_embeddings = self._replace_input_embeddings(
    input_embeddings,      # 原始的文本embeddings
    all_actions_mask,     # 布尔mask，标记哪些位置是action tokens
    action_queries        # 可学习的action query embeddings
)
# 结果：action token位置的embeddings被替换为action_query embeddings
```

#### Step 4: 拼接成多模态序列

```python
# modeling_prismatic.py 第636-637行
multimodal_embeddings, multimodal_attention_mask = self._build_multimodal_attention(
    input_embeddings,              # 包含ActionQuery的embeddings
    projected_patch_embeddings,    # Vision patches
    attention_mask
)
# 序列: [BOS] + [Vision Patches] + [Text + ActionQuery] + [STOP]
```

#### Step 5: 输入到LLM

```python
# modeling_prismatic.py 第644-655行
language_model_output = self.language_model(
    input_ids=None,  # ❌ 不使用token IDs
    inputs_embeds=multimodal_embeddings,  # ✅ 直接使用embeddings（包含ActionQuery）
    attention_mask=multimodal_attention_mask,
    output_hidden_states=True,
    ...
)
```

### 7.3 关键理解

1. **ActionQuery不是token IDs**：
   - ❌ 不是通过`input_ids`输入
   - ✅ 而是通过`inputs_embeds`作为embeddings输入

2. **ActionQuery是可学习的embeddings**：
   - ✅ 使用`nn.Embedding`层，可以训练
   - ✅ 初始化为全零，通过训练学习合适的表示

3. **ActionQuery在序列中的位置**：
   - 序列结构：`[BOS] + [Vision Patches] + [Text Tokens] + [ActionQuery] + [STOP]`
   - ActionQuery在Text Tokens之后，作为序列的一部分

4. **ActionQuery可以看到Vision和Text**：
   - ✅ 通过双向attention，ActionQuery可以看到所有Vision Patches
   - ✅ 通过双向attention，ActionQuery可以看到所有Text Tokens
   - ✅ Vision和Text也可以看到ActionQuery

### 7.4 与Vision Patches的对比

| 特性 | Vision Patches | ActionQuery |
|------|---------------|-------------|
| **来源** | 图像通过ViT编码 | 可学习的`nn.Embedding` |
| **初始化** | 预训练的ViT权重 | 全零初始化 |
| **是否可训练** | 通常冻结 | ✅ 可训练 |
| **输入方式** | 通过`inputs_embeds` | 通过`inputs_embeds` |
| **在序列中的位置** | BOS之后 | Text Tokens之后 |
| **是否能看到其他模态** | ✅ 可以（双向attention） | ✅ 可以（双向attention） |

---

## 八、关键设计决策

1. **为什么使用拼接式融合？**
   - 简单高效，所有模态共享同一个LLM
   - 视觉patches和文本tokens在同一个序列中，可以互相attention

2. **为什么提取倒数第二层特征？**
   - 避免最后一层过度抽象
   - 保留更多视觉细节信息

3. **为什么使用双向注意力？**
   - 支持并行计算所有位置的hidden states
   - 不需要自回归生成，提高效率
   - 让ActionQuery可以看到Vision和Text的信息

4. **为什么返回所有层的hidden states？**
   - Action Head需要多层信息
   - 不同层捕获不同级别的特征（细节→抽象）

5. **为什么使用MLP Projector而不是简单线性？**
   - MLP可以学习更复杂的视觉-语言对齐
   - 提高多模态融合效果

6. **为什么ActionQuery使用可学习的embeddings？**
   - 可以训练学习合适的动作查询表示
   - 初始化为全零，通过训练优化
   - 比固定的token IDs更灵活

---

## 九、融合Backbone模式的参数设置

### 9.1 什么是融合Backbone模式？

融合Backbone模式是指同时使用两个视觉编码器（如SigLIP + DINOv2），并将它们的特征在特征维度上拼接，以获得更丰富的视觉表示。

**支持的融合Backbone组合**：
- `dinosiglip-vit-so-224px`：DINOv2 + SigLIP（224px分辨率）
- `dinosiglip-vit-so-384px`：DINOv2 + SigLIP（384px分辨率）
- `dinoclip-vit-l-336px`：DINOv2 + CLIP（336px分辨率）

### 9.2 配置参数说明

#### 9.2.1 配置文件中的参数

在模型的`config.json`配置文件中，需要设置以下参数：

```json
{
  "vision_backbone_id": "dinosiglip-vit-so-224px",  // 融合backbone的ID
  "use_fused_vision_backbone": true,                 // 显式设置为true（可选）
  "llm_backbone_id": "qwen25-0_5b-extra",
  "arch_specifier": "no-align+gelu-mlp",
  "image_resize_strategy": "letterbox",
  // ... 其他配置
}
```

**关键参数**：

1. **`vision_backbone_id`**（必需）
   - 类型：`str`
   - 说明：指定使用的视觉backbone ID
   - 融合模式示例：`"dinosiglip-vit-so-224px"`、`"dinosiglip-vit-so-384px"`、`"dinoclip-vit-l-336px"`
   - 单backbone示例：`"siglip-vit-so400m"`、`"dinov2-vit-l"`

2. **`use_fused_vision_backbone`**（可选）
   - 类型：`bool` 或 `None`
   - 说明：是否使用融合backbone模式
   - **自动推断规则**：如果`vision_backbone_id`以`"dinoclip"`或`"dinosiglip"`开头，会自动设置为`True`
   - **显式设置**：可以显式设置为`true`或`false`来覆盖自动推断

**自动推断逻辑**（来自`configuration_prismatic.py`第107-111行）：

```python
self.use_fused_vision_backbone = (
    use_fused_vision_backbone
    if use_fused_vision_backbone is not None
    else any(self.vision_backbone_id.startswith(v) for v in ["dinoclip", "dinosiglip"])
)
```

### 9.3 微调时的参数设置

#### 方法1：使用预训练模型（推荐）

如果使用预训练的融合backbone模型（如`prism-qwen25-extra-dinosiglip-224px-0_5b`），配置文件已经包含正确的设置：

```python
# vla-scripts/finetune.py
from transformers import AutoConfig, AutoModelForVision2Seq

# 加载预训练模型的配置（自动包含use_fused_vision_backbone=True）
config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
```

#### 方法2：从预训练checkpoint加载

```python
# vla-scripts/finetune.py 第820-825行
vla = AutoModelForVision2Seq.from_pretrained(
    cfg.config_file_path,  # 包含config.json的目录路径
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    trust_remote_code=False,
).to(device_id)
```

**确保`cfg.config_file_path`目录下的`config.json`包含**：
```json
{
  "vision_backbone_id": "dinosiglip-vit-so-224px",
  "use_fused_vision_backbone": true,
  // ... 其他配置
}
```

#### 方法3：使用MiniVLM加载（特殊场景）

如果使用`use_minivlm=True`模式，需要手动处理state_dict的键名映射：

```python
# vla-scripts/finetune.py 第777-816行
if cfg.use_minivlm:
    vlm = load(cfg.vlm_path, hf_token=hf_token, load_for_training=True)
    config = AutoConfig.from_pretrained("pretrained_models/configs/config.json")
    vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
    
    # 注意：融合backbone的键名映射
    replace_map = [
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),      # DINOv2 → featurizer
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"), # SigLIP → fused_featurizer
        # ... 其他映射
    ]
```

### 9.4 推理时的参数设置

#### 方法1：从checkpoint加载（推荐）

```python
# experiments/robot/openvla_utils.py 第302-310行
from transformers import AutoModelForVision2Seq

vla = AutoModelForVision2Seq.from_pretrained(
    cfg.pretrained_checkpoint,  # 包含config.json的checkpoint路径
    torch_dtype=torch.bfloat16,
    load_in_8bit=cfg.load_in_8bit,
    load_in_4bit=cfg.load_in_4bit,
    low_cpu_mem_usage=True,
    trust_remote_code=False,
)
```

#### 方法2：使用deploy.py脚本

```python
# vla-scripts/deploy.py
@dataclass
class DeployConfig:
    pretrained_checkpoint: Union[str, Path] = "path/to/checkpoint"  # 包含config.json的路径
    # ... 其他配置
```

**确保checkpoint目录下的`config.json`包含正确的融合backbone配置**。

### 9.5 ImageProcessor的设置

**重要**：使用融合backbone时，`PrismaticImageProcessor`也需要相应的设置。

#### 自动设置（推荐）

如果从`config.json`加载模型，ImageProcessor会自动从配置中读取`use_fused_vision_backbone`：

```python
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(cfg.config_file_path)
# processor.use_fused_vision_backbone 会自动从config.json读取
```

#### 手动设置

```python
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor

processor = PrismaticImageProcessor(
    use_fused_vision_backbone=True,  # 必须与模型配置一致
    image_resize_strategy="letterbox",
)
```

**关键点**：
- 融合backbone模式下，`pixel_values`的通道数为`6`（3个RGB通道 × 2个backbone）
- ImageProcessor会将输入图像处理成`(batch_size, 6, height, width)`的形状
- 模型内部会将6通道拆分为两个3通道图像，分别输入到两个backbone

### 9.6 验证融合Backbone是否启用

#### 方法1：检查模型配置

```python
config = vla.config
print(f"vision_backbone_id: {config.vision_backbone_id}")
print(f"use_fused_vision_backbone: {config.use_fused_vision_backbone}")
print(f"timm_model_ids: {config.timm_model_ids}")  # 应该包含2个模型ID
```

#### 方法2：检查模型结构

```python
# 检查是否有fused_featurizer属性
hasattr(vla.vision_backbone, 'fused_featurizer')  # 应该返回True
print(f"embed_dim: {vla.vision_backbone.embed_dim}")  # 应该是两个backbone的embed_dim之和
```

#### 方法3：检查输入形状

```python
# 融合backbone模式下，pixel_values应该是6通道
# 单backbone: (batch_size, 3, height, width)
# 融合backbone: (batch_size, 6, height, width)
print(f"pixel_values shape: {pixel_values.shape}")  # 应该是 (B, 6, H, W)
```

### 9.7 常见问题

#### Q1: 如何从单backbone切换到融合backbone？

**答**：需要修改`config.json`中的`vision_backbone_id`，并确保：
1. 新的`vision_backbone_id`是融合backbone ID（如`"dinosiglip-vit-so-224px"`）
2. 预训练权重包含两个backbone的参数
3. ImageProcessor的`use_fused_vision_backbone`设置为`True`

#### Q2: 融合backbone是否支持多图像输入？

**答**：是的，融合backbone**必须**使用多图像输入。代码中有断言：
```python
assert self.use_fused_vision_backbone, "Multi-image inputs require using fused backbone!"
```

多图像输入时，每个图像的通道数为`6`（3×2），总通道数为`6 × num_images`。

#### Q3: 融合backbone的性能影响？

**答**：
- **计算量**：约为单backbone的2倍（需要运行两个ViT）
- **内存占用**：约为单backbone的2倍
- **特征维度**：`vision_dim = siglip_dim + dinov2_dim`（例如：1024 + 768 = 1792）
- **优势**：融合了两种不同预训练目标的特征（SigLIP：视觉-语言对齐；DINOv2：自监督学习）

#### Q4: 如何确认配置文件正确？

**答**：检查`config.json`中的关键字段：
```json
{
  "vision_backbone_id": "dinosiglip-vit-so-224px",  // 必须是融合backbone ID
  "use_fused_vision_backbone": true,                 // 必须为true
  "timm_model_ids": [                                 // 应该包含2个模型ID
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_so400m_patch14_siglip_224"
  ],
  "image_sizes": [224, 224]                          // 应该包含2个分辨率
}
```

### 9.8 总结

**微调时**：
1. 确保`config.json`中`vision_backbone_id`设置为融合backbone ID（如`"dinosiglip-vit-so-224px"`）
2. `use_fused_vision_backbone`会自动推断为`True`（或显式设置为`True`）
3. 使用`AutoModelForVision2Seq.from_pretrained()`加载模型时，配置会自动应用

**推理时**：
1. 确保checkpoint目录下的`config.json`包含正确的融合backbone配置
2. ImageProcessor会自动从配置中读取`use_fused_vision_backbone`
3. 输入图像的通道数应为`6`（融合backbone模式）

**关键检查点**：
- ✅ `vision_backbone_id`以`"dinoclip"`或`"dinosiglip"`开头
- ✅ `use_fused_vision_backbone = True`
- ✅ `timm_model_ids`包含2个模型ID
- ✅ `image_sizes`包含2个分辨率值
- ✅ `pixel_values`形状为`(B, 6, H, W)`（单图像）或`(B, 6*N, H, W)`（N图像）
