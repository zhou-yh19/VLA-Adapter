# VLA 训练与推理：模型加载与调用路径说明

本文档整理自对 `finute_stage_server.py` 与 `openvla_utils.py` 中模型加载方式、以及 `modeling_prismatic.py` 中实际执行路径的说明。

## 1. `from_config` 与 `from_pretrained` 的关系

两者都是 Hugging Face `AutoModelForVision2Seq` 的工厂方法，**在本项目中指向同一模型类**，区别主要在于**权重如何初始化**，而不是「父类 vs 子类」。

在本地 checkpoint 路径下，会先注册：

- `AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)`

因此：

- `AutoModelForVision2Seq.from_config(config, ...)` → 实例化 **`OpenVLAForActionPrediction`**，参数随机初始化（例如 minivlm 分支中再 `load_state_dict` 对齐权重）。
- `AutoModelForVision2Seq.from_pretrained(path, ...)` → 同样实例化 **`OpenVLAForActionPrediction`**，并从磁盘或 Hub 加载权重。

从 Hugging Face Hub 加载时，本地可能不显式 `register`，但仓库里 `config.json` 的 `auto_map` / `model_type` 会指向同一套建模代码；本地脚本中也有 `update_auto_map` 等逻辑，保证对 `modeling_prismatic.py` 的修改能生效。

**结论**：两种加载方式对应的都是 **`OpenVLAForActionPrediction`**（继承自 `PrismaticForConditionalGeneration`），不是「一个用父类、一个用子类」。

---

## 2. 训练时的调用路径

**位置**：`vla-scripts/finetune_stage_server.py` 中 `run_forward_pass` 等对 `vla` 的可调用写法：`vla(...)`。

这等价于调用 **`OpenVLAForActionPrediction.forward`**。子类 **`OpenVLAForActionPrediction` 未重写 `forward`**，因此实际执行的是父类 **`PrismaticForConditionalGeneration.forward`**（`prismatic/extern/hf/modeling_prismatic.py` 中从约第 596 行起的多模态前向逻辑）。

训练中的 L1 动作回归使用的是**单独的 `action_head`**：

- `action_head.module.predict_action(multi_layer_hidden_states, ...)`

这是 **动作头（action head）模块**上的 `predict_action`，用于从 `vla(...)` 返回的 `hidden_states` 预测连续动作，**不是** VLA 模型类上的 `predict_action`。

---

## 3. 推理时的调用路径

**位置**：`experiments/robot/openvla_utils.py` 中通过 `get_vla` 加载模型后，策略侧调用 **`vla.predict_action(...)`**。

这进入的是 **`OpenVLAForActionPrediction.predict_action`**：内部会准备占位 token、构造嵌入，再经 `_regression_or_discrete_prediction` 等逻辑，**不**再走训练时那条完整的 `PrismaticForConditionalGeneration.forward` 入口（与 `vla(...)` 是两条实现路径）。

---

## 4. 对照小结

| 维度 | 训练 | 推理 |
|------|------|------|
| 典型入口 | `vla(...)`（即 `forward`） | `vla.predict_action(...)` |
| 实际执行的类方法 | `OpenVLAForActionPrediction` 继承的 **`PrismaticForConditionalGeneration.forward`** | **`OpenVLAForActionPrediction.predict_action`** |
| 动作连续值 | 由外部 **`action_head.predict_action`** 基于 `forward` 的 `hidden_states` 得到 | 由 **`predict_action`** 内部流程（含 `_regression_or_discrete_prediction` 等）得到 |

**一句话**：实例类型在注册/auto_map 下均为 **`OpenVLAForActionPrediction`**；训练主路径是**继承来的 `forward`** + 外部 **`action_head`**；推理主路径是子类上的 **`predict_action`**。

---

## 5. 相关文件索引

| 文件 | 说明 |
|------|------|
| `vla-scripts/finetune_stage_server.py` | `AutoModelForVision2Seq.register`；`from_config` / `from_pretrained`；`run_forward_pass` 中 `vla(...)` |
| `experiments/robot/openvla_utils.py` | `get_vla` 中 `from_pretrained`；推理中 `predict_action` |
| `prismatic/extern/hf/modeling_prismatic.py` | `PrismaticForConditionalGeneration.forward`；`OpenVLAForActionPrediction` 与 `predict_action` |
