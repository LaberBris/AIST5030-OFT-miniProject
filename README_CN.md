# Qwen3-1.7B OFT 微调指南

本指南提供了在 Qwen3-1.7B 模型上使用 OFT (Orthogonal Fine-Tuning) 进行微调的详细步骤。

## 目录

- `src/test_model.py` - 测试模型推理和评估准确率
- `src/train_qwen3_oft.py` - OFT 微调训练脚本
- `report/` - 项目报告目录，包含 [report.pdf](report/report.pdf)

## 环境准备

### 安装依赖

```bash
pip install -r requirements.txt
```

所需包：
- torch>=2.0.0
- transformers>=4.30.0
- datasets>=2.14.0
- peft>=0.5.0
- trl>=0.7.0
- modelscope>=1.9.0

### 模型准备

确保已下载 Qwen3-1.7B 模型。模型将通过 ModelScope 自动加载，使用标识符 `Qwen/Qwen3-1.7B`。

## 使用方法

### 1. OFT 微调训练

运行训练脚本：

```bash
python src/train_qwen3_oft.py
```

**训练配置：**
- 模型：`Qwen/Qwen3-1.7B`
- 输出目录：`./output_qwen3_oft`
- 训练轮数：1
- 学习率：2e-5
- 每设备批量大小：1
- 梯度累积步数：4
- 预热步数：300
- 权重衰减：0.01
- 最大序列长度：1024

训练完成后，模型将保存到 `./output_qwen3_oft/oft_model`。

### 2. 测试模型性能

使用微调后的模型路径运行测试脚本：

```bash
python src/test_model.py ./output_qwen3_oft/oft_model
```

测试脚本将：
- 加载微调后的模型
- 在测试数据集上评估
- 计算准确率（匹配 `\boxed{}` 格式的答案）
- 将详细结果保存到 `result/[模型名称]/test_results_[时间戳].json`
- 将摘要保存到 `result/[模型名称]/summary_[时间戳].txt`

### 3. 加载和使用微调后的模型

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./output_qwen3_oft/oft_model"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# 生成文本
def generate_text(prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    output_text = tokenizer.decode(
        generated_ids[0][len(inputs.input_ids[0]):], 
        skip_special_tokens=True
    )
    return output_text

# 测试生成
test_prompt = "解决这个数学问题：2 + 2 = ?"
print(generate_text(test_prompt))
```

## 参数说明

### OFT 配置参数

基于 `adapter_config.json` 中的实际实现：

- `oft_block_size`: OFT 块大小，设置为 32
- `use_cayley_neumann`: 使用 Cayley-Neumann 近似，设置为 True
- `num_cayley_neumann_terms`: Cayley-Neumann 项数，设置为 5
- `target_modules`: 应用 OFT 的目标模块：
  - `gate_proj`, `q_proj`, `up_proj`, `o_proj`, `down_proj`, `v_proj`, `k_proj`
- `module_dropout`: 模块 dropout 率，设置为 0.0
- `init_weights`: 初始化权重，设置为 True
- `bias`: 偏置配置，设置为 "none"
- `task_type`: 任务类型，设置为 "CAUSAL_LM"
- `eps`: 数值稳定性 epsilon 值，设置为 6e-05

### 训练参数

- `num_train_epochs`: 训练轮数（默认：1）
- `learning_rate`: 学习率（默认：2e-5）
- `per_device_train_batch_size`: 每设备训练批量大小（默认：1）
- `gradient_accumulation_steps`: 梯度累积步数（默认：4）
- `warmup_steps`: 学习率预热步数（默认：300）
- `weight_decay`: 权重衰减（默认：0.01）
- `gradient_checkpointing`: 启用梯度检查点（默认：True）
- `logging_steps`: 日志记录步数间隔（默认：50）
- `seed`: 随机种子（默认：42）

### 数据集预处理

- `max_length`: 最大序列长度（默认：1024）
- 数据集格式：Parquet 文件，包含 "problem" 和 "solution" 列
- 文本格式：`Problem: {problem}\nSolution: {solution}`

## 注意事项

1. **内存要求**：OFT 微调相比全参数微调节省内存，但仍需要足够的 GPU 内存。默认启用了梯度检查点以减少内存使用。

2. **数据集准备**：确保数据集为 Parquet 格式且具有适当的列。当前实现期望训练数据包含 "problem" 和 "solution" 列，测试数据包含 "problem" 和 "answer" 列。

3. **答案提取**：测试脚本从生成文本中的 `\boxed{}` 格式提取答案。确保模型输出遵循此格式以进行准确评估。

4. **模型保存**：训练完成后，模型以 PEFT 格式保存。可以直接使用模型路径加载微调后的模型。

5. **评估指标**：通过比较标准化后的答案来计算准确率（移除空白字符、移除 `\left`/`\right`、统一分数格式）。

## 故障排除

- **CUDA 内存不足**：减小 `per_device_train_batch_size` 或降低 `oft_block_size`
- **模型加载错误**：确保模型路径正确且模型文件完整
- **数据集错误**：检查数据集格式和预处理函数
- **准确率低**：增加 `num_train_epochs`、调整 `learning_rate` 或验证数据集质量
- **答案提取失败**：确保模型以 `\boxed{}` 格式输出答案

## 项目结构

```
AIST5030-OFT-miniProject/
├── src/
│   ├── train_qwen3_oft.py    # 训练脚本
│   └── test_model.py          # 测试脚本
├── output_qwen3_oft/
│   └── oft_model/             # 微调模型输出
├── result/
│   └── [模型名称]/            # 测试结果和摘要
├── datasets/
│   ├── train-*.parquet        # 训练数据集
│   └── test-*.parquet         # 测试数据集
├── requirements.txt           # Python 依赖
├── README.md                  # 本文件（英文版）
└── README_CN.md               # 中文版
```

## 参考资料

- [PEFT 官方文档](https://huggingface.co/docs/peft/index)
- [Qwen3 模型文档](https://modelscope.cn/models/Qwen/Qwen3-1.7B/summary)
- [OFT 论文](https://arxiv.org/abs/2306.07280)
- [TRL 文档](https://huggingface.co/docs/trl/index)
