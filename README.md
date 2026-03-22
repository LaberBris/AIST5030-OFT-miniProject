# Qwen3-1.7B OFT Fine-Tuning Guide

This guide provides detailed steps for fine-tuning the Qwen3-1.7B model using OFT (Orthogonal Fine-Tuning).

## Table of Contents

- `src/test_model.py` - Test model inference and evaluate accuracy
- `src/train_qwen3_oft.py` - OFT fine-tuning training script

## Environment Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- torch>=2.0.0
- transformers>=4.30.0
- datasets>=2.14.0
- peft>=0.5.0
- trl>=0.7.0
- modelscope>=1.9.0

### Model Preparation

Ensure you have downloaded the Qwen3-1.7B model. The model will be automatically loaded from ModelScope using the identifier `Qwen/Qwen3-1.7B`.

## Usage

### 1. OFT Fine-Tuning Training

Run the training script:

```bash
python src/train_qwen3_oft.py
```

**Training Configuration:**
- Model: `Qwen/Qwen3-1.7B`
- Output directory: `./output_qwen3_oft`
- Training epochs: 1
- Learning rate: 2e-5
- Batch size per device: 1
- Gradient accumulation steps: 4
- Warmup steps: 300
- Weight decay: 0.01
- Max sequence length: 1024

After training completes, the model will be saved to `./output_qwen3_oft/oft_model`.

### 2. Test Model Performance

Run the test script with the fine-tuned model path:

```bash
python src/test_model.py ./output_qwen3_oft/oft_model
```

The test script will:
- Load the fine-tuned model
- Evaluate on the test dataset
- Calculate accuracy (matching answers in `\boxed{}` format)
- Save detailed results to `result/[model_name]/test_results_[timestamp].json`
- Save summary to `result/[model_name]/summary_[timestamp].txt`

### 3. Load and Use Fine-Tuned Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./output_qwen3_oft/oft_model"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Generate text
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

# Test generation
test_prompt = "Solve this math problem: 2 + 2 = ?"
print(generate_text(test_prompt))
```

## Parameter Description

### OFT Configuration Parameters

Based on the actual implementation in `adapter_config.json`:

- `oft_block_size`: OFT block size, set to 32
- `use_cayley_neumann`: Use Cayley-Neumann approximation, set to True
- `num_cayley_neumann_terms`: Number of Cayley-Neumann terms, set to 5
- `target_modules`: Target modules for OFT application:
  - `gate_proj`, `q_proj`, `up_proj`, `o_proj`, `down_proj`, `v_proj`, `k_proj`
- `module_dropout`: Module dropout rate, set to 0.0
- `init_weights`: Initialize weights, set to True
- `bias`: Bias configuration, set to "none"
- `task_type`: Task type, set to "CAUSAL_LM"
- `eps`: Epsilon for numerical stability, set to 6e-05

### Training Parameters

- `num_train_epochs`: Number of training epochs (default: 1)
- `learning_rate`: Learning rate (default: 2e-5)
- `per_device_train_batch_size`: Training batch size per device (default: 1)
- `gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `warmup_steps`: Learning rate warmup steps (default: 300)
- `weight_decay`: Weight decay (default: 0.01)
- `gradient_checkpointing`: Enable gradient checkpointing (default: True)
- `logging_steps`: Logging interval in steps (default: 50)
- `seed`: Random seed (default: 42)

### Dataset Preprocessing

- `max_length`: Maximum sequence length (default: 1024)
- Dataset format: Parquet file with "problem" and "solution" columns
- Text format: `Problem: {problem}\nSolution: {solution}`

## Notes

1. **Memory Requirements**: OFT fine-tuning is more memory-efficient than full parameter fine-tuning, but still requires sufficient GPU memory. Gradient checkpointing is enabled by default to reduce memory usage.

2. **Dataset Preparation**: Ensure your dataset is in Parquet format with appropriate columns. The current implementation expects "problem" and "solution" columns for training, and "problem" and "answer" columns for testing.

3. **Answer Extraction**: The test script extracts answers from the `\boxed{}` format in generated text. Ensure your model outputs follow this format for accurate evaluation.

4. **Model Saving**: After training, the model is saved in PEFT format. The fine-tuned model can be loaded directly using the model path.

5. **Evaluation Metrics**: Accuracy is calculated by comparing normalized answers (whitespace removed, `\left`/`\right` removed, fraction format unified).

## Troubleshooting

- **CUDA Out of Memory**: Reduce `per_device_train_batch_size` or decrease `oft_block_size`
- **Model Loading Error**: Ensure the model path is correct and model files are complete
- **Dataset Error**: Check dataset format and preprocessing function
- **Low Accuracy**: Increase `num_train_epochs`, adjust `learning_rate`, or verify dataset quality
- **Answer Extraction Failure**: Ensure model outputs answers in `\boxed{}` format

## Project Structure

```
AIST5030-OFT-miniProject/
├── src/
│   ├── train_qwen3_oft.py    # Training script
│   └── test_model.py          # Testing script
├── output_qwen3_oft/
│   └── oft_model/             # Fine-tuned model output
├── result/
│   └── [model_name]/          # Test results and summaries
├── datasets/
│   ├── train-*.parquet        # Training dataset
│   └── test-*.parquet         # Test dataset
├── requirements.txt           # Python dependencies
├── README.md                  # This file (English)
└── README_CN.md               # Chinese version
```

## References

- [PEFT Official Documentation](https://huggingface.co/docs/peft/index)
- [Qwen3 Model Documentation](https://modelscope.cn/models/Qwen/Qwen3-1.7B/summary)
- [OFT Paper](https://arxiv.org/abs/2306.07280)
- [TRL Documentation](https://huggingface.co/docs/trl/index)
