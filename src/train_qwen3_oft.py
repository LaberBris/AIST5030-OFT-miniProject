from peft import OFTConfig
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from datasets import Dataset

import torch
import os
import logging

SEED = 42

MODEL = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "./output_qwen3_oft"

# OFT Config
config = OFTConfig(
    oft_block_size=32,
    use_cayley_neumann=True,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM",
    module_dropout=0.0, 
    init_weights=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

model.supports_gradient_checkpointing = True
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.is_parallelizable = True

# Load dataset
def load_and_prepare_dataset():
    dataset = Dataset.from_parquet('./datasets/train-00000-of-00001.parquet')
    train_dataset = dataset
    
    print(f"Training set size: {len(train_dataset)}")
    
    def preprocess_function(examples):
        texts = []
        for problem, solution in zip(examples["problem"], examples["solution"]):
            text = f"Problem: {problem}\nSolution: {solution}"
            texts.append(text)
        
        tokenized = tokenizer(texts, truncation=True, max_length=1024, padding=True)
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized
    
    tokenized_dataset = train_dataset.map(
        preprocess_function, 
        batched=True, 
        remove_columns=train_dataset.column_names
    )
    
    return tokenized_dataset

tokenized_dataset = load_and_prepare_dataset()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=1,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=300,
    weight_decay=0.01,
    gradient_checkpointing=True,

    eval_strategy="no",

    logging_dir='logs/training.log',
    logging_strategy='steps',
    logging_steps=50,
    logging_first_step=True,
    seed=SEED,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    peft_config=config,
    data_collator=data_collator
)

print("Starting training...")
trainer.train()

for obj in trainer.state.log_history:
    logging.info(str(obj))

print("Saving model...")

trainer.save_model(
    os.path.join(OUTPUT_DIR, "oft_model"),
)

tokenizer.save_pretrained(
    os.path.join(OUTPUT_DIR, "oft_model"),
)

print("Training completed!")