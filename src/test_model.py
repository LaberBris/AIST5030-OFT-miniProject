from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread
import sys
from datasets import Dataset
import re
import json
import os
from datetime import datetime
from tqdm import tqdm

SYSTEM_PROMPT = r"""
You are an expert mathematician. Solve the problem step by step. 
At the end, place your final answer in \boxed{} format. 
Example: \boxed{42} or \boxed{\frac{1}{2}} or \boxed{\text{Alice}}.
Do not put any other text after the boxed answer.
"""

dataset_dir = "your_dataset_dir"

def load_test_dataset():
    """Load test dataset"""
    test_dataset = Dataset.from_parquet(dataset_dir)
    test_dataset = test_dataset.select(range(len(test_dataset)))
    # test_dataset = test_dataset.select(range(10))
    print(f"Test dataset size: {len(test_dataset)}")
    return test_dataset

def extract_answer(text):
    """Extract answer in \boxed{} format from generated text (supports arbitrary nesting)"""
    # Find the last occurrence of \boxed{
    start_marker = '\\boxed{'
    start_idx = text.rfind(start_marker)
    
    if start_idx == -1:
        return None
    
    # Start counting from after \boxed{
    content_start = start_idx + len(start_marker)
    brace_count = 1  # Already have one {
    i = content_start
    
    # Manually match brackets
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[content_start:i-1].strip()  # Exclude the final }
    
    return None

def normalize_answer(answer):
    """Normalize answer - only remove spaces"""
    if not answer:
        return ""
    
    # Remove \left and \right
    answer = answer.replace("\\left", "").replace("\\right", "")
    
    # Remove all whitespace characters (spaces, tabs, newlines, etc.)
    answer = re.sub(r'\s+', '', answer)
    
    # Unify fraction format (optional)
    answer = answer.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac')
    
    return answer

def check_answer(predicted, ground_truth):
    """Check if answer is correct"""
    pred_norm = normalize_answer(predicted)
    truth_norm = normalize_answer(ground_truth)
    
    # Output normalized answers for debugging (commented out)
    print(f"Prediction: {repr(pred_norm)}")
    print(f"Ground Truth: {repr(truth_norm)}")
    
    return pred_norm == truth_norm

def test_model(model_path):
    print(f"Loading model: {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Model loaded successfully!")

    test_dataset = load_test_dataset()

    total = len(test_dataset)
    correct = 0

    results = []
    
    # Extract model name from path for output directory
    model_name = os.path.basename(os.path.normpath(model_path))
    
    # Create output directory: result/[model]/
    output_dir = os.path.join("result", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nStarting test...")
    
    # Use tqdm for progress bar
    for i, example in tqdm(enumerate(test_dataset), total=total, desc="Testing", unit="sample"):
        problem = example["problem"]
        ground_truth = example["answer"]
        
        prompt = SYSTEM_PROMPT + "\n" + problem
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(generated_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        
        model_answer = extract_answer(generated_text)
        
        is_correct = False
        if model_answer and check_answer(model_answer, ground_truth):
            correct += 1
            is_correct = True
        else:
            pass

        results.append({
            "problem": problem,
            "ground_truth": ground_truth,
            "prediction": model_answer,
            "is_correct": is_correct
        })
        
        current_accuracy = correct / (i + 1) * 100
        tqdm.write(f"Test {i+1}/{total}: Accuracy = {current_accuracy:.2f}%")
    
    # Calculate accuracy
    accuracy = correct / total * 100
    print("=" * 80)
    print("\nTest completed!")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    output_file = os.path.join(output_dir, f"test_results_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Test Samples: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
    
    print(f"Summary saved to {summary_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <model path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_model(model_path)