import json
import logging
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)

instruction_string = """
The following are multiple choice questions (with answers) about {}. 
Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.
"""

def generate_5shot_prompt(data, category):
    """Generates a 5-shot prompt for the specified category."""
    category_data = data.filter(lambda x: x["category"] == category)
    
    if len(category_data) < 5:
        raise ValueError(f"Not enough examples for category {category}. Found {len(category_data)} examples.")
    
    prompt = instruction_string.format(category.strip())
    for i in range(5):  
        example = category_data[i]
        prompt += f"Question: {example['question']}\n"
        prompt += f"Options: {', '.join(example['options'])}\n"
        prompt += f"{example['cot_content']}\n\n"
    
    return prompt

def extract_answer(response):
    """
    Attempts to extract the multiple-choice answer from the response text using regex.
    Looks for patterns like "the answer is (X)", etc.
    """
    patterns = [
        r"answer is\s?\(?([A-J])\)?",
        r"\.*\s*[Aa]nswer:\s*\(?([A-J])\)?",
        r"\b([A-J])\b(?!.*\b[A-J]\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.group(1):
            return match.group(1).upper()
    return None

def clean_response(response, prompt):
    """Optionally remove the original prompt from the model's output."""
    if response.startswith(prompt):
        return response[len(prompt):].strip()
    return response.strip()

def generate_input_texts(category_data, cot_prompt):
    """
    Returns a list of input texts for a batch of examples.
    Each input text is a single prompt for the question.
    """
    input_texts = []
    for row in category_data:
        question = row["question"]
        options = row["options"]
        options_string = ", ".join([f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options)])
        input_texts.append(f"{cot_prompt}\nQuestion: {question}\nOptions: {options_string}\n")
    return input_texts

@torch.no_grad()
def generate_responses_in_batches(model, tokenizer, prompts, batch_size=4, max_new_tokens=256, **generate_kwargs):
    """
    Takes a list of prompts and performs batched generation.
    Returns a list of decoded model outputs (one per prompt).
    """
    responses = []
    for start_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start_idx : start_idx + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            **generate_kwargs
        )
        batch_responses = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
        responses.extend(batch_responses)
    return responses

def calculate_accuracy(correct, total):
    return correct / total if total > 0 else 0

def save_results(results, accuracies, save_path, file_name="deepseek1.5B_2.json"):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)  
    with open(save_path / file_name, "w", encoding="utf-8") as f:
        json.dump({"results": results, "accuracies": accuracies}, f, indent=4)
    logging.info(f"Results saved to {save_path / file_name}")

def query(ds, model_path, save_path, batch_size=4):
    """
    Load the model with Accelerate, run inference in a batched manner,
    and store results.
    """
    accelerator = Accelerator()
    
    logging.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = accelerator.prepare(model)

    results = []
    accuracies = {}
    overall_correct = 0
    overall_total = 0

    logging.info("Evaluating...")
    data, categories = ds["test"], ds["test"].unique("category")
    
    for category in categories:
        try:
            cot_prompt = generate_5shot_prompt(data, category)
        except ValueError as e:
            logging.warning(e)
            continue
        
        logging.info(f"Processing category: {category}")
        category_data = data.filter(lambda x: x["category"] == category)
        
        input_texts = generate_input_texts(category_data, cot_prompt)
        
        logging.info(f"Generating responses for {len(category_data)} samples with batch_size={batch_size} ...")
        batch_responses = generate_responses_in_batches(
            model=model, 
            tokenizer=tokenizer, 
            prompts=input_texts, 
            batch_size=batch_size
        )
        
        correct = 0
        total = 0
        
        for row, response_text, prompt_text in zip(category_data, batch_responses, input_texts):
            question_id = row["question_id"]
            question = row["question"]
            options = row["options"]
            true_answer = row["answer"].upper()

            cleaned_resp = clean_response(response_text, prompt_text)
            extracted_answer = extract_answer(cleaned_resp)
            if not extracted_answer:
                extracted_answer = random.choice([chr(65 + i) for i in range(len(options))])
            
            total += 1
            overall_total += 1
            if extracted_answer == true_answer:
                correct += 1
                overall_correct += 1

            results.append(
                {
                    "question_id": question_id,
                    "category": category,
                    "question": question,
                    "options": options,
                    "true_answer": true_answer,
                    "model_response": cleaned_resp,
                    "extracted_answer": extracted_answer,
                }
            )

        category_accuracy = calculate_accuracy(correct, total)
        logging.info(f"Accuracy for category {category}: {category_accuracy*100:.2f}%")
        accuracies[category] = category_accuracy

    overall_accuracy = calculate_accuracy(overall_correct, overall_total)
    logging.info(f"Overall Accuracy: {overall_accuracy*100:.2f}%")
    accuracies["overall"] = overall_accuracy

    save_results(results, accuracies, save_path, file_name="deepseek7B_1.json")

if __name__ == "__main__":
    logging.info("Loading dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    logging.info("Dataset loaded.")
    
    model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    model_name_or_path2 = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    save_path = "results/mmlupro"
    
    query(ds, model_name_or_path2, save_path, batch_size=4)

