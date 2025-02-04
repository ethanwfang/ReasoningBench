from string import ascii_uppercase
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(name):

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)

    return model, tokenizer

def generate_prompt_single(
    data, 
    instruction_str, 
    format_dict,
    n_shots=5, 
    category=""
):
    """
    Generates n-shot prompts for a single category

    format_dict: A dictionary mapping standard field names to the actual keys in the dataset.
                    For example:
                        {
                            "question": "question", 
                            "options": "choices", 
                            "category": "category", 
                            "cot_content": "cot_content"
                        }
    n_shots: The number of examples (shots) to include in the prompt for each category.
    categories: Either a single category (str) or a list of categories to build prompts for.
                If None, the function will use all unique categories found in `data` using the key provided in format_dict.

    Returns:
        Returns the prompt string
    """

    prompt = ""

    prompt += instruction_str + "\n"
    
    question_key = format_dict["question"]
    options_key = format_dict["options"]
    cot_key = format_dict["cot_content"]
    
    for i in range(n_shots):
        example = data[i]
        q_text = example[question_key]
        
        opts_list = example[options_key]
        labeled_opts = []
        for idx, opt in enumerate(opts_list):
            label = f"({ascii_uppercase[idx]})"
            labeled_opts.append(f"{label} {opt}")
        opts_str = ", ".join(labeled_opts)
        
        cot_content = example.get(cot_key, "")
        
        prompt += f"Question: {q_text}\n"
        prompt += f"Options: {opts_str}\n"
        if cot_content:
            prompt += f"{cot_content}\n\n"
        else:
            prompt += "\n"
                

    return prompt

def get_answer(model, tokenizer, prompt, max_tokens = 1000):

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_tokens)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

