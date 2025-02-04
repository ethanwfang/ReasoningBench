from script import get_answer, generate_prompt_single, load_model
from datasets import load_dataset
import json


def redux(model_name, instruction_string, max_tokens):
    import json
    from string import ascii_uppercase
    from datasets import load_dataset
    
    names = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions"
    ]

    model, tokenizer = load_model(model_name)

    redux_ds_path = "edinburgh-dawg/mmlu-redux-2.0"

    results = {}

    # iterate through the dataset
    for dataset_name in names:
        ds = load_dataset(redux_ds_path, dataset_name)
        ds = ds["test"]  

        dataset_results = {}

        for idx, row in enumerate(ds):
            question = row["question"]
            choices = row["choices"]

            # Label choices as (A), (B), (C), ...
            labeled_opts = [f"({ascii_uppercase[i]}) {opt}" for i, opt in enumerate(choices)]
            opts_str = ", ".join(labeled_opts)

            # build our prompt: instruction_string + question + options
            prompt = (
                f"{instruction_string}"
                f"Question: {question}\n"
                f"Options: {opts_str}\n"
            )

            answer = get_answer(model, tokenizer, prompt, max_tokens)

            dataset_results[idx] = {
                "question": question,
                "answer": answer
            }

        results[dataset_name] = dataset_results
        print(dataset_name, "saved.")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


def redux_distributed(model_name, instruction_string, max_tokens):
    import json
    from string import ascii_uppercase
    from datasets import load_dataset
    import ray

    names = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions"
    ]

    model, tokenizer = load_model(model_name)

    redux_ds_path = "edinburgh-dawg/mmlu-redux-2.0"

    results = {}

    
    ray.init(ignore_reinit_error=True)
    BATCH_SIZE = 8 
    NUM_WORKERS = 4 

   
    @ray.remote
    class ModelWorker:
        def __init__(self, model_name):
            self.model, self.tokenizer = load_model(model_name)
        def process_prompts(self, prompts, max_tokens):
            answers = []
            for prompt in prompts:
                answer = get_answer(self.model, self.tokenizer, prompt, max_tokens)
                answers.append(answer)
            return answers

    workers = [ModelWorker.remote(model_name) for _ in range(NUM_WORKERS)]
    worker_index = 0  

    for dataset_name in names:
        ds = load_dataset(redux_ds_path, dataset_name)
        ds = ds["test"]

        dataset_results = {}

        batch_prompts = []
        batch_indices = []
        remote_results = []

        for idx, row in enumerate(ds):
            question = row["question"]
            choices = row["choices"]

            # Label choices as (A), (B), (C), ...
            labeled_opts = [f"({ascii_uppercase[i]}) {opt}" for i, opt in enumerate(choices)]
            opts_str = ", ".join(labeled_opts)

            # build our prompt: instruction_string + question + options
            prompt = (
                f"{instruction_string}"
                f"Question: {question}\n"
                f"Options: {opts_str}\n"
            )

            batch_prompts.append(prompt)
            batch_indices.append(idx)

           
            if len(batch_prompts) >= BATCH_SIZE:
                remote_result = workers[worker_index].process_prompts.remote(batch_prompts, max_tokens)
                remote_results.append((batch_indices, remote_result))
                batch_prompts = []
                batch_indices = []
                worker_index = (worker_index + 1) % NUM_WORKERS

        if batch_prompts:
            remote_result = workers[worker_index].process_prompts.remote(batch_prompts, max_tokens)
            remote_results.append((batch_indices, remote_result))
            worker_index = (worker_index + 1) % NUM_WORKERS

        # collect all results from the distributed workers.
        for indices, future in remote_results:
            answers = ray.get(future)
            for i, answer in zip(indices, answers):
                dataset_results[i] = {
                    "question": ds[i]["question"],
                    "answer": answer
                }

        results[dataset_name] = dataset_results
        print(dataset_name, "saved.")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    instruction_string = """The following is a multiple choice question. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice. \n"""
    max_tokens = 512
    redux_distributed(model_name, instruction_string, max_tokens)
