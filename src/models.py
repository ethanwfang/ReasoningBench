from script import get_answer, generate_prompt_single, load_model
from datasets import load_dataset
import json


def redux(model_name, instruction_string, max_tokens):
    import json
    from string import ascii_uppercase
    from datasets import load_dataset
    
    # Assuming you have these helper functions or imports available:
    #  - load_model(model_name) -> returns (model, tokenizer)
    #  - get_answer(model, tokenizer, prompt, max_length) -> returns a string answer

    # List of dataset names
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

    # Load model and tokenizer using the provided model_name
    model, tokenizer = load_model(model_name)

    # The name of your dataset on Hugging Face
    redux_ds_path = "edinburgh-dawg/mmlu-redux-2.0"

    # Dictionary to hold overall results
    results = {}

    # Iterate through each dataset in names
    for dataset_name in names:
        ds = load_dataset(redux_ds_path, dataset_name)
        ds = ds["test"]  # Use the test split

        # Dictionary to hold the results for the current dataset
        dataset_results = {}

        # Process each row in the test set
        for idx, row in enumerate(ds):
            question = row["question"]
            choices = row["choices"]

            # Label choices as (A), (B), (C), ...
            labeled_opts = [f"({ascii_uppercase[i]}) {opt}" for i, opt in enumerate(choices)]
            opts_str = ", ".join(labeled_opts)

            # Build our prompt: instruction_string + question + options
            prompt = (
                f"{instruction_string}"
                f"Question: {question}\n"
                f"Options: {opts_str}\n"
            )

            # Get model answer with the specified max_tokens
            answer = get_answer(model, tokenizer, prompt, max_tokens)

            # Save the result
            dataset_results[idx] = {
                "question": question,
                "answer": answer
            }

        # Save the dataset's results into the global results dictionary
        results[dataset_name] = dataset_results
        print(dataset_name, "saved.")

    # Write all results to a JSON file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
