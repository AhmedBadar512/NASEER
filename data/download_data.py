import os
from datasets import load_dataset

def download_and_save(dataset_name, config=None, save_dir="data"):
    name = dataset_name if not config else f"{dataset_name}_{config}"
    dataset_dir = os.path.join(save_dir, name)
    if os.path.exists(dataset_dir):
        print(f"Skipping {name}: already downloaded")
        return

    print(f"Downloading {name}...")
    dataset = load_dataset(dataset_name, config) if config else load_dataset(dataset_name)
    dataset.save_to_disk(dataset_dir)
    print(f"Saved {name} to {dataset_dir}")

def main():
    os.makedirs("data", exist_ok=True)

    # Language modeling
    download_and_save("DKYoon/SlimPajama-6B")

    # MMLU selected subjects
    mmlu_subjects = [
        "philosophy",
        "machine_learning",
        "high_school_mathematics",
        "us_foreign_policy"
    ]
    for subject in mmlu_subjects:
        download_and_save("cais/mmlu", config=subject)

    # QA
    download_and_save("natural_questions")
    download_and_save("mandarjoshi/trivia_qa", config="rc.wikipedia")

    # Reasoning & NLI
    download_and_save("facebook/anli")
    download_and_save("hans")
    download_and_save("allenai/ai2_arc", config="ARC-Challenge")

    bbh_tasks = [
        "boolean_expressions",
        "multistep_arithmetic_two",
        "logical_deduction_three_objects",
        "web_of_lies"
    ]
    for task in bbh_tasks:
        download_and_save("lukaemon/bbh", config=task)

    download_and_save("gsm8k", config="main")

if __name__ == "__main__":
    main()
