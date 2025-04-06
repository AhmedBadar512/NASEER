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

    # Multitask understanding
    download_and_save("cais/mmlu")

    # QA
    download_and_save("natural_questions")
    download_and_save("mandarjoshi/trivia_qa")

    # Reasoning & NLI
    download_and_save("facebook/anli")
    download_and_save("hans")
    download_and_save("allenai/ai2_arc", config="ARC-Challenge")
    download_and_save("lukaemon/bbh")
    download_and_save("gsm8k")

if __name__ == "__main__":
    main()