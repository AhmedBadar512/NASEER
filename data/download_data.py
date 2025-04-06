#!/usr/bin/env python3

"""
Sample script to download common NLP benchmarks using Hugging Face datasets.
Make sure to install the datasets library first:

    pip install datasets

Then, you can run this script:

    python download_datasets.py

It will download and save each dataset to a local directory.
"""

import os
from datasets import load_dataset

def download_glue_tasks(save_dir="glue_data", tasks=None):
    """
    Download specified tasks from the GLUE benchmark.
    If tasks is None, downloads the entire GLUE suite.
    """
    if tasks is None:
        tasks = [
            "sst2", "mnli", "qqp", "qnli", "rte",
            "mrpc", "cola", "stsb", "wnli"
        ]
    os.makedirs(save_dir, exist_ok=True)

    for task in tasks:
        print(f"Downloading GLUE task: {task}")
        dataset = load_dataset("glue", task)
        task_dir = os.path.join(save_dir, task)
        dataset.save_to_disk(task_dir)
        print(f"Saved {task} to {task_dir}")

def download_squad(save_dir="squad_data", version="v1"):
    """
    Download SQuAD dataset (v1 or v2).
    """
    os.makedirs(save_dir, exist_ok=True)
    if version == "v1":
        dataset_name = "squad"
    elif version == "v2":
        dataset_name = "squad_v2"
    else:
        raise ValueError("Invalid SQuAD version. Use 'v1' or 'v2'.")

    print(f"Downloading SQuAD {version}...")
    dataset = load_dataset(dataset_name)
    dataset.save_to_disk(save_dir)
    print(f"Saved SQuAD {version} to {save_dir}")

def download_wikitext(save_dir="wikitext", version="wikitext-2"):
    """
    Download the WikiText language modeling datasets: wikitext-2 or wikitext-103
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading {version}...")
    dataset = load_dataset(version)
    dataset.save_to_disk(os.path.join(save_dir, version))
    print(f"Saved {version} to {os.path.join(save_dir, version)}")

def main():
    # Download GLUE tasks
    download_glue_tasks()

    # Download SQuAD v1
    download_squad(version="v2")

    # Download SQuAD v2 (optional, comment/uncomment as needed)
    # download_squad(save_dir="squad_v2_data", version="v2")

    # Download WikiText-2 (for language modeling)
    download_wikitext(version="wikitext-2")

    # Download WikiText-103 (much larger, also for language modeling)
    # download_wikitext(version="wikitext-103")

if __name__ == "__main__":
    main()
