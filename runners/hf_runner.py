# runners/hf_runner.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

class HFModelRunner:
    def __init__(self, model_name, dataset_path, output_dir="checkpoints", max_length=512):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # Set padding side to 'left' for decoder-only models
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Get a CUDA device if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.dataset = load_from_disk(dataset_path)
        if any(k.startswith("train_") for k in self.dataset.keys()):
            train_keys = [k for k in self.dataset.keys() if k.startswith("train")]
            eval_keys = [k for k in self.dataset.keys() if k.startswith("dev")]
            test_keys = [k for k in self.dataset.keys() if k.startswith("test")]

            train_ds = concatenate_datasets([self.dataset[k] for k in train_keys])
            eval_ds = concatenate_datasets([self.dataset[k] for k in eval_keys])
            test_ds = concatenate_datasets([self.dataset[k] for k in test_keys])

            self.dataset = DatasetDict({
                "train": train_ds,
                "validation": eval_ds,
                "test": test_ds
            })
        self.tokenized_dataset = self.dataset.map(self._tokenize, batched=True)

    def _tokenize(self, examples):
        if "text" in examples:
            inputs = examples["text"]

        elif "premise" in examples and "hypothesis" in examples:
            inputs = [f"Premise: {p} Hypothesis: {h}" for p, h in zip(examples["premise"], examples["hypothesis"])]

        elif "question" in examples and "answerKey" in examples:
            # ARC-style
            inputs = []
            for q in examples["question"]:
                if isinstance(q, dict):
                    stem = q.get("stem", "")
                    choices = " ".join([f"({c['label']}) {c['text']}" for c in q.get("choices", [])])
                    inputs.append(f"Q: {stem} {choices}")
                else:
                    inputs.append("Q: Unknown format")

        elif "question" in examples and "choices" in examples:
            # MMLU-style
            inputs = []
            for q, choice_list in zip(examples["question"], examples["choices"]):
                lettered = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choice_list)])
                # Make the task clearer with explicit instruction
                inputs.append(f"Q: {q}\nChoices: {lettered}\nAnswer: ")

        else:
            raise KeyError("Unrecognized format: no 'text', 'premise+hypothesis', ARC, or MMLU fields.")

        tokenized = self.tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def train(self, num_train_epochs=1, per_device_batch_size=4):
        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            num_train_epochs=num_train_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
        )

        train_split = self.tokenized_dataset.get("train") or \
              self.tokenized_dataset.get("dev") or \
              next(iter(self.tokenized_dataset.values()))

        eval_split = self.tokenized_dataset.get("validation") or \
                    self.tokenized_dataset.get("test") or \
                    self.tokenized_dataset.get("dev")

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_split,
            eval_dataset=eval_split,
        )

        trainer.train()

    def eval(self, batch_size=8):
        if "test" not in self.tokenized_dataset:
            raise ValueError("The dataset does not contain a 'test' split for evaluation.")

        print("Evaluating model on test set...")

        # Get the test split
        test_dataset = self.tokenized_dataset["test"]
        # If it's a Hugging Face Dataset, convert to a dict-of-lists.
        if hasattr(test_dataset, "to_dict"):
            test_dataset = test_dataset.to_dict()
        # If it's column-based (dict) then convert it into a list of dicts.
        if isinstance(test_dataset, dict):
            num_examples = len(test_dataset["input_ids"])
            test_dataset = [{k: v[i] for k, v in test_dataset.items()} for i in range(num_examples)]

        # Split the test dataset (now a list of dicts) into batches.
        batches = [test_dataset[i:i + batch_size] for i in range(0, len(test_dataset), batch_size)]

        predictions = []
        references = []

        # Evaluate each batch sequentially.
        for batch in tqdm(batches, desc="Evaluating"):
            # In case a batch is still in column format, convert it.
            if isinstance(batch, dict):
                num_examples = len(batch["input_ids"])
                batch = [{k: v[i] for k, v in batch.items()} for i in range(num_examples)]
            batch_results = self._evaluate_batch(batch)
            predictions.extend(batch_results["predictions"])
            references.extend(batch_results["references"])

        # Calculate accuracy.
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0

        print(f"[✓] Accuracy: {accuracy:.4f} ({correct}/{total})")
        return {"accuracy": accuracy}

    def _evaluate_batch(self, batch):
        """
        Evaluate a batch of examples.
        """
        predictions = []
        references = []

        # Ensure batch is a list of dictionaries with "input_ids"
        if not isinstance(batch, list) or not all(isinstance(example, dict) and "input_ids" in example for example in batch):
            raise ValueError("Batch is not properly tokenized or does not contain 'input_ids'.")

        # Prepare inputs for the batch
        input_ids = torch.tensor([example["input_ids"] for example in batch]).to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Generate outputs for the batch
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode predictions and extract references
        for i, example in enumerate(batch):
            full_output = self.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            prediction = self._extract_answer_letter(full_output)
            predictions.append(prediction)

            # Get reference answer
            if "answer" in example and isinstance(example["answer"], str):
                reference = example["answer"]
            elif "label" in example and isinstance(example["label"], int):
                reference = chr(65 + example["label"])  # Convert 0,1,2,3 to A,B,C,D
            else:
                reference = ""
            references.append(reference)

        return {"predictions": predictions, "references": references}

    def _extract_answer_letter(self, text):
        """Extract the first letter from the generated text that matches A, B, C, or D"""
        for char in text[:20]:  # Look at first 20 chars
            if char in "ABCD":
                return char
        return ""  # No valid answer found
