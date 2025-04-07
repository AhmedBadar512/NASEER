# runners/hf_runner.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from tqdm import tqdm
import re
from time import time

# TODO: For hans add entailment vs non-entailment accuracy

class HFModelRunner:
    def __init__(self, model_name, dataset_path, output_dir="checkpoints", max_length=512, anli_round=None):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_length = max_length
        self.anli_round = anli_round

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.dataset = load_from_disk(dataset_path)
        if "anli" in dataset_path.lower():
            self._load_anli_dataset()
        elif any(k.startswith("train_") for k in self.dataset.keys()):
            train_keys = [k for k in self.dataset.keys() if k.startswith("train")]
            eval_keys = [k for k in self.dataset.keys() if k.startswith("dev")]
            test_keys = [k for k in self.dataset.keys() if k.startswith("test")]
            self.dataset = DatasetDict({
                "train": concatenate_datasets([self.dataset[k] for k in train_keys]),
                "validation": concatenate_datasets([self.dataset[k] for k in eval_keys]),
                "test": concatenate_datasets([self.dataset[k] for k in test_keys])
            })

        self.tokenized_dataset = self.dataset.map(self._tokenize, batched=True)

    def _load_anli_dataset(self):
        rounds = ["r1", "r2", "r3"]
        if self.anli_round and self.anli_round in rounds:
            print(f"Loading ANLI dataset for round: {self.anli_round}")
            start_time = time()
            self.dataset = DatasetDict({
                "train": self.dataset[f"train_{self.anli_round}"],
                "validation": self.dataset[f"dev_{self.anli_round}"],
                "test": self.dataset[f"test_{self.anli_round}"]
            })
            print(f"Loaded ANLI round {self.anli_round} in {time() - start_time:.2f} seconds.")
        else:
            print("Loading ANLI dataset sequentially for rounds: r1, r2, r3")
            start_time = time()
            self.dataset = DatasetDict({
                "train": concatenate_datasets([self.dataset[f"train_{r}"] for r in rounds]),
                "validation": concatenate_datasets([self.dataset[f"dev_{r}"] for r in rounds]),
                "test": concatenate_datasets([self.dataset[f"test_{r}"] for r in rounds])
            })
            print(f"Loaded ANLI sequentially in {time() - start_time:.2f} seconds.")

    def _tokenize(self, examples):
        if "text" in examples:
            inputs = examples["text"]
        elif "premise" in examples and "hypothesis" in examples:
            inputs = [f"Premise: {p}\nHypothesis: {h}\nAnswer:" for p, h in zip(examples["premise"], examples["hypothesis"])]
        elif "question" in examples and isinstance(examples["choices"], list) and isinstance(examples["choices"][0], list):
            inputs = []
            for question, choices in zip(examples["question"], examples["choices"]):
                formatted = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
                prompt = f"Q: {question}\n{formatted}\nAnswer:"
                inputs.append(prompt)
        elif "question" in examples and "choices" in examples and "answerKey" in examples:
            inputs = []
            for question, choices in zip(examples["question"], examples["choices"]):
                labels = choices.get("label", [])
                texts = choices.get("text", [])
                formatted = "\n".join([f"({label}) {text}" for label, text in zip(labels, texts)])
                prompt = f"Q: {question}\n{formatted}\nAnswer:"
                inputs.append(prompt)
        elif "question" in examples and "choices" in examples:
            inputs = []
            for q, choice_list in zip(examples["question"], examples["choices"]):
                lettered = " ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choice_list)])
                inputs.append(f"Q: {q}\nChoices: {lettered}\nAnswer: ")
        elif "question" in examples and "answer" in examples:
            inputs = [f"Q: {q}\nAnswer:" for q in examples["question"]]
        else:
            raise KeyError("Unrecognized format: no supported fields found.")

        tokenized = self.tokenizer(
            inputs,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()

        for field in ["answerKey", "answer", "label"]:
            if field in examples:
                tokenized[field] = examples[field]

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
        print("Evaluating model on evaluation split...")
        eval_split = self.tokenized_dataset.get("validation") or \
                     self.tokenized_dataset.get("test") or \
                     self.tokenized_dataset.get("dev")

        if hasattr(eval_split, "to_dict"):
            eval_split = eval_split.to_dict()
        if isinstance(eval_split, dict):
            num_examples = len(eval_split["input_ids"])
            eval_split = [{k: v[i] for k, v in eval_split.items()} for i in range(num_examples)]

        batches = [eval_split[i:i + batch_size] for i in range(0, len(eval_split), batch_size)]

        predictions = []
        references = []

        for batch in tqdm(batches, desc="Evaluating"):
            batch_results = self._evaluate_batch(batch)
            predictions.extend(batch_results["predictions"])
            references.extend(batch_results["references"])

        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0

        print(f"[\u2713] Accuracy: {accuracy:.4f} ({correct}/{total})")
        return {"accuracy": accuracy}

    def _evaluate_batch(self, batch):
        predictions = []
        references = []

        input_ids = torch.tensor([example["input_ids"] for example in batch]).to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        for i, example in enumerate(batch):
            full_output = self.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            prediction = self._extract_answer_letter(full_output)
            predictions.append(prediction)

            if "answerKey" in example:
                reference = example["answerKey"].strip().upper()
            elif "answer" in example and isinstance(example["answer"], str):
                reference = example["answer"].strip().upper()
            elif "answer" in example and isinstance(example["answer"], int):
                reference = chr(65 + example["answer"])
            elif "label" in example and isinstance(example["label"], int):
                reference = str(example["label"])
            else:
                reference = ""

            references.append(reference)

        return {"predictions": predictions, "references": references}

    def _extract_answer_letter(self, text):
        match = re.search(r"Answer\s*[:\-]?\s*([ABCD01])", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        for token in text.split():
            token = token.strip("():.").upper()
            if token in ["A", "B", "C", "D", "0", "1"]:
                return token

        return ""