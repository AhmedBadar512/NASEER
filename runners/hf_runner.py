# runners/hf_runner.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import evaluate

class HFModelRunner:
    def __init__(self, model_name, dataset_path, output_dir="checkpoints", max_length=512):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.dataset = load_from_disk(dataset_path)
        self.tokenized_dataset = self.dataset.map(self._tokenize, batched=True)

    def _tokenize(self, example):
        return self.tokenizer(example['text'], truncation=True, padding="max_length", max_length=self.max_length)

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

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset.get("validation"),
        )

        trainer.train()

    def evaluate(self):
        metric = evaluate.load("accuracy")
        predictions = []
        labels = []

        for example in self.tokenized_dataset["test"]:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model.generate(input_ids, max_length=self.max_length)
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(decoded)
            labels.append(example.get("label", ""))  # Use 'label' if available

        return metric.compute(predictions=predictions, references=labels)
