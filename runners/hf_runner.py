import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from tqdm import tqdm
import re
from time import time
import os

# TODO: For hans add entailment vs non-entailment accuracy

class HFModelRunner:
    def __init__(self,
                 model_name, dataset_path,
                 output_dir="checkpoints", max_length=512, anli_round=None,
                 use_naseer=False, entangle_method='gated',
                 top_k=None, rank=8,
                 layer_hidden_size=None, layer_num_heads=None):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.max_length = max_length
        self.anli_round = anli_round
        self.use_naseer = use_naseer
        self.entangle_method = entangle_method
        self.top_k = top_k
        self.rank = rank
        self.layer_hidden_size = layer_hidden_size
        self.layer_num_heads = layer_num_heads

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_naseer:
            from models.gpt_neo_naseer import load_naseer_gpt_neo
            self.model = load_naseer_gpt_neo(
                pretrained_model=model_name,
                entangle_method=self.entangle_method,
                top_k=self.top_k,
                rank=self.rank,
                hidden_size=self.layer_hidden_size,
                num_heads=self.layer_num_heads
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.dataset = load_from_disk(dataset_path)
        
        # Check if this is TriviaQA dataset
        first_key = next(iter(self.dataset.keys()))
        if first_key in self.dataset and len(self.dataset[first_key]) > 0:
            sample = self.dataset[first_key][0]
            if "question" in sample and "question_id" in sample and "entity_pages" in sample:
                print("Detected TriviaQA dataset format")
        
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
        # Check for TriviaQA format
        if "question" in examples and "question_id" in examples and "entity_pages" in examples:
            # TriviaQA dataset
            inputs = []
            for question, entity_pages in zip(examples["question"], examples["entity_pages"]):
                context = ""
                # Extract context from entity_pages if available
                if entity_pages and "wiki_context" in entity_pages and entity_pages["wiki_context"]:
                    context = entity_pages["wiki_context"][0][:1000]  # Limit context size
                
                # Format the input as a QA prompt
                prompt = f"Question: {question}\n"
                if context:
                    prompt += f"Context: {context}\n"
                prompt += "Answer:"
                inputs.append(prompt)
        elif "question" in examples and "document" in examples and "annotations" in examples:
            # Natural Questions dataset
            inputs = []
            for question, document in zip(examples["question"], examples["document"]):
                # Create a prompt combining question and relevant document content
                # Truncating document to avoid excessive length
                doc_text = document.get('document_text', '')[:1000]  # Limit document size
                inputs.append(f"Question: {question}\nContext: {doc_text}\nAnswer:")
        elif "text" in examples and "meta" in examples and "__index_level_0__" in examples:
            # Slim Pajama dataset
            inputs = examples["text"]
        elif "text" in examples:
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

        for field in ["answerKey", "answer", "label", "annotations"]:
            if field in examples:
                tokenized[field] = examples[field]

        return tokenized

    def train(self,
              num_train_epochs=1,
              per_device_batch_size=4,
              report_to=None,
              save_every_n_epochs=None,
              logging_dir=None,
              learning_rate=None,
              lr_scheduler_type=None,
              warmup_steps=0):
        # prepare logging dir
        logging_dir = logging_dir or os.path.join(self.output_dir, "logs")

        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            num_train_epochs=num_train_epochs,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir=logging_dir,
            logging_steps=10,
            report_to=report_to or ["tensorboard"],
            learning_rate=learning_rate or 5e-5,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            # keep default save limit or adjust here
        )

        # build optional save-every-N-epochs callback
        callbacks = []
        if save_every_n_epochs and save_every_n_epochs > 0:
            class SaveEveryNEpochsCallback(TrainerCallback):
                def __init__(self, n, out_dir):
                    self.n = n; self.out_dir = out_dir
                def on_epoch_end(self, args, state, control, **kwargs):
                    epoch = int(state.epoch or 0)
                    if epoch and epoch % self.n == 0:
                        ckpt = os.path.join(self.out_dir, f"checkpoint-epoch-{epoch}")
                        kwargs["model"].save_pretrained(ckpt)
            callbacks.append(SaveEveryNEpochsCallback(save_every_n_epochs, self.output_dir))

        train_split = self.tokenized_dataset.get("train") or \
                      self.tokenized_dataset.get("dev") or \
                      next(iter(self.tokenized_dataset.values()))
        eval_split  = self.tokenized_dataset.get("validation") or \
                      self.tokenized_dataset.get("test") or \
                      self.tokenized_dataset.get("dev")

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_split,
            eval_dataset=eval_split,
            callbacks=callbacks or None,
        )

        # run training and return metrics
        train_output = trainer.train()
        return train_output.metrics

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

        # build torch tensors from the list-of-dicts
        input_ids = torch.tensor([ex["input_ids"] for ex in batch]).to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_length,
                use_cache=not self.use_naseer,
                num_beams=self.eval_beams if hasattr(self, "eval_beams") else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        for i, example in enumerate(batch):
            full_output = self.tokenizer.decode(outputs[i], skip_special_tokens=True).strip()
            
            # Check if this is TriviaQA format
            if "question_id" in example and "answer" in example and isinstance(example["answer"], dict):
                # For TriviaQA, extract the full answer (not just a letter)
                prediction = self._extract_triviaqa_answer(full_output)
                # Normalize both prediction and reference for TriviaQA
                prediction = prediction.strip().rstrip(".").lower()
                reference = example["answer"].get("value", "").strip().rstrip(".").lower()
                predictions.append(prediction)
                references.append(reference)
            else:
                # Handle other dataset formats as before
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

    def _extract_triviaqa_answer(self, text):
        """Extract answer from model output for TriviaQA format with normalization."""
        answer_match = re.search(r"Answer\s*:(.+)", text, re.IGNORECASE)
        if answer_match:
            ans = answer_match.group(1).strip()
            # Take only the first line if there is extra text
            ans = ans.split("\n")[0].strip()
            # Remove any trailing period
            ans = ans.rstrip(".")
            return ans
        # If no "Answer:" pattern, return the last sentence or phrase
        sentences = text.split(".")
        if sentences:
            ans = sentences[-1].strip().rstrip(".")
            return ans
        return text.strip()
