import torch
from transformers import GPTNeoForCausalLM
from models.gpt_neo_naseer import load_naseer_gpt_neo

def run_fwd_bwd(model, input_ids):
    model.train()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    print(f"{model.__class__.__name__} loss: {loss.item():.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, seq_len = 2, 8

    # instantiate baseline
    baseline = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
    # instantiate NASEER‑ified GPT‑Neo
    naseer = load_naseer_gpt_neo().to(device)

    # dummy input
    vocab_size = baseline.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for model in (baseline, naseer):
        run_fwd_bwd(model, input_ids)

if __name__ == "__main__":
    main()
