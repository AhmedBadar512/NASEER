from transformers import GPTNeoModel, GPTNeoConfig

config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
model = GPTNeoModel(config)

if __name__ == "__main__":
    from transformers import GPTNeoModel, GPTNeoConfig
    # load baseline GPT‑Neo‑125M
    cfg = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
    model = GPTNeoModel(cfg)
    print("Baseline GPT‑Neo‑125M block attention modules:")
    print("Block\tAttentionLayer")
    for idx, block in enumerate(model.h):
        print(f"{idx}\t{block.attn.__class__.__name__}")
