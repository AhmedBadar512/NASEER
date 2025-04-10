from transformers import GPTNeoModel, GPTNeoConfig

config = GPTNeoConfig.from_pretrained("EleutherAI/gpt-neo-125M")
model = GPTNeoModel(config)
