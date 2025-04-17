# NASEER
Non-Attentive, Superposed, Entangled, Expressive Representation


**Non-Attentive**: The model foregoes the traditional attention mechanism.

**Superposed**: Each token is represented as a weighted combination of multiple latent basis states.

**Entangled**: Tokens interact via symmetric, bidirectional updates, reinforcing mutual relationships.

**Expressive Representation**: The result is a rich, robust representation that captures complex, holistic relationships.

## NASEER‑GPT‑Neo (125M)

Load a GPT‑Neo‑125M where all multihead‐attention blocks are replaced by NASEERLayer:

```python
from models.gpt_neo_naseer import load_naseer_gpt_neo

model = load_naseer_gpt_neo(
    pretrained_model="EleutherAI/gpt-neo-125M",
    entangle_method="gated",   # or "mlp"/"lowrank"
    top_k=8,
    rank=16
)
```
