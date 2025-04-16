import torch
from torch import nn
import time
import numpy as np

class NASEERLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=4,
        top_k=None,
        entangle_method='gated',
        rank=8
    ):
        """
        NASEER layer with head-split superposition and configurable entanglement.

        Args:
            hidden_size (int): Dimensionality of model (must be divisible by num_heads).
            num_heads (int): Number of heads for head-split superposition.
            top_k (int or None): Number of top interactions to keep (if set).
            entangle_method (str): One of ['gated','mlp','lowrank'].
            rank (int): Rank for low-rank MLP (if entangle_method='lowrank').
        """
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.top_k = top_k
        self.entangle_method = entangle_method
        self.rank = rank

        # Superposition: head-split basis
        # g: compute head weights
        self.g = nn.Linear(hidden_size, num_heads)
        # project back to hidden_size
        self.super_proj = nn.Linear(self.head_dim, hidden_size)

        # Entanglement methods
        if entangle_method == 'mlp':
            self.f = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )
        elif entangle_method == 'lowrank':
            self.f1 = nn.Linear(2 * hidden_size, rank)
            self.f2 = nn.Linear(rank, hidden_size)
        else:  # gated elementwise mixing
            self.gate_proj = nn.Linear(hidden_size, hidden_size)
            self.sigmoid = nn.Sigmoid()

        # Decoherence
        self.U = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, T, D)
        """
        B, T, D = x.size()
        H, hd = self.num_heads, self.head_dim

        # Head-split superposition
        # B, T, H, hd
        phi = x.view(B, T, H, hd)
        # B, T, H -> B, T, H, 1
        alpha = torch.softmax(self.g(x), dim=-1).unsqueeze(-1)
        # Weighted sum over heads -> B, T, hd
        x_super = (alpha * phi).sum(dim=2)
        # Project back to hidden_size -> B, T, D
        x_super = self.super_proj(x_super)

        # Prepare pairwise tokens for entanglement
        xi = x_super.unsqueeze(2).expand(-1, -1, T, -1)
        xj = x_super.unsqueeze(1).expand(-1, T, -1, -1)
        pair = torch.cat([xi, xj], dim=-1)  # B, T, T, 2D

        # Compute interactions
        if self.entangle_method == 'mlp':
            interaction = self.f(pair)
        elif self.entangle_method == 'lowrank':
            interaction = self.f2(torch.relu(self.f1(pair)))
        else:  # gated
            gate = self.sigmoid(self.gate_proj(x_super))  # B, T, D
            gi = gate.unsqueeze(2)  # B, T, 1, D
            gj = gate.unsqueeze(1)  # B, 1, T, D
            interaction = gi * gj  # B, T, T, D

        # Top-k selection
        if self.top_k is not None and self.top_k < T:
            scores = interaction.norm(dim=-1)  # B, T, T
            topk_idx = scores.topk(self.top_k, dim=-1).indices  # B, T, k
            idx = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
            interaction = torch.gather(interaction, 2, idx)

        # Sum over tokens -> B, T, D
        entangled = interaction.sum(dim=2)

        # Collapse & decoherence
        x_out = x_super + entangled
        x_final = self.norm(self.U(x_out))
        return x_final

# Example usage
if __name__ == "__main__":
    hidden_size = 256
    num_heads = 8
    top_k = 8
    batch_size = 8
    seq_length = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    naseer_layer = NASEERLayer(hidden_size=hidden_size, num_heads=num_heads).to(device)
    attn_layer = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads).to(device)

    num_iterations = 1500

    # side-by-side profiling of NASEERLayer vs PyTorch MultiheadAttention
    def profile_layer(layer):
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        times, losses = [], []
        for _ in range(num_iterations):
            x = torch.randn(batch_size, seq_length, hidden_size).to(device)
            target = torch.randn(batch_size, seq_length, hidden_size).to(device)
            optimizer.zero_grad()
            t0 = time.time()
            # handle nn.MultiheadAttention signature
            if isinstance(layer, nn.MultiheadAttention):
                out, _ = layer(x, x, x)
            else:
                out = layer(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            times.append(time.time() - t0)
            losses.append(loss.item())
        return np.mean(times), losses[-1]

    naseer_avg, naseer_final = profile_layer(naseer_layer)
    multi_avg,  multi_final  = profile_layer(attn_layer)

    print("\nProfiling Results:")
    print(f"{'Layer':<15}{'Avg Time(s)':<15}{'Final Loss':<15}")
    print(f"{'NASEER':<15}{naseer_avg:<15.4f}{naseer_final:<15.6f}")
    print(f"{'Multihead':<15}{multi_avg:<15.4f}{multi_final:<15.6f}")
