from torch import nn
import torch

class NASEERLayer(nn.Module):
    def __init__(self, hidden_size, num_basis_states=4, top_k=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_basis = num_basis_states
        self.top_k = top_k  # top_k interactions to keep (if set)
        self.phi = nn.Linear(hidden_size, hidden_size * num_basis_states)
        self.g = nn.Linear(hidden_size, num_basis_states)
        self.f = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.U = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        B, T, D = x.size()

        # Superposition: Represent each token as a weighted mixture of basis states.
        phi = self.phi(x).view(B, T, self.num_basis, D)
        alpha = torch.softmax(self.g(x), dim=-1).unsqueeze(-1)  # (B, T, num_basis, 1)
        x_super = (alpha * phi).sum(dim=2)  # (B, T, D)

        # Entangled update: Build symmetric interactions among tokens.
        x_i = x_super.unsqueeze(2).repeat(1, 1, T, 1)  # (B, T, T, D)
        x_j = x_super.unsqueeze(1).repeat(1, T, 1, 1)  # (B, T, T, D)
        interaction = self.f(torch.cat([x_i, x_j], dim=-1))  # (B, T, T, D)

        # Top-k selection: Limit the interactions to only the top-k tokens (if specified)
        if self.top_k is not None and self.top_k < T:
            # Compute a scalar score for each token pair using the L2 norm of the interaction.
            scores = interaction.norm(dim=-1)  # (B, T, T)
            # Select the top_k indices along the token j dimension for each token i.
            topk = torch.topk(scores, k=self.top_k, dim=-1)
            topk_indices = topk.indices  # (B, T, top_k)
            # Gather the top-k interaction vectors
            topk_interactions = torch.gather(
                interaction,
                dim=2,
                index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, D)
            )  # (B, T, top_k, D)
            # Sum only the top-k interactions.
            entangled = topk_interactions.sum(dim=2)  # (B, T, D)
        else:
            # Sum across all tokens if no top-k is specified.
            entangled = interaction.sum(dim=2)  # (B, T, D)

        x_entangled = x_super + entangled

        # Collapse: Use a soft collapse (in this case, simply the summed output).
        x_collapsed = x_entangled

        # Decoherence: Stabilize through a learned projection and normalization.
        x_final = self.norm(self.U(x_collapsed))

        return x_final


class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Use PyTorch's built-in MultiheadAttention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True  # This handles the permutation for us
        )
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # PyTorch's MultiheadAttention with batch_first=True
        # expects input of shape (batch_size, seq_len, hidden_size)
        attn_output, _ = self.self_attn(x, x, x)
        
        # Add & Norm (residual connection)
        return self.norm(x + attn_output)


if __name__ == "__main__":
    import torch
    import time
    import numpy as np
    import gc
    
    # Set parameters for testing
    batch_size = 4
    seq_length = 16
    hidden_size = 256
    num_basis_states = 8
    top_k = None  # Only consider top 8 interactions per token
    num_heads = 8  # For attention layer

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create both layers and move to device
    naseer_layer = NASEERLayer(hidden_size, num_basis_states, top_k=top_k).to(device)
    attn_layer = SelfAttentionLayer(hidden_size, num_heads=num_heads).to(device)

    # Create a random input tensor
    x = torch.randn(batch_size, seq_length, hidden_size).to(device)

    # Test shapes for both layers
    naseer_output = naseer_layer(x)
    attn_output = attn_layer(x)
    
    print(f"Input shape: {x.shape}")
    print(f"NASEER output shape: {naseer_output.shape}")
    print(f"Attention output shape: {attn_output.shape}")
    
    # Verify that shapes match
    assert naseer_output.shape == x.shape, "NASEER output shape doesn't match input!"
    assert attn_output.shape == x.shape, "Attention output shape doesn't match input!"
    print("Shape tests passed for both layers.\n")
    
    # Comparative performance testing
    print("Running comparative performance tests...")
    
    # Set up optimizers for both models
    naseer_optimizer = torch.optim.Adam(naseer_layer.parameters(), lr=0.001)
    attn_optimizer = torch.optim.Adam(attn_layer.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Tracking metrics
    naseer_times = []
    attn_times = []
    naseer_losses = []
    attn_losses = []
    
    num_iterations = 100
    
    # NASEER Layer test
    print("\nTesting NASEER Layer:")
    # Count number of trainable parameters
    num_params = sum(p.numel() for p in naseer_layer.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in NASEER Layer: {num_params}")
    torch.cuda.empty_cache()
    gc.collect()
    
    start_time = time.time()
    for i in range(num_iterations):
        # Generate new random input and target for each iteration
        x = torch.randn(batch_size, seq_length, hidden_size).to(device)
        target = torch.randn(batch_size, seq_length, hidden_size).to(device)
        
        # Reset gradients
        naseer_optimizer.zero_grad()
        
        # Time the forward and backward pass
        iter_start = time.time()
        
        # Forward pass
        output = naseer_layer(x)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        naseer_optimizer.step()
        
        iter_time = time.time() - iter_start
        naseer_times.append(iter_time)
        naseer_losses.append(loss.item())
        
        if (i + 1) % 20 == 0:
            print(f"Iteration {i + 1}/{num_iterations}, Loss: {loss.item():.6f}, Time: {iter_time:.4f}s")
    
    naseer_total_time = time.time() - start_time
    naseer_avg_time = np.mean(naseer_times)
    print(f"NASEER Layer - Total time: {naseer_total_time:.2f}s, Avg time per iteration: {naseer_avg_time:.4f}s")
    
    # Self-Attention Layer test
    print("\nTesting Self-Attention Layer:")
    # Count number of trainable parameters
    num_params = sum(p.numel() for p in attn_layer.parameters() if p.requires_grad)
    print(f"Attention Layer - Number of trainable parameters: {num_params}")
    torch.cuda.empty_cache()
    gc.collect()
    
    start_time = time.time()
    for i in range(num_iterations):
        # Generate new random input and target for each iteration
        x = torch.randn(batch_size, seq_length, hidden_size).to(device)
        target = torch.randn(batch_size, seq_length, hidden_size).to(device)
        
        # Reset gradients
        attn_optimizer.zero_grad()
        
        # Time the forward and backward pass
        iter_start = time.time()
        
        # Forward pass
        output = attn_layer(x)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        attn_optimizer.step()
        
        iter_time = time.time() - iter_start
        attn_times.append(iter_time)
        attn_losses.append(loss.item())
        
        if (i + 1) % 20 == 0:
            print(f"Iteration {i + 1}/{num_iterations}, Loss: {loss.item():.6f}, Time: {iter_time:.4f}s")
    
    attn_total_time = time.time() - start_time
    attn_avg_time = np.mean(attn_times)
    print(f"Attention Layer - Total time: {attn_total_time:.2f}s, Avg time per iteration: {attn_avg_time:.4f}s")
    
    # Comparison summary
    print("\nPerformance Comparison Summary:")
    print(f"NASEER Layer - Avg time: {naseer_avg_time:.4f}s, Final loss: {naseer_losses[-1]:.6f}")
    print(f"Attention Layer - Avg time: {attn_avg_time:.4f}s, Final loss: {attn_losses[-1]:.6f}")
    print(f"Speed ratio (Attention/NASEER): {attn_avg_time/naseer_avg_time:.2f}x")
    
    # Optional: Compare with larger sequence lengths if there's time and memory
    print("\nTest complete!")
