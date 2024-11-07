# %%
import torch
import torch.nn as nn


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, k = 25):
        super(TopKSparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()  # ReLU for non-linearity
        )
        self.decoder = nn.Linear(latent_dim, input_dim)  # Decoder without activation
        self.k = k  # Keep the top-k activations

    def forward(self, x):
        # Encode the input
        latent = self.encoder(x)

        # Zero out all but the top-k activations
        top_k_values, _ = torch.topk(latent, self.k, dim=-1)
        mask = latent >= top_k_values[..., -1].unsqueeze(-1)
        latent_k_sparse = latent * mask.float()
        # Decode the sparsified latent representation
        reconstructed = self.decoder(latent_k_sparse)
        return reconstructed, latent_k_sparse

# %%
