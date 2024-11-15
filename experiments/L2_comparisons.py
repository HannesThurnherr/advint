# %%
import os
import sys

# Change directory to project root
os.chdir('/root/advint')
# Add the new working directory to sys.path
sys.path.append(os.getcwd())

# %%
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import torch.nn as nn
from SAE import TopKSparseAutoencoder
import transformer_lens
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from tiny_story_data import load_tiny_stories_data
from torch.cuda.amp import GradScaler
import torch.optim as optim
import torch.nn.utils as utils
import heapq  # Priority queue for efficient top-k tracking
import pickle  # For saving the results to a file
from transformer_lens import HookedTransformer

# %%
# Ensure we're using the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the activation key
activation_key = 'blocks.2.hook_resid_post'

# %%
# Load the validation dataset
val_tokens = torch.load("val_tokens.pt")  # Adjust the path if necessary
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# %%
def compute_average_L1_L0_L2_norms(model, SAE, data_loader, activation_key):
    model.to(device)
    SAE.to(device)
    model.eval()
    SAE.eval()
    total_L1_norm = 0.0
    total_L0_norm = 0.0
    total_L2_norm = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing Validation Set")):
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)

            # Run the model and get activations
            _, activations = model.run_with_cache(input_ids)

            # Get the activations at the desired layer
            activation = activations[activation_key]  # Shape: [batch_size, seq_len, resid_dim]

            # Pass through SAE encoder
            _, latent_activation = SAE(activation)  # Shape: [batch_size, seq_len, latent_dim]

            batch_size, seq_len, latent_dim = latent_activation.shape
            total_samples += batch_size * seq_len

            # Compute L1 norm per activation vector
            L1_norms_per_sample = torch.sum(torch.abs(latent_activation), dim=-1)  # Shape: [batch_size, seq_len]
            total_L1_norm += torch.sum(L1_norms_per_sample).item()

            # Compute L0 norm per activation vector
            L0_norms_per_sample = torch.count_nonzero(latent_activation, dim=-1)  # Shape: [batch_size, seq_len]
            total_L0_norm += torch.sum(L0_norms_per_sample).item()

            # Compute L2 norm per activation vector
            L2_norms_per_sample = torch.norm(latent_activation, p=2, dim=-1)  # Shape: [batch_size, seq_len]
            total_L2_norm += torch.sum(L2_norms_per_sample).item()
            if batch_idx>100:
                break

    average_L1_norm = total_L1_norm / total_samples
    average_L0_norm = total_L0_norm / total_samples
    average_L2_norm = total_L2_norm / total_samples

    return average_L1_norm, average_L0_norm, average_L2_norm


# %%
# Function to load model and SAE
def load_model_and_SAE(load_adv_model):
    """
    Loads the specified transformer model and its corresponding SAE.

    Args:
        load_adv_model (bool): If True, loads the adversarially trained model and SAE.
                               If False, loads the normal model and SAE.

    Returns:
        tuple: (model, SAE)
    """
    # Load the model
    model = HookedTransformer.from_pretrained("tiny-stories-33M")
    if load_adv_model:
        model.load_state_dict(torch.load("saved_models/symbiotically_trained_model.pth"))
    model.to(device)

    # Get activation dimensions
    _, activations = model.run_with_cache(["this", "is", "a", "test"])
    activation_key = f'blocks.{2}.hook_resid_post'
    resid_dim = activations[activation_key].shape[-1]
    latent_dim = resid_dim * 10  # Adjust as per your SAE configuration

    # Load the SAE
    SAE = TopKSparseAutoencoder(input_dim=resid_dim, latent_dim=latent_dim).to(device)
    if load_adv_model:
        SAE.load_state_dict(torch.load("saved_SAEs/adv_model_sae.pth"))
    else:
        SAE.load_state_dict(torch.load("saved_SAEs/model_sae.pth"))

    return model, SAE

# %%
# Load normal model and SAE
model_normal, SAE_normal = load_model_and_SAE(load_adv_model=False)

# Load adversarially trained model and SAE
model_adv, SAE_adv = load_model_and_SAE(load_adv_model=True)

# %%
# Compute the average L1, L0, and L2 norms for the normal model
avg_L1_norm_normal, avg_L0_norm_normal, avg_L2_norm_normal = compute_average_L1_L0_L2_norms(
    model_normal, SAE_normal, val_loader, activation_key
)

# Compute the average L1, L0, and L2 norms for the adversarially trained model
avg_L1_norm_adv, avg_L0_norm_adv, avg_L2_norm_adv = compute_average_L1_L0_L2_norms(
    model_adv, SAE_adv, val_loader, activation_key
)

# %%
# Print the results
print(f"Average L1 norm of SAE latent activations for the normal model: {avg_L1_norm_normal:.4f}")
print(f"Average L0 norm of SAE latent activations for the normal model: {avg_L0_norm_normal:.4f}")
print(f"Average L2 norm of SAE latent activations for the normal model: {avg_L2_norm_normal:.4f}\n")

print(f"Average L1 norm of SAE latent activations for the adversarially trained model: {avg_L1_norm_adv:.4f}")
print(f"Average L0 norm of SAE latent activations for the adversarially trained model: {avg_L0_norm_adv:.4f}")
print(f"Average L2 norm of SAE latent activations for the adversarially trained model: {avg_L2_norm_adv:.4f}")

# %%
