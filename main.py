import time
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
from SAE import TopKSparseAutoencoder
import transformer_lens
from torch.cuda.amp import autocast
import os
import matplotlib.pyplot as plt
from tiny_story_data import load_tiny_stories_data



print("packages imported")
# Load a model (eg GPT-2 Small)
model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
print("model loaded")

# Check for MPS device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print(x)
else:
    device = torch.device("cpu")
    print("MPS device not found. Using CPU.")

model.to(device)


# Run the model and get logits and activations
logits, activations = model.run_with_cache(["this", "is", "a", "test"])

activation_key = f'blocks.{2}.hook_resid_post'

print(activations[activation_key].shape)


#check if data is loaded and tokenised, otherwise load it tokenise it and save it to a .pt file
if os.path.exists("train.pt"):
    train_tokens = torch.load('train_tokens.pt')
    val_tokens = torch.load('val_tokens.pt')
else:
    load_tiny_stories_data()
    train_tokens = torch.load('train_tokens.pt')
    val_tokens = torch.load('val_tokens.pt')

# Convert tokenized tensors to TensorDataset
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])



# DataLoader settings
batch_size = 32  # Set your desired batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define your SAE and training components
latent_dim = 768 * 2  # 8192
embedding_dim = 768  # Replace with actual embedding dim from the model

SAE = TopKSparseAutoencoder(input_dim=embedding_dim, latent_dim=latent_dim).to(device)
sae_optimizer = Adam(SAE.parameters(), lr=5e-4)

# Loss functions
mse_loss_fn = MSELoss()
l1_loss_fn = L1Loss()
max_grad_norm = 1.0  # Gradient clipping value

# Training SAE
num_epochs = 1
activation_key = f'blocks.{2}.hook_resid_post'  # Layer 2 residual stream



sae_losses = []



# Define hook function to capture activations
activation_key = f'blocks.{2}.hook_resid_post'
activations = None


def activation_hook(module, input, output):
    global activations
    activations = output


# Register hook
hook = model.get_submodule(activation_key).register_forward_hook(activation_hook)

# Training Loop
for epoch in range(num_epochs):
    SAE.train()
    cumulative_sae_recon_loss, cumulative_sae_sparsity_loss = 0, 0

    for batch_idx, (input_ids, attention_mask) in enumerate(train_loader):
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        sae_optimizer.zero_grad()

        # Model forward pass
        start_time = time.time()
        with torch.no_grad():
            _ = model(input_ids)  # Forward pass to trigger hook
        model_forward_time = time.time() - start_time

        # SAE forward pass
        start_time = time.time()
        with autocast():
            reconstructed, latent = SAE(activations)
            sae_forward_time = time.time() - start_time

            # Loss calculation
            start_time = time.time()
            sae_reconstruction_loss = mse_loss_fn(reconstructed, activations)
            sae_sparsity_loss = l1_loss_fn(latent, torch.zeros_like(latent))
            total_sae_loss = sae_reconstruction_loss
            loss_calc_time = time.time() - start_time

            # SAE backward pass
            start_time = time.time()
            total_sae_loss.backward()
            sae_backward_time = time.time() - start_time

            # Gradient clipping and optimizer step
            torch.nn.utils.clip_grad_norm_(SAE.parameters(), max_grad_norm)
            sae_optimizer.step()

        # Track and display batch losses and times
        cumulative_sae_recon_loss += sae_reconstruction_loss.item()
        cumulative_sae_sparsity_loss += sae_sparsity_loss.item()

        print(f"Batch {batch_idx + 1}/{len(train_loader)}: "
              f"Recon Loss={sae_reconstruction_loss.item():.4f}, "
              f"Sparsity Loss={sae_sparsity_loss.item():.4f}, "
              f"Model Forward Time={model_forward_time:.4f}s, "
              f"SAE Forward Time={sae_forward_time:.4f}s, "
              f"Loss Calc Time={loss_calc_time:.4f}s, "
              f"SAE Backward Time={sae_backward_time:.4f}s")
        sae_losses.append(sae_reconstruction_loss.item())
        if batch_idx % 100 == 0:
            plt.plot(sae_losses)
            plt.show()
    # Average epoch losses
    avg_recon_loss = cumulative_sae_recon_loss / len(train_loader)
    avg_sparsity_loss = cumulative_sae_sparsity_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Recon Loss: {avg_recon_loss:.4f}, Avg Sparsity Loss: {avg_sparsity_loss:.4f}")

print("SAE Training Complete!")

# Remove hook after training
hook.remove()





