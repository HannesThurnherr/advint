# %%
from tqdm import tqdm
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
from torch.cuda.amp import GradScaler

print("packages imported")

# %%

# Load model
model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
print("model loaded")
# %%
# Check for cuda device
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(1, device=device)
    print(x)
else:
    device = torch.device("cpu")
    print("Cuda device not found. Using CPU.")

model.to(device)


# Run the model and get logits and activations
logits, activations = model.run_with_cache(["this", "is", "a", "test"])

activation_key = f'blocks.{2}.hook_resid_post'

print(activations[activation_key].shape)
# %%

#check if data is loaded and tokenised, otherwise load it tokenise it and save it to a .pt file
if os.path.exists("train_tokens.pt"):
    train_tokens = torch.load('train_tokens.pt')
    val_tokens = torch.load('val_tokens.pt')
    print("loaded data from local")
else:
    print("tokenising data")
    load_tiny_stories_data()
    train_tokens = torch.load('train_tokens.pt')
    val_tokens = torch.load('val_tokens.pt')
    print("data loaded")


# %%
# Convert tokenized tensors to TensorDataset
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])



# DataLoader settings
# DataLoader settings
batch_size = 128  # Set your desired batch size
num_workers = 16  # Adjust this based on your system's CPU capacity
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

# Define your SAE and training components
latent_dim = 768 * 10  # 8192
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

# Initialize GradScaler
scaler = GradScaler()

sae_losses = []


# %%

# Define hook function to capture activations
activation_key = f'blocks.{2}.hook_resid_post'
activations = None


def activation_hook(module, input, output):
    global activations
    activations = output


# Register hook
hook = model.get_submodule(activation_key).register_forward_hook(activation_hook)

# %%
print("starting SAE training")
# Training Loop
# Training Loop
SAE.train()
model.eval()
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    
    cumulative_sae_recon_loss, cumulative_sae_sparsity_loss = 0, 0
    cumulative_batch_time = 0  # To accumulate batch processing times

    # Progress bar for each batch within the epoch
    for batch_idx, (input_ids, attention_mask) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}", total=len(train_loader), leave=False):
        # Model forward pass
        start_time = time.time()
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        sae_optimizer.zero_grad()
        
        with torch.no_grad():
            _ = model(input_ids)  # Forward pass to trigger hook

        # SAE forward pass
        # SAE forward pass
        with autocast():
            reconstructed, latent = SAE(activations)
            
            # Loss calculation
            sae_reconstruction_loss = mse_loss_fn(reconstructed, activations)
            sae_sparsity_loss = l1_loss_fn(latent, torch.zeros_like(latent))
            total_sae_loss = sae_reconstruction_loss

        # SAE backward pass with gradient scaling
        scaler.scale(total_sae_loss).backward()

        # Gradient clipping and optimizer step
        torch.nn.utils.clip_grad_norm_(SAE.parameters(), max_grad_norm)
        scaler.step(sae_optimizer)
        scaler.update()

        # Track and display batch losses
        cumulative_sae_recon_loss += sae_reconstruction_loss.item()
        cumulative_sae_sparsity_loss += sae_sparsity_loss.item()
        sae_losses.append(sae_reconstruction_loss.item())

        # Track batch time
        batch_time = time.time() - start_time
        cumulative_batch_time += batch_time

        # Print progress every 100 batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_time_per_batch = cumulative_batch_time / (batch_idx + 1)
            print(f"Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Recon Loss={sae_reconstruction_loss.item():.4f}, "
                  f"Sparsity Loss={sae_sparsity_loss.item():.4f}, "
                  f"Avg Time per Batch={avg_time_per_batch:.4f}s")
            plt.plot(sae_losses)
            plt.show()

    # Average epoch losses
    avg_recon_loss = cumulative_sae_recon_loss / len(train_loader)
    avg_sparsity_loss = cumulative_sae_sparsity_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - Avg Recon Loss: {avg_recon_loss:.4f}, Avg Sparsity Loss: {avg_sparsity_loss:.4f}")

print("SAE Training Complete!")

# Remove hook after training
hook.remove()





# %%
