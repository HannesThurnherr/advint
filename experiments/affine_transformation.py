# %%
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
import transformer_lens
import numpy as np
# %%
# Load the adversarially trained model
adversarial_model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
adversarial_model.load_state_dict(torch.load("../saved_models/adversarially_trained_model.pth"))
print("adversarial_model loaded from checkpoint.")
unaugmented_model = HookedTransformer.from_pretrained("tiny-stories-33M")

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adversarial_model.to(device)
unaugmented_model.to(device)

# Load the dataset (assuming Tiny Stories data tokens are pre-saved)
train_tokens = torch.load("../train_tokens.pt")
val_tokens = torch.load("../val_tokens.pt")
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Set model to evaluation mode
adversarial_model.eval()
unaugmented_model.eval()

# Activation hook setup
activation_key = 'blocks.2.hook_resid_post'
activations_adv, activations_non_adv = [], []

# Hook function to capture activations
def get_activation_hook(activation_storage):
    def hook(module, input, output):
        activation_storage.append(output.detach().cpu())
    return hook

# Register hooks
adv_hook = adversarial_model.get_submodule(activation_key).register_forward_hook(get_activation_hook(activations_adv))
non_adv_hook = unaugmented_model.get_submodule(activation_key).register_forward_hook(get_activation_hook(activations_non_adv))
# %%
# Capture activations
print("Capturing activations for both models...")
with torch.no_grad():
    for batch_idx, batch in tqdm(enumerate(train_loader), desc="Processing batches"):
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)
        
        # Forward pass for adversarial model
        adversarial_model(input_ids)

        # Forward pass for unaugmented model
        unaugmented_model(input_ids)

        if batch_idx > 100:
            break
# %%
# Stack activations and align dataset
print("stacking tensors")
activation_adv_tensor = torch.cat(activations_adv, dim=0)
activation_non_adv_tensor = torch.cat(activations_non_adv, dim=0)
assert activation_adv_tensor.shape == activation_non_adv_tensor.shape, "Mismatch in activation shapes"
print("creating dataset")
# Create dataset
activation_dataset = TensorDataset(activation_adv_tensor, activation_non_adv_tensor)
activation_loader = DataLoader(activation_dataset, batch_size=16, shuffle=True, num_workers = 16)
print("Activation dataset created successfully!")

# Cleanup hooks
adv_hook.remove()
non_adv_hook.remove()
# %%
# Define affine transformation model (linear + bias)
class AffineTransform(nn.Module):
    def __init__(self, input_dim):
        super(AffineTransform, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.linear(x)

# Initialize affine transformation and optimizer
affine_transform = AffineTransform(input_dim=activation_adv_tensor.shape[-1]).to(device)
optimizer = Adam(affine_transform.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()
# %%
# Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    print(f"epoch {epoch}")
    for adv, non_adv in tqdm(activation_loader):
        adv, non_adv = adv.to(device), non_adv.to(device)
        
        # Forward pass
        reconstructed_non_adv = affine_transform(adv)
        loss = mse_loss(reconstructed_non_adv, non_adv)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(activation_loader):.4f}")

print("Affine transformation training complete.")

# After training, check how well SAE features map between transformed activations
# transformed_activations = affine_transform(activation_adv_tensor).detach()
# You can now run your SAE on `transformed_activations` to see if the features are interpretable.
# %%
import matplotlib.pyplot as plt

# Extract weights from the linear layer
weight_matrix = affine_transform.linear.weight.detach().cpu().numpy()

# Plot the weight matrix
plt.figure(figsize=(6, 6))
plt.imshow(weight_matrix, cmap="viridis")
plt.colorbar(label="Weight magnitude")
plt.title("Affine Transformation Weight Matrix")
plt.xlabel("Input features")
plt.ylabel("Output features")
plt.show()

# %%

import numpy as np

# Extract weights from the linear layer
weight_matrix = affine_transform.linear.weight.detach().cpu().numpy()

# Set the diagonal to 0
np.fill_diagonal(weight_matrix, 0)

# Plot the modified weight matrix
plt.figure(figsize=(6, 6))
plt.imshow(weight_matrix**4, cmap="viridis")
plt.colorbar(label="Weight magnitude")
plt.title("Affine Transformation Weight Matrix with Diagonal Set to 0")
plt.xlabel("Input features")
plt.ylabel("Output features")
plt.show()
# %%
# Flatten the weight matrix to get all weights in a 1D array
weights = weight_matrix.flatten()

# Plot the histogram of weights
plt.figure(figsize=(8, 6))
plt.hist(weights, bins=200, color="blue", edgecolor="black", alpha=0.7)
plt.xlabel("Weight value")
plt.ylabel("Frequency")
plt.yscale("log")
plt.title("Distribution of Affine Transformation Weights")
plt.show()
# %%






