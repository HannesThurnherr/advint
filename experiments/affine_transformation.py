# %%
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
import transformer_lens
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
print(os.getcwd())
# %%
# Load the adversarially trained model
adversarial_model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
adversarial_model.load_state_dict(torch.load("saved_models/adversarially_trained_model.pth"))
print("adversarial_model loaded from checkpoint.")
unaugmented_model = HookedTransformer.from_pretrained("tiny-stories-33M")

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
adversarial_model.to(device)
unaugmented_model.to(device)

# Load the dataset (assuming Tiny Stories data tokens are pre-saved)
train_tokens = torch.load("train_tokens.pt")
val_tokens = torch.load("val_tokens.pt")
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

        if batch_idx > 500:
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
weight_matrix = affine_transform.linear.weight.detach().cpu().numpy()
optimizer = Adam(affine_transform.parameters(), lr=1e-3)
mse_loss = nn.MSELoss()
diff_total = None
# %%
# Training loop
epochs = 5
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

        # Calculate the absolute difference between activations
        if epoch == 0:
            abs_diff = torch.abs(reconstructed_non_adv - non_adv).detach().cpu()
            if diff_total == None:
                diff_total = abs_diff
            else:
                diff_total += abs_diff

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(activation_loader):.4f}")

print("Affine transformation training complete.")

# After training, check how well SAE features map between transformed activations
# transformed_activations = affine_transform(activation_adv_tensor).detach()
# You can now run your SAE on `transformed_activations` to see if the features are interpretable.
# %%


# Extract weights from the linear layer
# Extract weights from the linear layer (using torch)
weight_matrix = affine_transform.linear.weight.detach().cpu().numpy()  # Leave out  for now


# Plot using matplotlib
plt.figure(figsize=(6, 6))
plt.imshow(weight_matrix, cmap="viridis")
plt.colorbar(label="Weight magnitude")
plt.title("Affine Transformation Weight Matrix with Diagonal Set to 0")
plt.xlabel("Input features")
plt.ylabel("Output features")
plt.show()

# %%


# Extract weights from the linear layer
weight_matrix = affine_transform.linear.weight.detach().cpu().numpy()

# Plot the modified weight matrix with a 0-tolerant log
plt.figure(figsize=(6, 6))
l = np.abs(weight_matrix)
plt.imshow(np.log(l), cmap="viridis")
plt.colorbar(label="Log-Scaled Absolute Weight Magnitude")
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
weight_matrix = affine_transform.linear.weight.detach().cpu().numpy()
diagonal_elements = np.diag(weight_matrix)
print("Diagonal elements of the weight matrix:", diagonal_elements)

plt.figure(figsize=(8, 6))
plt.hist(diagonal_elements, bins=20, color="blue", edgecolor="black", alpha=0.7)
plt.xlabel("Weight value")
plt.ylabel("Frequency")
plt.yscale("log")
plt.title("Distribution of Affine Transformation Diagonal Weights")
plt.show()



# %%
print(diagonal_elements.shape)
# %%
indices = (diagonal_elements < 0.8).nonzero()
print(indices)
# %%
print("diff_total shape:", diff_total.shape)
avg1 = torch.mean(diff_total, dim = 0)
print("avg1 shape:", avg1.shape)
avg2 = torch.mean(avg1, dim = 0)
print("avg2 shape:", avg2.shape)
# %%
plt.figure(figsize=(8, 6))
plt.imshow(avg2.broadcast_to(100,768))
plt.show()
# %%
