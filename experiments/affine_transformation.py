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
import sys
os.chdir('/root/advint')
# Add the new working directory to sys.path
sys.path.append(os.getcwd())
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
lowest_indices = np.argsort(diagonal_elements)[:10]
lowest_values = diagonal_elements[lowest_indices]

print("\nIndices and values of 10 lowest diagonal elements:")
for idx, val in zip(lowest_indices, lowest_values):
    print(f"Index {idx}: {val:.4f}")

# Use these indices for our analysis
indices = lowest_indices

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
# Load the SAE models
# Load the SAE models
from SAE import TopKSparseAutoencoder
import numpy as np

# Initialize SAEs
resid_dim = 768  # Based on your model's residual dimension
latent_dim = resid_dim * 10  # 8192 as in your original code

adv_SAE = TopKSparseAutoencoder(input_dim=resid_dim, latent_dim=latent_dim).to(device)
normal_SAE = TopKSparseAutoencoder(input_dim=resid_dim, latent_dim=latent_dim).to(device)

# Load saved states
adv_SAE.load_state_dict(torch.load("saved_SAEs/adv_model_sae.pth"))
normal_SAE.load_state_dict(torch.load("saved_SAEs/model_sae.pth"))

def compute_axis_specific_losses(activations, sae_model, indices):
    """Compute reconstruction loss specifically for interesting axes vs others"""
    with torch.no_grad():
        reconstructed, _ = sae_model(activations)
        
        # Compute per-dimension squared error
        squared_error = (activations - reconstructed) ** 2
        
        # Mean across batch and sequence dimensions
        dimension_loss = torch.mean(squared_error, dim=[0, 1])
        
        # Handle tuple output from nonzero()
        if isinstance(indices, tuple):
            interesting_indices = indices[0]
        else:
            interesting_indices = indices
        
        # Create complement set for other indices
        all_indices = np.arange(activations.shape[-1])
        other_indices = np.array([i for i in all_indices if i not in interesting_indices])
            
        # Create mask for other indices
        all_indices = np.arange(activations.shape[-1])
        other_indices = np.array([i for i in all_indices if i not in interesting_indices])
        
        # Compute losses
        interesting_loss = torch.mean(dimension_loss[interesting_indices])
        other_loss = torch.mean(dimension_loss[other_indices]) if len(other_indices) > 0 else torch.tensor(0.0)
        
        # Compute total loss and contributions
        total_loss = torch.mean(dimension_loss)
        
        # Compute weighted contributions based on number of dimensions
        n_total = activations.shape[-1]
        n_interesting = len(interesting_indices)
        n_other = n_total - n_interesting
        
        interesting_contribution = interesting_loss * (n_interesting / n_total)
        other_contribution = other_loss * (n_other / n_total)
        
        return {
            'interesting_axes_loss': interesting_loss.item(),
            'other_axes_loss': other_loss.item(),
            'total_loss': total_loss.item(),
            'interesting_contribution': interesting_contribution.item(),
            'other_contribution': other_contribution.item(),
            'per_dimension_loss': dimension_loss,
            'n_interesting': n_interesting,
            'n_other': n_other
        }

# Analysis loop
print("Computing losses for both models...")
running_results = {
    'adversarial': [],
    'normal': []
}

with torch.no_grad():
    for batch_idx, (adv_act, non_adv_act) in enumerate(tqdm(activation_loader)):
        adv_act = adv_act.to(device)
        non_adv_act = non_adv_act.to(device)
        
        # Compute losses for both models
        adv_results = compute_axis_specific_losses(adv_act, adv_SAE, indices)
        normal_results = compute_axis_specific_losses(non_adv_act, normal_SAE, indices)
        
        running_results['adversarial'].append(adv_results)
        running_results['normal'].append(normal_results)
        
        if batch_idx > 100:  # Limit number of batches for memory efficiency
            break

# Compute statistics
all_results = {
    'adversarial': {'avg': {}, 'std': {}},
    'normal': {'avg': {}, 'std': {}}
}

for model_type in ['adversarial', 'normal']:
    for metric in ['interesting_axes_loss', 'other_axes_loss', 'interesting_contribution', 'other_contribution']:
        values = [r[metric] for r in running_results[model_type]]
        all_results[model_type]['avg'][metric] = np.mean(values)
        all_results[model_type]['std'][metric] = np.std(values)

# Print results
# Print diagnostic information
print("\nDiagnostic Information:")
print(f"Type of indices: {type(indices)}")
if isinstance(indices, tuple):
    print(f"Shape of indices[0]: {indices[0].shape}")
    print(f"Number of indices where diagonal < 0.8: {len(indices[0])}")
else:
    print(f"Shape of indices: {indices.shape}")
    print(f"Number of indices: {len(indices)}")

print("\nResults Summary:")
for model_type in ['adversarial', 'normal']:
    print(f"\n{model_type.capitalize()} Model:")
    n_interesting = running_results[model_type][0]['n_interesting']
    n_other = running_results[model_type][0]['n_other']
    print(f"Number of interesting axes: {n_interesting}")
    print(f"Number of other axes: {n_other}")
    print(f"Average loss in interesting axes: {all_results[model_type]['avg']['interesting_axes_loss']:.4f} ± {all_results[model_type]['std']['interesting_axes_loss']:.4f}")
    print(f"Average loss in other axes: {all_results[model_type]['avg']['other_axes_loss']:.4f} ± {all_results[model_type]['std']['other_axes_loss']:.4f}")
    print(f"Contribution of interesting axes: {all_results[model_type]['avg']['interesting_contribution']:.4f} ± {all_results[model_type]['std']['interesting_contribution']:.4f}")
    print(f"Contribution of other axes: {all_results[model_type]['avg']['other_contribution']:.4f} ± {all_results[model_type]['std']['other_contribution']:.4f}")

# Visualization of per-dimension losses
plt.figure(figsize=(12, 6))
avg_dim_loss_adv = torch.stack([r['per_dimension_loss'] for r in running_results['adversarial']]).mean(0).cpu()
avg_dim_loss_normal = torch.stack([r['per_dimension_loss'] for r in running_results['normal']]).mean(0).cpu()

plt.plot(avg_dim_loss_adv, label='Adversarial Model', alpha=0.6)
plt.plot(avg_dim_loss_normal, label='Normal Model', alpha=0.6)

# Convert indices to proper format for visualization
if isinstance(indices, tuple):
    viz_indices = indices[0]
else:
    viz_indices = indices

plt.vlines(viz_indices, ymin=0, ymax=max(avg_dim_loss_adv.max(), avg_dim_loss_normal.max()),
           colors='r', linestyles='dashed', alpha=0.3, label='Interesting Axes')
plt.yscale('log')
plt.xlabel('Dimension')
plt.ylabel('Average Loss')
plt.title('Per-Dimension Reconstruction Loss Comparison')
plt.legend()
plt.show()
 # %%
