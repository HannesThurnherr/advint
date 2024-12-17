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
def compute_activation_statistics(activations, indices_of_interest):
    """
    Compute average magnitude and variance statistics for specific axes vs others
    Args:
        activations: tensor of shape [batch, seq_len, hidden_dim]
        indices_of_interest: indices of axes we want to analyze separately
    """
    # Convert to numpy for easier stats computation
    if torch.is_tensor(activations):
        activations = activations.detach().cpu().numpy()
    
    # Create mask for other indices
    all_indices = np.arange(activations.shape[-1])
    other_indices = np.array([i for i in all_indices if i not in indices_of_interest])
    
    # Compute statistics for interesting axes
    interesting_activations = activations[..., indices_of_interest]
    interesting_stats = {
        'mean_magnitude': np.mean(np.abs(interesting_activations)),
        'std_magnitude': np.std(np.abs(interesting_activations)),
        'mean_variance': np.mean(np.var(interesting_activations, axis=(0, 1))),
        'std_variance': np.std(np.var(interesting_activations, axis=(0, 1)))
    }
    
    # Compute statistics for other axes
    other_activations = activations[..., other_indices]
    other_stats = {
        'mean_magnitude': np.mean(np.abs(other_activations)),
        'std_magnitude': np.std(np.abs(other_activations)),
        'mean_variance': np.mean(np.var(other_activations, axis=(0, 1))),
        'std_variance': np.std(np.var(other_activations, axis=(0, 1)))
    }
    
    return interesting_stats, other_stats

# Storage for activation statistics
adv_stats = {'interesting': [], 'other': []}
normal_stats = {'interesting': [], 'other': []}

print("Computing activation statistics...")
with torch.no_grad():
    for batch_idx, (adv_act, non_adv_act) in enumerate(tqdm(activation_loader)):
        # Get statistics for both models
        adv_interesting, adv_other = compute_activation_statistics(adv_act, lowest_indices)
        normal_interesting, normal_other = compute_activation_statistics(non_adv_act, lowest_indices)
        
        # Store results
        adv_stats['interesting'].append(adv_interesting)
        adv_stats['other'].append(adv_other)
        normal_stats['interesting'].append(normal_interesting)
        normal_stats['other'].append(normal_other)
        
        if batch_idx > 100:  # Limit number of batches for memory efficiency
            break

# Aggregate statistics
def aggregate_stats(stats_list):
    """Compute mean and std of statistics across batches"""
    return {
        'mean_magnitude': np.mean([s['mean_magnitude'] for s in stats_list]),
        'std_magnitude': np.std([s['mean_magnitude'] for s in stats_list]),
        'mean_variance': np.mean([s['mean_variance'] for s in stats_list]),
        'std_variance': np.std([s['mean_variance'] for s in stats_list])
    }

# Compute final statistics
final_stats = {
    'adversarial': {
        'interesting': aggregate_stats(adv_stats['interesting']),
        'other': aggregate_stats(adv_stats['other'])
    },
    'normal': {
        'interesting': aggregate_stats(normal_stats['interesting']),
        'other': aggregate_stats(normal_stats['other'])
    }
}

# Print results
print("\nActivation Statistics Summary:")
for model_type in ['adversarial', 'normal']:
    print(f"\n{model_type.capitalize()} Model:")
    print("Interesting Axes:")
    stats = final_stats[model_type]['interesting']
    print(f"  Average magnitude: {stats['mean_magnitude']:.4f} ± {stats['std_magnitude']:.4f}")
    print(f"  Average variance: {stats['mean_variance']:.4f} ± {stats['std_variance']:.4f}")
    
    print("Other Axes:")
    stats = final_stats[model_type]['other']
    print(f"  Average magnitude: {stats['mean_magnitude']:.4f} ± {stats['std_magnitude']:.4f}")
    print(f"  Average variance: {stats['mean_variance']:.4f} ± {stats['std_variance']:.4f}")

# Create visualization of the statistics
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
models = ['Adversarial', 'Normal']
x = np.arange(len(models))
width = 0.35

interesting_magnitudes = [final_stats['adversarial']['interesting']['mean_magnitude'],
                        final_stats['normal']['interesting']['mean_magnitude']]
other_magnitudes = [final_stats['adversarial']['other']['mean_magnitude'],
                   final_stats['normal']['other']['mean_magnitude']]

plt.bar(x - width/2, interesting_magnitudes, width, label='Interesting Axes', color='blue', alpha=0.6)
plt.bar(x + width/2, other_magnitudes, width, label='Other Axes', color='red', alpha=0.6)
plt.ylabel('Average Magnitude')
plt.title('Activation Magnitudes')
plt.xticks(x, models)
plt.legend()

plt.subplot(1, 2, 2)
interesting_variances = [final_stats['adversarial']['interesting']['mean_variance'],
                        final_stats['normal']['interesting']['mean_variance']]
other_variances = [final_stats['adversarial']['other']['mean_variance'],
                   final_stats['normal']['other']['mean_variance']]

plt.bar(x - width/2, interesting_variances, width, label='Interesting Axes', color='blue', alpha=0.6)
plt.bar(x + width/2, other_variances, width, label='Other Axes', color='red', alpha=0.6)
plt.ylabel('Average Variance')
plt.title('Activation Variances')
plt.xticks(x, models)
plt.legend()

plt.tight_layout()
plt.show()

# %%
def collect_activations(model, loader, device, num_batches=100):
    activations = []
    
    # Define hook function
    def activation_hook(module, input, output):
        activations.append(output.detach())
    
    # Register hook
    activation_key = f'blocks.{2}.hook_resid_post'
    hook = model.get_submodule(activation_key).register_forward_hook(activation_hook)
    
    # Collect activations
    model.eval()
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask) in enumerate(tqdm(loader, desc="Collecting activations")):
            if batch_idx >= num_batches:
                break
            input_ids = input_ids.to(device)
            _ = model(input_ids)
    
    # Remove hook
    hook.remove()
    
    # Stack all activations
    return torch.cat(activations, dim=0)


val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Collect activations from both models
print("Collecting activations from normal model...")
normal_activations = collect_activations(unaugmented_model, val_loader, device)
print("Collecting activations from adversarial model...")
adv_activations = collect_activations(adversarial_model, val_loader, device)

# Compute differences
print("Computing differences...")
activation_differences = adv_activations - normal_activations

# Reshape for PCA
diff_reshaped = activation_differences.reshape(-1, activation_differences.shape[-1])
print(f"Activation differences shape: {diff_reshaped.shape}")

# Center the data
diff_mean = torch.mean(diff_reshaped, dim=0)
diff_centered = diff_reshaped - diff_mean

# Compute SVD (equivalent to PCA)
print("Computing SVD...")
U, S, V = torch.svd(diff_centered)

# Calculate explained variance ratios
explained_variance = (S ** 2) / (S ** 2).sum()
cumulative_variance = torch.cumsum(explained_variance, 0)

# Convert to numpy for plotting
explained_variance = explained_variance.cpu().numpy()
cumulative_variance = cumulative_variance.cpu().numpy()

# Plot explained variance ratio (log scale)
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, len(explained_variance) + 1), 
         explained_variance, 'b-', label='Individual')
plt.plot(np.arange(1, len(explained_variance) + 1),
         cumulative_variance, 'r-', label='Cumulative')
plt.yscale('log')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio (log scale)')
plt.title('PCA Components Importance in Activation Differences')
plt.legend()
plt.grid(True)
plt.show()

# Print summary statistics
num_components_90 = torch.where(torch.tensor(cumulative_variance) >= 0.9)[0][0].item() + 1
num_components_99 = torch.where(torch.tensor(cumulative_variance) >= 0.99)[0][0].item() + 1

print(f"\nSummary Statistics:")
print(f"Number of components explaining 90% of variance: {num_components_90}")
print(f"Number of components explaining 99% of variance: {num_components_99}")
print(f"Top 5 components explain {explained_variance[:5].sum()*100:.2f}% of variance")

# Plot histogram of component contributions
plt.figure(figsize=(12, 6))
plt.hist(explained_variance, bins=50, log=True)
plt.xlabel('Explained Variance Ratio')
plt.ylabel('Count (log scale)')
plt.title('Distribution of PCA Component Importance')
plt.grid(True)
plt.show()

# Additional visualization: Scree plot (elbow plot)
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, 51), explained_variance[:50], 'bo-')  # Plot first 50 components
plt.yscale('log')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio (log scale)')
plt.title('Scree Plot of Top 50 Components')
plt.grid(True)
plt.show()
# %%
# Given S is the singular values from the SVD
explained_variance = (S ** 2) / (S ** 2).sum()
variance_first_component = explained_variance[0]
print(f"First component explains {variance_first_component * 100:.2f}% of variance")
# %%
# Project reconstruction losses into PCA basis
def compute_pca_basis_losses(per_dimension_losses, V):
    """
    Project per-dimension losses into PCA basis
    Args:
        per_dimension_losses: tensor of shape [n_samples, n_dims]
        V: PCA basis vectors from SVD (V matrix)
    Returns:
        Losses in PCA basis
    """
    # Ensure we're working with the right shapes
    if torch.is_tensor(per_dimension_losses):
        losses = per_dimension_losses.cpu()
    else:
        losses = torch.tensor(per_dimension_losses)
        
    if torch.is_tensor(V):
        basis = V.cpu()
    else:
        basis = torch.tensor(V)
    
    # Project losses into PCA basis
    pca_basis_losses = torch.matmul(losses.unsqueeze(0), basis).squeeze(0)
    return pca_basis_losses

# Stack losses and compute mean for both models
avg_dim_loss_adv = torch.stack([r['per_dimension_loss'] for r in running_results['adversarial']]).mean(0).cpu()
avg_dim_loss_normal = torch.stack([r['per_dimension_loss'] for r in running_results['normal']]).mean(0).cpu()

# Project losses into PCA basis
pca_loss_adv = compute_pca_basis_losses(avg_dim_loss_adv, V)
pca_loss_normal = compute_pca_basis_losses(avg_dim_loss_normal, V)

# Create visualization
plt.figure(figsize=(12, 6))

# Plot losses in PCA basis
plt.plot(pca_loss_adv, label='Adversarial Model', alpha=0.6)
plt.plot(pca_loss_normal, label='Normal Model', alpha=0.6)

# Mark top 10 principal components
top_10_indices = torch.arange(10)
plt.vlines(top_10_indices, ymin=0, 
          ymax=max(pca_loss_adv.max(), pca_loss_normal.max()),
          colors='r', linestyles='dashed', alpha=0.3, 
          label='Top 10 Principal Components')

plt.yscale('log')
plt.xlabel('Principal Component Index')
plt.ylabel('Average Loss (in PCA basis)')
plt.title('Per-Component Reconstruction Loss Comparison (PCA Basis)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add explained variance annotation for first few components
for i in range(5):
    plt.text(i, plt.ylim()[0] * 1.1, 
             f'{explained_variance[i]*100:.1f}%', 
             rotation=45, 
             horizontalalignment='right')

plt.tight_layout()
plt.show()

# Print some statistics
print("\nLoss Statistics in PCA Basis:")
print(f"Adversarial Model:")
print(f"  Top 10 components average loss: {pca_loss_adv[:10].mean():.4f}")
print(f"  Other components average loss: {pca_loss_adv[10:].mean():.4f}")
print(f"Normal Model:")
print(f"  Top 10 components average loss: {pca_loss_normal[:10].mean():.4f}")
print(f"  Other components average loss: {pca_loss_normal[10:].mean():.4f}")
#%%
# Project reconstruction losses into PCA basis
# Project reconstruction losses into PCA basis
def compute_pca_basis_losses(per_dimension_losses, V):
    """
    Project per-dimension losses into PCA basis
    Args:
        per_dimension_losses: tensor of shape [n_samples, n_dims]
        V: PCA basis vectors from SVD (V matrix)
    Returns:
        Losses in PCA basis
    """
    # Ensure we're working with the right shapes
    if torch.is_tensor(per_dimension_losses):
        losses = per_dimension_losses.cpu()
    else:
        losses = torch.tensor(losses)
        
    if torch.is_tensor(V):
        basis = V.cpu()
    else:
        basis = torch.tensor(V)
    
    # Project losses into PCA basis
    pca_basis_losses = torch.matmul(losses.unsqueeze(0), basis).squeeze(0)
    return pca_basis_losses

# Stack losses and compute mean for both models
avg_dim_loss_adv = torch.stack([r['per_dimension_loss'] for r in running_results['adversarial']]).mean(0).cpu()
avg_dim_loss_normal = torch.stack([r['per_dimension_loss'] for r in running_results['normal']]).mean(0).cpu()

# Project losses into PCA basis
pca_loss_adv = compute_pca_basis_losses(avg_dim_loss_adv, V)
pca_loss_normal = compute_pca_basis_losses(avg_dim_loss_normal, V)

# Get sorting indices based on adversarial model's losses
sort_indices = torch.argsort(pca_loss_adv, descending=True)

# Sort everything
pca_loss_adv_sorted = pca_loss_adv[sort_indices]
pca_loss_normal_sorted = pca_loss_normal[sort_indices]
explained_variance_sorted = explained_variance[sort_indices]

# Create visualization
plt.figure(figsize=(15, 8))

# Plot sorted losses in PCA basis
x = np.arange(len(pca_loss_adv))
plt.plot(x, pca_loss_adv_sorted, label='Adversarial Model', alpha=0.8)
plt.plot(x, pca_loss_normal_sorted, label='Normal Model', alpha=0.8)

# Mark original top 10 PCA components
top_10_positions = torch.where(sort_indices < 10)[0]
plt.vlines(top_10_positions, ymin=0, 
          ymax=max(pca_loss_adv_sorted.max(), pca_loss_normal_sorted.max()),
          colors='r', linestyles='dashed', alpha=0.3, 
          label='Original Top 10 PCs')

plt.yscale('log')
plt.xlabel('Component Index (Sorted by Adversarial Model Loss)')
plt.ylabel('Average Loss (in PCA basis)')
plt.title('Per-Component Reconstruction Loss Comparison (PCA Basis)\nSorted by Adversarial Model Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Add explained variance annotation for original top PCs
for pos in top_10_positions[:5]:  # Annotate first 5 original PCs
    orig_idx = sort_indices[pos]
    plt.text(pos, plt.ylim()[0] * 1.1, 
             f'PC{orig_idx}\n{explained_variance[orig_idx]*100:.1f}%', 
             rotation=45, 
             horizontalalignment='right')

plt.tight_layout()
plt.show()

# Print statistics
print("\nLoss Statistics in Sorted PCA Basis:")
print(f"Adversarial Model:")
print(f"  Top 10 components average loss: {pca_loss_adv_sorted[:10].mean():.4f}")
print(f"  Other components average loss: {pca_loss_adv_sorted[10:].mean():.4f}")
print(f"Normal Model:")
print(f"  Top 10 components average loss: {pca_loss_normal_sorted[:10].mean():.4f}")
print(f"  Other components average loss: {pca_loss_normal_sorted[10:].mean():.4f}")

print("\nOriginal PCA Components in Top Losses:")
for i, idx in enumerate(sort_indices[:10]):
    print(f"Loss rank {i+1}: PC{idx} (explains {explained_variance[idx]*100:.2f}% variance)")

# Compute ratio of adversarial to normal loss for top components
ratios = pca_loss_adv_sorted[:10] / pca_loss_normal_sorted[:10]
print("\nRatio of Adversarial to Normal Loss for Top 10 Components:")
for i, ratio in enumerate(ratios):
    orig_idx = sort_indices[i]
    print(f"Component {i+1} (PC{orig_idx}): {ratio:.2f}x")
# %%
