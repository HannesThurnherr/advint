# %%

"""
python -m experiments.plot_ce_loss experiments/out/ce_loss.csv experiments/img/ce_loss.svg
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Path to the CSV file
csv_file_path = sys.argv[1]
out_path = sys.argv[2]

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract unique model and SAE combinations
model_sae_combinations = df[['Model', 'SAE']].drop_duplicates()

# Create subplots for loss, accuracy, reconstruction L2, and sparsity L1 in a 2x2 grid
scale = 1.3
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(5*scale, 3*scale))

# Define colors for each model and SAE combination
color_map = {
    ('base', 'base'): '#009900',
    ('adv', 'adv'): '#990000', 
    ('adv', 'post_adv'): '#0066CC'
}

num_models = len(model_sae_combinations)
width = 0.2  # Bar width

# Get unique metrics for consistent ordering
loss_metrics = ['lm_loss', 'e2e_loss']
accuracy_metrics = ['lm_acc', 'e2e_acc']
recon_metrics = ['recon_l2']
sparsity_metrics = ['sparsity_l1']
loss_indices = np.arange(len(loss_metrics))
accuracy_indices = np.arange(len(accuracy_metrics))
recon_indices = np.arange(len(recon_metrics))
sparsity_indices = np.arange(len(sparsity_metrics))

# Set for tracking added labels for legend
added_labels = set()

for i, (model_name, sae_name) in enumerate(model_sae_combinations.values):
    model_df = df[(df['Model'] == model_name) & (df['SAE'] == sae_name)]
    color = color_map.get((model_name, sae_name), '#000000')  # Default to black if not found
    
    # Only assign label if this SAE hasn't been added yet
    label = sae_name if sae_name not in added_labels else ""
    added_labels.add(sae_name)
    
    # Plot loss metrics
    loss_means = [model_df[metric].values[0] for metric in loss_metrics]
    loss_ci95s = [model_df[f"{metric}_se"].values[0] for metric in loss_metrics]
    bars = ax1.bar(loss_indices + i * width - (num_models * width) / 2 + width/2, 
                   loss_means, 
                   yerr=loss_ci95s, 
                   width=width, 
                   label=label,
                   capsize=5, 
                   color=color)
    for bar, mean, ci95 in zip(bars, loss_means, loss_ci95s):
        height = bar.get_height()
        if height < 0.1:
            ax1.text(bar.get_x() + bar.get_width() / 2, height + ci95, 
                     f'{mean:.2f}\n±{ci95:.2f}', 
                     ha='center', va='bottom', fontsize=8, rotation=90)
        else:
            ax1.text(bar.get_x() + bar.get_width() / 2, height / 2, 
                     f'{mean:.2f}\n±{ci95:.2f}', 
                     ha='center', va='center', fontsize=8, rotation=90, color='white')
    
    # Plot accuracy metrics
    accuracy_means = [model_df[metric].values[0] for metric in accuracy_metrics]
    accuracy_ci95s = [model_df[f"{metric}_se"].values[0] for metric in accuracy_metrics]
    bars = ax2.bar(accuracy_indices + i * width - (num_models * width) / 2 + width/2, 
                   accuracy_means, 
                   yerr=accuracy_ci95s, 
                   width=width, 
                   label=label,
                   capsize=5, 
                   color=color)
    for bar, mean, ci95 in zip(bars, accuracy_means, accuracy_ci95s):
        height = bar.get_height()
        if height < 0.1:
            ax2.text(bar.get_x() + bar.get_width() / 2, height + ci95, 
                     f'{mean:.2f}\n±{ci95:.2f}', 
                     ha='center', va='bottom', fontsize=8, rotation=90)
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2, height / 2, 
                     f'{mean:.2f}\n±{ci95:.2f}', 
                     ha='center', va='center', fontsize=8, rotation=90, color='white')
    
    # Plot reconstruction L2 metrics
    recon_means = [model_df[metric].values[0] for metric in recon_metrics]
    recon_ci95s = [model_df[f"{metric}_se"].values[0] for metric in recon_metrics]
    bars = ax3.bar(recon_indices + i * width - (num_models * width) / 2 + width/2, 
                   recon_means, 
                   yerr=recon_ci95s, 
                   width=width, 
                   label=label,
                   capsize=5, 
                   color=color)
    for bar, mean, ci95 in zip(bars, recon_means, recon_ci95s):
        height = bar.get_height()
        if height < 0.1:
            ax3.text(bar.get_x() + bar.get_width() / 2, height + ci95, 
                     f'{mean:.2f}\n±{ci95:.2f}', 
                     ha='center', va='bottom', fontsize=8, rotation=90)
        else:
            ax3.text(bar.get_x() + bar.get_width() / 2, height / 2, 
                     f'{mean:.2f}\n±{ci95:.2f}', 
                     ha='center', va='center', fontsize=8, rotation=90, color='white')
    
    # Plot sparsity L1 metrics
    sparsity_means = [model_df[metric].values[0] for metric in sparsity_metrics]
    sparsity_ci95s = [model_df[f"{metric}_se"].values[0] for metric in sparsity_metrics]
    bars = ax4.bar(sparsity_indices + i * width - (num_models * width) / 2 + width/2, 
                   sparsity_means, 
                   yerr=sparsity_ci95s, 
                   width=width, 
                   label=label,
                   capsize=5, 
                   color=color)
    for bar, mean, ci95 in zip(bars, sparsity_means, sparsity_ci95s):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, height / 2, 
                 f'{mean:.2f}\n±{ci95:.2f}', 
                 ha='center', va='center', fontsize=8, rotation=90, color='white')

# Configure loss plot
ax1.set_ylabel("Val Loss")
ax1.set_xticks(loss_indices)
ax1.set_xticklabels(["LM Loss", "E2E Loss"])
ax1.set_title("Validation Loss")

# Configure accuracy plot
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1)
ax2.set_xticks(accuracy_indices)
ax2.set_xticklabels(["LM Accuracy", "E2E Accuracy"])
ax2.set_title("Validation Accuracy")

# Configure reconstruction L2 plot
ax3.set_ylabel("L2 error")
ax3.set_xticks(recon_indices)
ax3.set_xticklabels([])
ax3.set_title("Reconstruction L2")

# Configure sparsity L1 plot
ax4.set_ylabel("L1 norm")
ax4.set_xticks(sparsity_indices)
ax4.set_xticklabels([])
ax4.set_title("Sparsity L1")

# Create a single legend for all subplots using SAE names
handles, _ = ax1.get_legend_handles_labels()  # Get handles from the first axis (ax1)
manual_labels = ["Base", "Adv.", "Post adv."]
fig.legend(handles, manual_labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3)
fig.tight_layout()
plt.savefig(out_path, bbox_inches='tight', format='svg')
plt.show()

# %%
