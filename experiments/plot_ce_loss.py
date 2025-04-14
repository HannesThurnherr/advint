# %%

"""
python -m experiments.plot_ce_loss experiments/out/ce_loss.csv experiments/img/ce_loss_v2.svg
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


# Path to the CSV file
csv_file_path = sys.argv[1]
out_path = sys.argv[2]

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract unique model and SAE combinations
model_sae_combinations = df[['Model', 'SAE']].drop_duplicates()

# Create vertical grid layout
scale = 1
fig = plt.figure(figsize=(5 * scale, 8 * scale))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])

ax1 = fig.add_subplot(gs[0, :])  # Loss (full row)
ax2 = fig.add_subplot(gs[1, :])  # Accuracy (full row)
ax3 = fig.add_subplot(gs[2, 0])  # Recon L2
ax4 = fig.add_subplot(gs[2, 1])  # Sparsity L1
ax5 = fig.add_subplot(gs[3, 0])  # R² Score (full row)

# Define colors
color_map = {
    ('base', 'base'): '#009900',
    ('base', 'base_e2e'): '#00cc00',
    ('adv', 'adv'): '#990000',
    ('adv', 'post_adv'): '#0066CC',
    ('adv', 'post_adv_e2e'): '#7dbeff' 
}

num_models = len(model_sae_combinations)
width = 0.18
added_labels = set()

# Metrics to plot
metrics = {
    ax1: (['lm_loss', 'e2e_loss'], 'Val Loss', ['LM Loss', 'E2E Loss']),
    ax2: (['lm_acc', 'e2e_acc'], 'Accuracy', ['LM Accuracy', 'E2E Accuracy']),
    ax3: (['recon_l2'], 'L2 Error', ['Reconstruction L2']),
    ax4: (['sparsity_l1'], 'L1 Norm', ['Sparsity L1']),
    ax5: (['r2'], 'R² Score', ['Explained Variance'])
}

for i, (model_name, sae_name) in enumerate(model_sae_combinations.values):
    model_df = df[(df['Model'] == model_name) & (df['SAE'] == sae_name)]
    color = color_map.get((model_name, sae_name), '#000000')
    label = sae_name if sae_name not in added_labels else ""
    added_labels.add(sae_name)
    
    for ax, (metric_keys, ylabel, xticklabels) in metrics.items():
        indices = np.arange(len(metric_keys))
        means = [model_df[metric].values[0] for metric in metric_keys]
        ci95s = [model_df[f"{metric}_se"].values[0] for metric in metric_keys]
        bars = ax.bar(indices + i * width - (num_models * width) / 2 + width/2, 
                      means, yerr=ci95s, width=width, label=label, capsize=5, color=color)
        for bar, mean, ci95 in zip(bars, means, ci95s):
            height = bar.get_height()
            if ax == ax3 and model_name == 'base':  # Special case for L2 error (base model)
                text_color = 'black'
                va_position = 'bottom'
                text_y_position = height + 0.01  # Move the text higher up
            else:
                text_color = 'white'
                va_position = 'center'
                text_y_position = height / 2
            ax.text(bar.get_x() + bar.get_width() / 2, text_y_position, 
                    f'{mean:.3f}\n±{ci95:.3f}', ha='center', va=va_position, fontsize=8, rotation=90, color=text_color)
        ax.set_ylabel(ylabel)
        ax.set_xticks(indices)
        ax.set_xticklabels(xticklabels)
        ax.set_title(ylabel)

# Adjust accuracy plot
ax2.set_ylim(0, 1)

# Create a single legend
handles, _ = ax1.get_legend_handles_labels()
fig.legend(handles, ["Base", "Base E2E", "Adv.", "Post Adv.", "Post Adv. E2E"], loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=5)

fig.tight_layout()
plt.savefig(out_path, bbox_inches='tight', format='svg')
plt.show()


# %%
