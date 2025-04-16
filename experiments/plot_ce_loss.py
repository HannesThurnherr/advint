# %%

"""
python -m experiments.plot_ce_loss experiments/out/ce_loss.csv experiments/img/ce_loss_v2.svg
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
# Import colormaps
import matplotlib.cm as cm


# Path to the CSV file
csv_file_path = sys.argv[1]
out_path = sys.argv[2]

# Read the CSV file
df = pd.read_csv(csv_file_path)

# Extract unique model and SAE combinations
model_sae_combinations = df[['Model', 'SAE']].drop_duplicates().values

# Create vertical grid layout
scale = 1
fig = plt.figure(figsize=(5 * scale, 8 * scale))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1])

ax1 = fig.add_subplot(gs[0, :])  # Loss (full row)
ax2 = fig.add_subplot(gs[1, :])  # Accuracy (full row)
ax3 = fig.add_subplot(gs[2, 0])  # Recon L2
ax4 = fig.add_subplot(gs[2, 1])  # Sparsity L1
ax5 = fig.add_subplot(gs[3, 0])  # R² Score (full row)

# --- Dynamic Color Map ---
# Define fixed colors
fixed_colors = {
    ('base', 'base'): '#009900',
    ('base', 'base_e2e'): '#00cc00',
}

# Identify combinations needing dynamic colors
dynamic_combinations = [
    tuple(combo) for combo in model_sae_combinations
    if tuple(combo) not in fixed_colors
]

# Generate dynamic colors using a palette (e.g., tab10)
num_dynamic = len(dynamic_combinations)
# Use tab10 colormap, ensuring enough colors
# If more than 10 dynamic colors are needed, consider a different map or cycling
dynamic_colors_list = [cm.tab10(i / 10) for i in range(num_dynamic)] # Use tab10, adjust if > 10 needed

# Create the final color map
color_map = fixed_colors.copy()
color_map.update(dict(zip(dynamic_combinations, dynamic_colors_list)))
# --- End Dynamic Color Map ---


num_models = len(model_sae_combinations)
width = 0.18
# Use a dictionary to store handles for the legend, mapping sae_name to handle
legend_map = {}

# Metrics to plot
metrics = {
    ax1: (['lm_loss', 'e2e_loss'], 'Val Loss', ['LM Loss', 'E2E Loss']),
    ax2: (['lm_acc', 'e2e_acc'], 'Accuracy', ['LM Accuracy', 'E2E Accuracy']),
    ax3: (['recon_l2'], 'L2 Error', ['Reconstruction L2']),
    ax4: (['sparsity_l1'], 'L1 Norm', ['Sparsity L1']),
    ax5: (['r2'], 'R² Score', ['Explained Variance'])
}

for i, (model_name, sae_name) in enumerate(model_sae_combinations):
    model_df = df[(df['Model'] == model_name) & (df['SAE'] == sae_name)]
    # Use the generated color_map
    color = color_map.get((model_name, sae_name), '#000000') # Default black if somehow missed
    
    # Assign label only if sae_name is new for the legend
    label_for_legend = sae_name if sae_name not in legend_map else None

    for ax, (metric_keys, ylabel, xticklabels) in metrics.items():
        indices = np.arange(len(metric_keys))
        # Ensure data exists before accessing .values[0]
        means = [model_df[metric].iloc[0] if not model_df.empty else np.nan for metric in metric_keys]
        ci95s = [model_df[f"{metric}_se"].iloc[0] if not model_df.empty else np.nan for metric in metric_keys]
        
        # Use label_for_legend for the bar's legend entry
        bars = ax.bar(indices + i * width - (num_models * width) / 2 + width/2,
                      means, yerr=ci95s, width=width, label=label_for_legend, capsize=5, color=color)

        # Store the handle for the legend if this sae_name is new
        if label_for_legend is not None and bars:
             legend_map[sae_name] = bars[0] # Store the first bar handle for this sae_name

        for bar, mean, ci95 in zip(bars, means, ci95s):
            # Skip text if mean or ci95 is NaN
            if np.isnan(mean) or np.isnan(ci95):
                continue
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

# --- Create dynamic legend ---
# Extract handles and labels from the collected map
legend_handles = list(legend_map.values())
legend_labels = list(legend_map.keys())

# Create the legend dynamically
fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=min(5, len(legend_labels))) # Adjust ncol dynamically
# --- End dynamic legend ---

fig.tight_layout()
plt.savefig(out_path, bbox_inches='tight', format='svg')
plt.show()


# %%
