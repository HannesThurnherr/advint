# %%
"""
python -m experiments.plot_steering_vector
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Load the JSON data from file
with open('experiments/out/steering_vector.json', 'r') as file:
    data = json.load(file)

# Extract data
tokens = data["normal_model"]["positive_tokens"] + data["normal_model"]["negative_tokens"]
clean_changes = data["normal_model"]["positive_log_prob_changes"] + data["normal_model"]["negative_log_prob_changes"]
adversarial_changes = data["adversarial_model"]["positive_log_prob_changes"] + data["adversarial_model"]["negative_log_prob_changes"]

# Compute average positive and negative changes and 95% confidence intervals
clean_positive = [x for x in clean_changes if x > 0]
clean_negative = [x for x in clean_changes if x < 0]
adversarial_positive = [x for x in adversarial_changes if x > 0]
adversarial_negative = [x for x in adversarial_changes if x < 0]

mean_clean_positive = np.mean(clean_positive) if clean_positive else 0
mean_clean_negative = np.mean(clean_negative) if clean_negative else 0
mean_adv_positive = np.mean(adversarial_positive) if adversarial_positive else 0
mean_adv_negative = np.mean(adversarial_negative) if adversarial_negative else 0

ci95_clean_positive = stats.t.ppf(0.975, len(clean_positive) - 1) * stats.sem(clean_positive) if len(clean_positive) > 1 else 0
ci95_clean_negative = stats.t.ppf(0.975, len(clean_negative) - 1) * stats.sem(clean_negative) if len(clean_negative) > 1 else 0
ci95_adv_positive = stats.t.ppf(0.975, len(adversarial_positive) - 1) * stats.sem(adversarial_positive) if len(adversarial_positive) > 1 else 0
ci95_adv_negative = stats.t.ppf(0.975, len(adversarial_negative) - 1) * stats.sem(adversarial_negative) if len(adversarial_negative) > 1 else 0

# Plotting
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(tokens) + 2)  # Extra space for averages
width = 0.4

bars1 = ax.bar(x[:len(tokens)] - width/2, clean_changes, width, label='Base', color='#009900')
bars2 = ax.bar(x[:len(tokens)] + width/2, adversarial_changes, width, label='Adv.', color='#990000')

# Add average positive and negative change bars next to each other
pos_index = len(tokens)
neg_index = len(tokens) + 1

ax.errorbar(pos_index - width/2, mean_clean_positive, yerr=ci95_clean_positive, fmt='o', color='black')
ax.bar(pos_index - width/2, mean_clean_positive, width, color='#009900', alpha=0.5)
ax.errorbar(pos_index + width/2, mean_adv_positive, yerr=ci95_adv_positive, fmt='o', color='black')
ax.bar(pos_index + width/2, mean_adv_positive, width, color='#990000', alpha=0.5)

ax.errorbar(neg_index - width/2, mean_clean_negative, yerr=ci95_clean_negative, fmt='o', color='black')
ax.bar(neg_index - width/2, mean_clean_negative, width, color='#009900', alpha=0.5)
ax.errorbar(neg_index + width/2, mean_adv_negative, yerr=ci95_adv_negative, fmt='o', color='black')
ax.bar(neg_index + width/2, mean_adv_negative, width, color='#990000', alpha=0.5)

ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_ylabel("Log Prob Change")
#ax.set_title("Tokenwise Log Probsin Normal vs Adversarial Models")
ax.set_xticks(np.append(np.arange(len(tokens)), [pos_index, neg_index]))
ax.set_xticklabels(tokens + ["Avg Pos$\Delta$", "Avg Neg$\Delta$"], rotation=45, ha='right')
ax.legend(loc='lower left')
ax.set_xlabel("Tokens")
plt.tight_layout()
plt.savefig("experiments/img/steering_vector.svg")
plt.show()
# %%
# Plotting averages only
fig_avg, ax_avg = plt.subplots(figsize=(3, 2))

# Define positions for average bars
avg_x = np.array([0, 1])  # Two bars for average positive and negative changes

# Plot average positive and negative changes with error bars
ax_avg.bar(avg_x[0], mean_clean_positive, width=0.4, color='#009900', alpha=1, label='Base')
ax_avg.errorbar(avg_x[0], mean_clean_positive, yerr=ci95_clean_positive, fmt='o', color='black')
ax_avg.bar(avg_x[1], mean_clean_negative, width=0.4, color='#009900', alpha=1)
ax_avg.errorbar(avg_x[1], mean_clean_negative, yerr=ci95_clean_negative, fmt='o', color='black')

ax_avg.bar(avg_x[0] + 0.5, mean_adv_positive, width=0.4, color='#990000', alpha=1,label='Adv.')
ax_avg.errorbar(avg_x[0] + 0.5, mean_adv_positive, yerr=ci95_adv_positive, fmt='o', color='black')
ax_avg.bar(avg_x[1] + 0.5, mean_adv_negative, width=0.4, color='#990000', alpha=1)
ax_avg.errorbar(avg_x[1] + 0.5, mean_adv_negative, yerr=ci95_adv_negative, fmt='o', color='black')

# Add horizontal line at y=0
ax_avg.axhline(0, color='black', linewidth=0.8, linestyle='--')

ax_avg.set_ylabel("Log Prob Change")
ax_avg.set_xticks(avg_x + 0.25)
ax_avg.set_xticklabels(["Avg$\Delta$. Pos", "Avg$\Delta$. Neg"])  # Adjusted to match the number of bars
ax_avg.legend(loc='upper right')
plt.tight_layout()
plt.savefig("experiments/img/steering_vector_avg.svg")
plt.show()


# %%
