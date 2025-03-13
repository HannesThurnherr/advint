"""
python -m experiments.linear_probe.get_features
python -m experiments.linear_probe.get_last_activations base
python -m experiments.linear_probe.get_last_activations adv
python -m experiments.linear_probe.feature_probe
"""

# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# %%
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
import transformer_lens
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch.nn.functional as F
# %%
# Ensure tokenizers are not parallelized
while os.path.basename(os.getcwd()) != 'advint':
    os.chdir('..')
print(f'Current working directory: {os.getcwd()}')


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

features_train = json.load(open("data/TinyStories/features_train.json"))
features_val = json.load(open("data/TinyStories/features_val.json"))
feature_list = ['Dialogue', 'BadEnding', 'Conflict', 'Foreshadowing', 'Twist', 'MoralValue']

#'Dialogue', 'BadEnding', 'Conflict', 'Foreshadowing', 'Twist', 'MoralValue'
# %%
def gen_labels(json):
    labels = []
    for entry in tqdm(json):
        features = entry['instruction']['features']
        many_hot_vector = [1 if feature in features else 0 for feature in feature_list]
        labels.append(many_hot_vector)
    return labels

features_train_labels = torch.tensor(gen_labels(features_train))
features_val_labels = torch.tensor(gen_labels(features_val))




train_losses = {'adv': [], 'base': []}
val_losses = {'adv': [], 'base': []}
val_acc = {'adv': [], 'base': []}

def evaluate(model_name, linear_probe, val_loader):
    linear_probe.eval()
    total_loss = 0
    correct_class_predictions = 0
    correct_label_predictions = 0
    total_samples = 0
    total_labels = 0

    with torch.no_grad():
        for last_activations, labels in tqdm(val_loader, desc=f"Evaluating {model_name}", total=len(val_loader)):
            last_activations = last_activations.to(device)
            labels = labels.to(device)

            logits = linear_probe(last_activations)
            probs = torch.sigmoid(logits)

            loss = F.binary_cross_entropy(probs, labels.float())
            total_loss += loss.item()

            predictions = (probs > 0.5).float()
            correct_class_predictions += (predictions == labels).sum().item()
            correct_label_predictions += (predictions == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)
            total_labels += labels.numel()

    val_loss = total_loss / len(val_loader)
    class_accuracy = correct_class_predictions / total_labels
    label_accuracy = correct_label_predictions / total_samples

    print(f"Validation - {model_name}: Loss = {val_loss:.4f}, Class Accuracy = {class_accuracy:.4f}, Label Accuracy = {label_accuracy:.4f}")

    return val_loss, class_accuracy, label_accuracy


# %%
num_epochs = 2  # Define the number of epochs
for model_name in ['base', 'adv']:  

    train_activations = torch.load(f"data/TinyStories/features_train_{model_name}.pt")
    val_activations = torch.load(f"data/TinyStories/features_val_{model_name}.pt")
    
    train_dataset = TensorDataset(train_activations, features_train_labels)
    val_dataset = TensorDataset(val_activations, features_val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train activations loaded for {model_name} model.")

    hidden_size = train_activations.shape[-1]
    
    
    linear_probe = nn.Sequential(
        nn.Linear(hidden_size, len(feature_list))
    ).to(device)
    linear_probe.train()

    optimizer = Adam(linear_probe.parameters(), lr=1e-5)

    eval_iter = 5000  # Define how often to evaluate

    best_val_acc = 0
    for epoch in range(num_epochs):
        runner = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Linear probe {model_name}", total=len(train_loader))
        for batch_idx, (last_activations, labels) in runner:
            last_activations = last_activations.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = linear_probe(last_activations)
            probs = torch.sigmoid(logits)

            loss = F.binary_cross_entropy(probs, labels.float())

            loss.backward()
            optimizer.step()

            train_losses[model_name].append(loss.item())
            
            if batch_idx % eval_iter == 0:
                val_loss, class_accuracy, label_accuracy = evaluate(model_name, linear_probe, val_loader)
                val_losses[model_name].append(val_loss)
                val_acc[model_name].append((class_accuracy, label_accuracy))
                if label_accuracy > best_val_acc:
                    best_val_acc = label_accuracy
                    print(f"Saving best model {model_name} with val_acc_label {label_accuracy}")
                    torch.save(linear_probe.state_dict(), f"models/linear_probe_{model_name}.pth")

            status = [f'probe {model_name}',
                      f'epoch {epoch+1}/{num_epochs}',
                      f'loss: {loss.item():.4f}',
                      f'val_loss: {val_losses[model_name][-1]:.4f}',
                      f'val_acc_class: {val_acc[model_name][-1][0]:.4f}',
                      f'val_acc_label: {val_acc[model_name][-1][1]:.4f}']
            runner.set_description(" ".join(status))

# %%


def smooth(data, window_size=100):
    """Smooth the data using a simple moving average."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Measure class accuracy if always guessing label [1,0,0,0,0,0]
most_freq_guess = torch.tensor([1, 0, 0, 0, 0, 0]).long()

correct_class_predictions = 0
correct_label_predictions = 0
total_samples = 0
total_labels = 0

for labels in tqdm(features_val_labels):
    correct_class_predictions += (most_freq_guess == labels).sum().item()
    correct_label_predictions += (most_freq_guess == labels).all().item()
    total_samples += 6
    total_labels += 1

val_class_acc = correct_class_predictions / total_samples
val_label_acc = correct_label_predictions / total_labels

print(f"Class accuracy if always guessing [1,0,0,0,0,0]: {val_class_acc:.4f}")
print(f"Label accuracy if always guessing [1,0,0,0,0,0]: {val_label_acc:.4f}")

# %%

# Create a figure and axes for plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot training and validation losses on a log scale
for model_type, color in zip(['base', 'adv'], ['blue', 'orange']):
    axs[0].plot(np.log(smooth(train_losses[model_type])), label=f'{model_type.capitalize()} Model - Training Loss', color=color)
    axs[0].plot(np.log(val_losses[model_type]), label=f'{model_type.capitalize()} Model - Validation Loss', linestyle='--', color=color)

axs[0].set_title('Losses (Log Scale)')
axs[0].set_xlabel('Batch')
axs[0].set_ylabel('Log Loss')
axs[0].legend()

# Plot validation accuracies as line charts
for model_type, color in zip(['base', 'adv'], ['blue', 'orange']):
    val_acc_class = [acc[0] for acc in val_acc[model_type]]
    val_acc_label = [acc[1] for acc in val_acc[model_type]]
    axs[1].plot(val_acc_class, label=f'{model_type.capitalize()} Model - Class Accuracy', color=color)
    axs[1].plot(val_acc_label, label=f'{model_type.capitalize()} Model - Label Accuracy', linestyle='--', color=color)

axs[1].set_title('Validation Accuracies')
axs[1].set_xlabel(f'Batch (every {eval_iter} steps)')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.tight_layout()
plt.savefig('experiments/img/feature_probe_train.svg')
plt.show()
plt.close(fig)

# Summarize the results in bar charts with annotations
fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# Last recorded validation losses and accuracies
val_loss_last = {model_type: val_losses[model_type][-1] for model_type in ['base', 'adv']}
val_acc_last = {model_type: (val_acc[model_type][-1][0], val_acc[model_type][-1][1]) for model_type in ['base', 'adv']}

axs[0].bar(['Base Model', 'Adversarial Model'], 
           [val_loss_last['base'], val_loss_last['adv']], 
           color=['blue', 'orange'])
axs[0].set_title('Last Recorded Validation Losses')
axs[0].set_ylabel('Loss')
for i, model_type in enumerate(['base', 'adv']):
    axs[0].text(i, val_loss_last[model_type], f'{val_loss_last[model_type]:.4f}', ha='center', va='bottom')

axs[1].bar(['Base Class', 'Base Label', 'Adv Class', 'Adv Label', 'Most Freq Class', 'Most Freq Label'],
           [val_acc_last['base'][0], val_acc_last['base'][1], val_acc_last['adv'][0], val_acc_last['adv'][1], 
            val_class_acc, val_label_acc],
           color=['blue', 'blue', 'orange', 'orange', 'green', 'green'])
axs[1].set_title('Last Recorded Validation Accuracies')
axs[1].set_ylabel('Accuracy')
for i, acc in enumerate([val_acc_last['base'][0], val_acc_last['base'][1], val_acc_last['adv'][0], val_acc_last['adv'][1], val_class_acc, val_label_acc]):
    axs[1].text(i, acc, f'{acc:.4f}', ha='center', va='bottom')

axs[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('experiments/img/feature_probe_stats.svg')
plt.show()
plt.close(fig)

# %%



# %%
plt.hist(features_val_labels.sum(dim=-1), bins=np.arange(features_val_labels.sum(dim=-1).min() - 0.5, features_val_labels.sum(dim=-1).max() + 1.5, 1), rwidth=0.8)
plt.yscale('log')
plt.title("feature count")
plt.xlabel("num of features")
plt.ylabel("frequency (log scale)")
plt.show()
# %%
