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
val_adv = torch.load("data/TinyStories/features_val_adv.pt")
val_base = torch.load("data/TinyStories/features_val_base.pt")

# %%


# num_epochs = 1  # Define the number of epochs
# for model_name in ['base', 'adv']:  

#     train_activations = torch.load(f"data/TinyStories/features_train_{model_name}.pt")
#     val_activations = torch.load(f"data/TinyStories/features_val_{model_name}.pt")
    
#     train_dataset = TensorDataset(train_activations, features_train_labels)
#     val_dataset = TensorDataset(val_activations, features_val_labels)
    
#     train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
#     print(f"Train activations loaded for {model_name} model.")

#     hidden_size = train_activations.shape[-1]
#     linear_probe = nn.Linear(hidden_size, len(feature_list)).to(device)
#     linear_probe.train()

#     optimizer = Adam(linear_probe.parameters(), lr=1e-5)

#     eval_iter = 5000  # Define how often to evaluate

#     best_val_acc = 0
#     for epoch in range(num_epochs):
#         runner = tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} - Linear probe {model_name}", total=len(train_loader))
#         for batch_idx, (last_activations, labels) in runner:
#             last_activations = last_activations.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             logits = linear_probe(last_activations)
#             probs = torch.sigmoid(logits)

#             loss = F.binary_cross_entropy(probs, labels.float())

#             loss.backward()
#             optimizer.step()

#             train_losses[model_name].append(loss.item())
            
#             if batch_idx % eval_iter == 0:
#                 val_loss, class_accuracy, label_accuracy = evaluate(model_name, linear_probe, val_loader)
#                 val_losses[model_name].append(val_loss)
#                 val_acc[model_name].append((class_accuracy, label_accuracy))
#                 if label_accuracy > best_val_acc:
#                     best_val_acc = label_accuracy
#                     print(f"Saving best model {model_name} with val_acc_label {label_accuracy}")
#                     torch.save(linear_probe.state_dict(), f"models/linear_probe_{model_name}.pth")

#             status = [f'probe {model_name}',
#                       f'epoch {epoch+1}/{num_epochs}',
#                       f'loss: {loss.item():.4f}',
#                       f'val_loss: {val_losses[model_name][-1]:.4f}',
#                       f'val_acc_class: {val_acc[model_name][-1][0]:.4f}',
#                       f'val_acc_label: {val_acc[model_name][-1][1]:.4f}']
#             runner.set_description(" ".join(status))


