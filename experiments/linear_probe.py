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
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# %%
# Ensure tokenizers are not parallelized
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load the model (assuming you have the adversarially trained model)
model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
model.load_state_dict(torch.load("saved_models/adversarially_trained_model.pth"))
print("Model loaded from checkpoint.")

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# %%
# Load the dataset (assuming Tiny Stories data tokens are pre-saved)
train_tokens = torch.load("train_tokens.pt")
val_tokens = torch.load("val_tokens.pt")
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

# Set model to evaluation mode
model.eval()

# Sentiment analysis setup using NLTK's VADER

nltk.download('vader_lexicon', quiet=True)

sia = SentimentIntensityAnalyzer()
# %%
# Define the linear probe
activation_key = 'blocks.2.hook_resid_post'
hidden_size = model.cfg['d_model']
linear_probe = nn.Linear(hidden_size, 1).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(linear_probe.parameters(), lr=1e-3)
# %%
def hook_fn(module, input, output):
            activations.append(output.detach())

# Register hook
hook = model.get_submodule(activation_key).register_forward_hook(hook_fn)

# %%
# Training loop
num_epochs = 5  # Adjust the number of epochs as needed
for epoch in range(num_epochs):
    model.eval()
    linear_probe.train()
    running_loss = 0.0

    # Loop over batches
    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", total=len(train_loader)):
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)

        # Convert input_ids to sequences
        sequences = model.to_string(input_ids)

        # Compute sentiment scores
        sentiment_scores = [sia.polarity_scores(seq)['compound'] for seq in sequences]
        sentiment_scores = torch.tensor(sentiment_scores, dtype=torch.float32).unsqueeze(1).to(device)  # Shape: (batch_size, 1)

        # Hook function to capture activations
        activations = []
        
        # Forward pass
        with torch.no_grad():  # No gradients for the model parameters
            outputs = model(input_ids)


        # Process activations
        activations_batch = activations[0]  # Shape: (batch_size, sequence_length, hidden_size)
        activations_mean = activations_batch.mean(dim=1)  # Shape: (batch_size, hidden_size)

        # Zero gradients for the linear probe
        optimizer.zero_grad()

        # Forward pass through the linear probe
        predictions = linear_probe(activations_mean)  # Shape: (batch_size, 1)

        # Compute loss
        loss = criterion(predictions, sentiment_scores)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Remove hook
hook.remove()