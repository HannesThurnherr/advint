# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# %%
# Ensure tokenizers are not parallelized

torch.manual_seed(42)
np.random.seed(42)
model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

train_tokens = torch.load("../data/TinyStories/train_tokens.pt")
val_tokens = torch.load("../data/TinyStories/val_tokens.pt")

train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

for model_name in ['base', 'adv']:
# Load the model (assuming you have the adversarially trained model)

    if model_name == 'adv':
        model.load_state_dict(torch.load("../models/lm_adv.pth"))
    print(f"Model loaded {model_name} from checkpoint.")


    train_activations = torch.load(f"../data/TinyStories/activations_train_{model_name}.pt")
    print(f"Train activations loaded for {model_name} model.")

    hidden_size = model.cfg.d_model
    linear_probe = nn.Linear(hidden_size, 1).to(device)
    linear_probe.train()

    criterion = nn.MSELoss()
    optimizer = Adam(linear_probe.parameters(), lr=1e-5)

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

        # # Hook function to capture activations
        # activations = []
        
        # Forward pass
        # with torch.no_grad():  # No gradients for the model parameters
        #     outputs = model(input_ids)
        with torch.no_grad():
            activations_batch = model(input_ids, attention_mask=attention_mask, stop_at_layer=3).detach() # run layers 0, 1, 2 (batch_size, sequence_length, hidden_size)

        # Process activations
        last_activations = activations_batch[:,-1,:]  # Shape: (batch_size, hidden_size)

        # Zero gradients for the linear probe
        optimizer.zero_grad()

        # Forward pass through the linear probe
        predictions = linear_probe(last_activations)  # Shape: (batch_size, 1)

        # Compute loss
        loss = criterion(predictions, sentiment_scores)

        # Backpropagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        losses.append(loss.item())
        if batch_idx % 100 == 0:
              plt.plot(losses)
              plt.show()
        if batch_idx > 600:
              break
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# %%
print(torch.mean(torch.tensor(losses[100:])))
#the probe on the adv model has achived tensor(0.5856) here (after 1232 batches)

# %%
plt.plot(losses)
plt.show()
# %%
import pickle

file_name = f"losses_{"adv" if load_adv_model else "normal"}_model_sa_probe.pkl"

with open(file_name, "wb") as f:
        pickle.dump(losses, f)
# %%

with open(f"losses_{"adv" if not load_adv_model else "normal"}_model_sa_probe.pkl", "rb") as f:
        old_losses = pickle.load( f)

plt.plot(losses)
plt.plot(old_losses)
plt.show()
# %%
