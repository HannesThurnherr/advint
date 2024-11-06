
# %%
from tqdm import tqdm
import time
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from torch.nn import MSELoss, L1Loss
import torch.nn as nn
from SAE import TopKSparseAutoencoder
import transformer_lens
from torch.cuda.amp import autocast
import os
import matplotlib.pyplot as plt
from tiny_story_data import load_tiny_stories_data
from torch.cuda.amp import GradScaler
import torch.optim as optim
import torch.nn.utils as utils
# %%
# Check for cuda device
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(1, device=device)
    print(x)
else:
    device = torch.device("cpu")
    print("Cuda device not found. Using CPU.")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %%
# get the validation set
#check if data is loaded and tokenised, otherwise load it tokenise it and save it to a .pt file
if os.path.exists("train_tokens.pt"):
    val_tokens = torch.load('val_tokens.pt')
    print("loaded data from local")
else:
    print("tokenising data")
    load_tiny_stories_data()
    val_tokens = torch.load('val_tokens.pt')
    print("data loaded")


# %%¨
torch.cuda.empty_cache()
batch_size = 128  # Set your desired batch size
num_workers = 16  # Adjust this based on your system's CPU capacity
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)


# %%
# get the model and the SAE
model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
model.load_state_dict(torch.load("saved_models/adversarially_trained_model.pth"))
logits, activations = model.run_with_cache(["this", "is", "a", "test"])
activation_key = f'blocks.{2}.hook_resid_post'
resid_dim = activations[activation_key].shape[-1]
latent_dim = resid_dim * 10  # 8192
SAE = TopKSparseAutoencoder(input_dim=resid_dim, latent_dim=latent_dim).to(device)
SAE.load_state_dict(torch.load("saved_SAEs/adv_model_sae.pth"))
model.to(device)
SAE.to(device)
# %%
# get the feature activations on all the inputs
# Initialize GradScaler
scaler = GradScaler()

sae_losses = []
SAE.eval()
model.eval()

# %%

# Define hook function to capture activations
activation_key = f'blocks.{2}.hook_resid_post'
activations = None


def activation_hook(module, input, output):
    global activations
    activations = output


# Register hook
hook = model.get_submodule(activation_key).register_forward_hook(activation_hook)

# %%

print("starting SAE training")
# Progress bar for each batch within the epoch
for batch_idx, (input_ids, attention_mask) in tqdm(enumerate(val_loader), desc=f"Getting feature activations", total=len(val_loader), leave=False):
    # Model forward pass
    start_time = time.time()
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    

    with autocast():
        with torch.no_grad():
            _ = model(input_ids)  # Forward pass to trigger hook
            reconstructed, latent = SAE(activations)
    
    print(latent.shape)
    features = top_k_indices = torch.topk(latent, k=25, dim=-1).indices
    

    break



# %%
# %%
import heapq  # Priority queue for efficient top-k tracking
import pickle  # For saving the results to a file

# Dictionary to store top activations per feature
top_activations_per_feature = {}

# Define max storage per feature (e.g., 20)
TOP_K = 20

# Function to add new activations to our top activations dictionary
def update_top_activations(feature_idx, input_id, activation_value):
    if feature_idx not in top_activations_per_feature:
        top_activations_per_feature[feature_idx] = []
    heapq.heappush(top_activations_per_feature[feature_idx], (activation_value, input_id))
    
    # Maintain only TOP_K activations per feature (smallest activation will be discarded first)
    if len(top_activations_per_feature[feature_idx]) > TOP_K:
        heapq.heappop(top_activations_per_feature[feature_idx])

# Progress bar for each batch within the epoch
for batch_idx, (input_ids, attention_mask) in tqdm(enumerate(val_loader), desc="Getting feature activations", total=len(val_loader), leave=False):
    input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
    
    with autocast():
        with torch.no_grad():
            _ = model(input_ids)  # Forward pass to trigger hook
            reconstructed, latent = SAE(activations)
    
    # Iterate over each item in the batch and its sequence positions
    for batch_i in range(latent.shape[0]):
        for seq_i in range(latent.shape[1]):
            activation_vector = latent[batch_i, seq_i]
            
            # Get top k activations and corresponding feature indices for this position
            top_k_activations = torch.topk(activation_vector, k=25)
            top_k_values, top_k_indices = top_k_activations.values, top_k_activations.indices
            
            # Unique identifier for this specific input (e.g., batch and position)
            input_id = (batch_idx * batch_size + batch_i, seq_i)
            
            # Update dictionary for each top-activated feature
            for idx, feature_idx in enumerate(top_k_indices):
                update_top_activations(int(feature_idx), input_id, float(top_k_values[idx]))
    if batch_idx>9:
        break

# Save results to file
with open("top_activations_per_feature.pkl", "wb") as f:
    pickle.dump(top_activations_per_feature, f)

print("Top activations saved.")

# %%
def get_top_inputs_for_feature(feature_idx, token_ids):
    """
    Retrieve and display the top inputs (as decoded text) for a given feature.
    
    Args:
    - feature_idx (int): The feature index to inspect.
    - token_ids (torch.Tensor): Tensor containing the token IDs for the entire validation dataset.
    
    Returns:
    - List of tuples with activation values and decoded text strings.
    """
    if feature_idx not in top_activations_per_feature:
        print(f"No activations found for feature {feature_idx}.")
        return []

    # Retrieve the top activations and corresponding input positions
    top_activations = sorted(top_activations_per_feature[feature_idx], reverse=True)  # Sort by activation value

    # Decode and store the text for each top activation
    top_inputs_text = []
    for activation_value, (batch_idx, seq_pos) in top_activations:
        # Extract token IDs for this specific input position
        tokens = token_ids[batch_idx, max(0,seq_pos-10 ):seq_pos + 1]  # Assuming seq_pos is the start of the segment you want
        decoded_text = model.to_string(tokens)
        
        # Append the decoded text with the activation value
        top_inputs_text.append((activation_value, decoded_text))

    return top_inputs_text

# Example usage
for feature_idx in range(200,230):
    top_inputs = get_top_inputs_for_feature(feature_idx, val_tokens['input_ids'])
    print(f"Top inputs for feature {feature_idx}:")
    for i in top_inputs[:20]:
        print(i) 
# %%
