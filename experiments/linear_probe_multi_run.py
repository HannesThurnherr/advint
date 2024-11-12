# %%
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
import numpy as np
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
import matplotlib.pyplot as plt
# %%
# Ensure tokenizers are not parallelized
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentiment analysis setup using NLTK's VADER
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Load the dataset (adjust paths as necessary)
train_tokens = torch.load("../train_tokens.pt")
val_tokens = torch.load("../val_tokens.pt")

# Create Datasets
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])

# Define the activation key
activation_key = 'blocks.2.hook_resid_post'

def run_experiment(load_adv_model, run_number):
    # Set random seeds
    torch.manual_seed(run_number)
    np.random.seed(run_number)
    
    # Load the model
    model = HookedTransformer.from_pretrained("tiny-stories-33M")
    if load_adv_model:
        model.load_state_dict(torch.load("../saved_models/adversarially_trained_model.pth"))
    print(f"Model loaded. Adversarial model: {load_adv_model}")
    model.to(device)
    
    # Get hidden_size
    _, activations = model.run_with_cache(["this", "is", "a", "test"])
    hidden_size = activations[activation_key].shape[-1]
    
    # Define the linear probe
    linear_probe = nn.Linear(hidden_size, 1).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(linear_probe.parameters(), lr=1e-5)
    
    # Define hook function
    def hook_fn(module, input, output):
        activations.append(output.detach())
    
    # Register hook
    hook = model.get_submodule(activation_key).register_forward_hook(hook_fn)
    
    # Create DataLoaders with shuffling
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)
    
    # Limit training to 500 batches
    max_train_batches = 500
    max_val_batches = 100  # For validation
    
    # Training loop
    model.eval()
    linear_probe.train()
    training_losses = []
    training_metrics = {
        'mse': [],
        'mae': [],
        'r2': []
    }

    train_batch_counter = 0
    for batch_idx, batch in tqdm(enumerate(train_loader), desc=f"Training Run {run_number}", total=max_train_batches):
        if train_batch_counter >= max_train_batches:
            break
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)

        # Convert input_ids to sequences
        sequences = model.to_string(input_ids)

        # Compute sentiment scores
        sentiment_scores = [sia.polarity_scores(seq)['compound'] for seq in sequences]
        sentiment_scores = torch.tensor(sentiment_scores, dtype=torch.float32).unsqueeze(1).to(device)

        # Hook function to capture activations
        activations = []
        
        # Forward pass
        with torch.no_grad():  # No gradients for the model parameters
            _ = model(input_ids)

        # Process activations
        activations_batch = activations[0]  # Shape: (batch_size, sequence_length, hidden_size)
        last_activations = activations_batch[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Zero gradients for the linear probe
        optimizer.zero_grad()

        # Forward pass through the linear probe
        predictions = linear_probe(last_activations)  # Shape: (batch_size, 1)

        # Compute loss
        loss = criterion(predictions, sentiment_scores)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Store training loss
        training_losses.append(loss.item())

        # Compute additional metrics using torch functions
        mse = torch.mean((predictions - sentiment_scores) ** 2).item()
        mae = torch.mean(torch.abs(predictions - sentiment_scores)).item()
        
        # Compute R² score
        total_variance = torch.var(sentiment_scores, unbiased=False) * sentiment_scores.size(0)
        unexplained_variance = torch.sum((sentiment_scores - predictions) ** 2)
        r2 = 1 - (unexplained_variance / total_variance).item()

        training_metrics['mse'].append(mse)
        training_metrics['mae'].append(mae)
        training_metrics['r2'].append(r2)

        train_batch_counter += 1

    # Validation
    linear_probe.eval()
    val_losses = []
    val_metrics = {
        'mse': [],
        'mae': [],
        'r2': []
    }
    val_batch_counter = 0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(val_loader), desc=f"Validation Run {run_number}", total=max_val_batches):
            if val_batch_counter >= max_val_batches:
                break
            input_ids, attention_mask = batch[0].to(device), batch[1].to(device)

            # Convert input_ids to sequences
            sequences = model.to_string(input_ids)

            # Compute sentiment scores
            sentiment_scores = [sia.polarity_scores(seq)['compound'] for seq in sequences]
            sentiment_scores = torch.tensor(sentiment_scores, dtype=torch.float32).unsqueeze(1).to(device)

            # Hook function to capture activations
            activations = []
            
            # Forward pass
            _ = model(input_ids)

            # Process activations
            activations_batch = activations[0]  # Shape: (batch_size, sequence_length, hidden_size)
            last_activations = activations_batch[:, -1, :]  # Shape: (batch_size, hidden_size)

            # Forward pass through the linear probe
            predictions = linear_probe(last_activations)  # Shape: (batch_size, 1)

            # Compute loss
            loss = criterion(predictions, sentiment_scores)
            val_losses.append(loss.item())

            # Compute additional metrics using torch functions
            mse = torch.mean((predictions - sentiment_scores) ** 2).item()
            mae = torch.mean(torch.abs(predictions - sentiment_scores)).item()
            
            # Compute R² score
            total_variance = torch.var(sentiment_scores, unbiased=False) * sentiment_scores.size(0)
            unexplained_variance = torch.sum((sentiment_scores - predictions) ** 2)
            r2 = 1 - (unexplained_variance / total_variance).item()

            val_metrics['mse'].append(mse)
            val_metrics['mae'].append(mae)
            val_metrics['r2'].append(r2)

            val_batch_counter += 1

    # Remove hook
    hook.remove()
    
    # Compute average metrics
    avg_training_metrics = {metric: np.mean(values) for metric, values in training_metrics.items()}
    avg_val_metrics = {metric: np.mean(values) for metric, values in val_metrics.items()}

    # Log results
    results = {
        'training_losses': training_losses,
        'training_metrics': training_metrics,
        'avg_training_metrics': avg_training_metrics,
        'val_losses': val_losses,
        'val_metrics': val_metrics,
        'avg_val_metrics': avg_val_metrics
    }

    # Save results
    model_type = "adv" if load_adv_model else "normal"
    file_name = f"metrics_{model_type}_model_run{run_number}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(results, f)
    print(f"Metrics saved to {file_name}")

    return avg_training_metrics, avg_val_metrics

# %%
# Results dictionary to store average metrics across runs
overall_results = {'normal': [], 'adv': []}
for load_adv_model in [False, True]:
    model_type = "adv" if load_adv_model else "normal"
    for run_number in range(1, 11):  # 10 runs
        print(f"\nStarting run {run_number} for model '{model_type}'")
        avg_train_metrics, avg_val_metrics = run_experiment(load_adv_model, run_number)
        overall_results[model_type].append({
            'avg_training_metrics': avg_train_metrics,
            'avg_val_metrics': avg_val_metrics
        })
        print(f"Run {run_number}, Model '{model_type}' completed.")
        print(f"Average Training Metrics: {avg_train_metrics}")
        print(f"Average Validation Metrics: {avg_val_metrics}")
# Save the overall results
with open("experiment_overall_results.pkl", "wb") as f:
    pickle.dump(overall_results, f)
print("\nExperiment completed. Overall results saved to 'experiment_overall_results.pkl'")

# %%
# %%

# %%


# Load the aggregated results
with open("experiment_overall_results.pkl", "rb") as f:
    overall_results = pickle.load(f)

# Function to extract metric values for box plot
def extract_metrics_for_boxplot(overall_results, metric):
    normal_metrics = [run['avg_val_metrics'][metric] for run in overall_results['normal']]
    adv_metrics = [run['avg_val_metrics'][metric] for run in overall_results['adv']]
    return normal_metrics, adv_metrics

# Visualization function for box plots
def plot_boxplot(metric_name, normal_values, adv_values):
    plt.figure()
    plt.boxplot([normal_values, adv_values], labels=["Normal Model", "Adversarially trained Model"])
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} of the SC Linear Probe Distribution")
    plt.show()

# Plot each metric (MSE, MAE, R²) as box plots
metrics = ["mse", "mae", "r2"]
for metric in metrics:
    normal_values, adv_values = extract_metrics_for_boxplot(overall_results, metric)
    plot_boxplot(metric.upper(), normal_values, adv_values)
# %%
