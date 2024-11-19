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
from collections import defaultdict
import sys
import random
# %%
# Ensure tokenizers are not parallelized
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Change directory to project root
os.chdir('/root/advint')
# Add the new working directory to sys.path
sys.path.append(os.getcwd())
# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentiment analysis setup using NLTK's VADER
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Load the dataset (adjust paths as necessary)
train_tokens = torch.load("train_tokens.pt")
val_tokens = torch.load("val_tokens.pt")

# Create Datasets
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])

# Define the activation key
activation_key = 'blocks.2.hook_resid_post'

def compute_steering_vector(model, activation_key, val_loader, num_batches=30):
    happy_activations = []
    unhappy_activations = []
    happy_weights = []
    unhappy_weights = []

    batch_counter = 0
    for batch_idx, batch in tqdm(enumerate(val_loader), desc="Computing Steering Vector", total=num_batches):
        if batch_counter >= num_batches:
            break
        input_ids, attention_mask = batch[0].to(device), batch[1].to(device)

        batch_size, seq_len = input_ids.shape

        # Random cutoff positions for each sequence
        cutoffs = [random.randint(5, min(100,seq_len)) for _ in range(batch_size)]  # Ensure minimum length

        # Collect activations and sentiment scores
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach())

        # Register hook
        hook = model.get_submodule(activation_key).register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            _ = model(input_ids)

        # Remove hook
        hook.remove()

        # Process each sequence individually
        activations_batch = activations[0]  # Shape: (batch_size, sequence_length, hidden_size)
        for idx in range(batch_size):
            cutoff = cutoffs[idx]
            seq_ids = input_ids[idx, :cutoff]
            sequence = model.to_string(seq_ids.unsqueeze(0))[0]

            # Compute sentiment score for the truncated sequence
            sentiment_score = sia.polarity_scores(sequence)['compound']

            # Get activation at the cutoff position
            activation = activations_batch[idx, cutoff - 1, :]  # Shape: (hidden_size,)

            if sentiment_score > 0.1:
                happy_activations.append(activation)
                happy_weights.append(sentiment_score)
            elif sentiment_score < -0.1:
                unhappy_activations.append(activation)
                unhappy_weights.append(-sentiment_score)

        batch_counter += 1

    # Compute weighted average activations
    if happy_activations and unhappy_activations:
        happy_activations = torch.stack(happy_activations)
        unhappy_activations = torch.stack(unhappy_activations)
        happy_weights = torch.tensor(happy_weights, dtype=torch.float32).unsqueeze(1).to(device)
        unhappy_weights = torch.tensor(unhappy_weights, dtype=torch.float32).unsqueeze(1).to(device)

        happy_mean = torch.sum(happy_activations * happy_weights, dim=0) / torch.sum(happy_weights)
        unhappy_mean = torch.sum(unhappy_activations * unhappy_weights, dim=0) / torch.sum(unhappy_weights)

        steering_vector = happy_mean - unhappy_mean
        return steering_vector
    else:
        print("Not enough data to compute steering vector.")
        return None

def modify_forward_with_steering(model, activation_key, steering_vector):
    def hook_fn(module, input, output):
        return output + steering_vector/2

    # Register hook
    hook = model.get_submodule(activation_key).register_forward_hook(hook_fn)
    return hook

def test_steering_effect(model, activation_key, steering_vector, prompt):
    # Tokenize the prompt
    input_ids = model.to_tokens(prompt).to(device)

    # Get logits without steering
    with torch.no_grad():
        logits = model(input_ids)
    last_logits = logits[0, -1, :]  # Shape: (vocab_size,)

    # Get log probabilities
    log_probs = torch.log_softmax(last_logits, dim=-1)

    # List of positive and negative tokens
    positive_tokens = [" happy", " good", " great", " joyful", " excited", " pleased", " wonderful"]
    negative_tokens = [" sad", " bad", " terrible", " unhappy", " depressed", " miserable", " awful"]

    positive_indices = [model.to_single_token(tok) for tok in positive_tokens]
    negative_indices = [model.to_single_token(tok) for tok in negative_tokens]

    # Log probabilities for positive and negative tokens without steering
    positive_log_probs = log_probs[positive_indices]
    negative_log_probs = log_probs[negative_indices]

    # Apply steering vector
    hook = modify_forward_with_steering(model, activation_key, steering_vector)

    # Get logits with steering
    with torch.no_grad():
        logits_steered = model(input_ids)
    last_logits_steered = logits_steered[0, -1, :]

    # Remove hook
    hook.remove()

    # Get log probabilities with steering
    log_probs_steered = torch.log_softmax(last_logits_steered, dim=-1)

    # Log probabilities for positive and negative tokens with steering
    positive_log_probs_steered = log_probs_steered[positive_indices]
    negative_log_probs_steered = log_probs_steered[negative_indices]

    # Compute the change in log probabilities
    positive_log_prob_changes = positive_log_probs_steered - positive_log_probs
    negative_log_prob_changes = negative_log_probs_steered - negative_log_probs

    # Aggregate results
    results = {
        "positive_tokens": positive_tokens,
        "negative_tokens": negative_tokens,
        "positive_log_probs": positive_log_probs.tolist(),
        "negative_log_probs": negative_log_probs.tolist(),
        "positive_log_probs_steered": positive_log_probs_steered.tolist(),
        "negative_log_probs_steered": negative_log_probs_steered.tolist(),
        "positive_log_prob_changes": positive_log_prob_changes.tolist(),
        "negative_log_prob_changes": negative_log_prob_changes.tolist(),
        "avg_positive_log_prob_change": positive_log_prob_changes.mean().item(),
        "avg_negative_log_prob_change": negative_log_prob_changes.mean().item()
    }

    return results

def run_steering_experiment(load_adv_model):
    # Load the model
    model = HookedTransformer.from_pretrained("tiny-stories-33M")
    if load_adv_model:
        model.load_state_dict(torch.load("saved_models/symbiotically_trained_model.pth"))
    model.to(device)
    model.eval()

    # Prepare DataLoader
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    # Compute steering vector
    steering_vector = compute_steering_vector(model, activation_key, val_loader, num_batches=30)
    if steering_vector is None:
        return None

    # Ensure steering_vector has the correct shape
    steering_vector = steering_vector.unsqueeze(0).to(device)  # Shape: (1, hidden_size)

    # Define the prompt
    prompt = "Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy felt very"

    # Test the effect of steering
    results = test_steering_effect(model, activation_key, steering_vector, prompt)

    model_type = "adversarially trained" if load_adv_model else "normal"
    print(f"\nResults for the {model_type} model:")
    print(f"Average change in log probability for positive tokens: {results['avg_positive_log_prob_change']}")
    print(f"Average change in log probability for negative tokens: {results['avg_negative_log_prob_change']}")

    # Detailed token-wise changes
    print("\nPositive Tokens Log Probability Changes:")
    for token, change in zip(results['positive_tokens'], results['positive_log_prob_changes']):
        print(f"  Token '{token}': {change}")

    print("\nNegative Tokens Log Probability Changes:")
    for token, change in zip(results['negative_tokens'], results['negative_log_prob_changes']):
        print(f"  Token '{token}': {change}")

    return results

# %%
# Run the experiment for both models
results_normal = run_steering_experiment(load_adv_model=False)
results_adv = run_steering_experiment(load_adv_model=True)

# Compare the changes
def compare_results(results_normal, results_adv):
    print("\nComparing the effectiveness of steering between the two models:\n")

    print(f"Average change in log probability for positive tokens:")
    print(f"  Normal model: {results_normal['avg_positive_log_prob_change']}")
    print(f"  Adversarially trained model: {results_adv['avg_positive_log_prob_change']}")

    print(f"\nAverage change in log probability for negative tokens:")
    print(f"  Normal model: {results_normal['avg_negative_log_prob_change']}")
    print(f"  Adversarially trained model: {results_adv['avg_negative_log_prob_change']}")

compare_results(results_normal, results_adv)
# %%
