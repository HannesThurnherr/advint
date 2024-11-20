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

# Change directory to project root (adjust if necessary)
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
        cutoffs = [random.randint(5, min(100, seq_len)) for _ in range(batch_size)]  # Ensure minimum length

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
        return output + steering_vector / 2  # Adjust scaling if necessary

    # Register hook
    hook = model.get_submodule(activation_key).register_forward_hook(hook_fn)
    return hook

def test_steering_effect(model, activation_key, steering_vector, prompts):
    # List of positive and negative tokens
    positive_tokens = [" happy", " good", " great", " joyful", " excited", " pleased", " wonderful"]
    negative_tokens = [" sad", " bad", " terrible", " unhappy", " depressed", " miserable", " awful"]

    positive_indices = [model.to_single_token(tok) for tok in positive_tokens]
    negative_indices = [model.to_single_token(tok) for tok in negative_tokens]

    # Initialize accumulators
    positive_log_prob_changes = []
    negative_log_prob_changes = []

    for prompt in prompts:
        # Tokenize the prompt
        input_ids = model.to_tokens(prompt).to(device)

        # Get logits without steering
        with torch.no_grad():
            logits = model(input_ids)
        last_logits = logits[0, -1, :]  # Shape: (vocab_size,)

        # Get log probabilities
        log_probs = torch.log_softmax(last_logits, dim=-1)

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
        positive_log_prob_change = positive_log_probs_steered - positive_log_probs
        negative_log_prob_change = negative_log_probs_steered - negative_log_probs

        positive_log_prob_changes.append(positive_log_prob_change)
        negative_log_prob_changes.append(negative_log_prob_change)

    # Convert lists to tensors
    positive_log_prob_changes = torch.stack(positive_log_prob_changes)  # Shape: (num_prompts, num_tokens)
    negative_log_prob_changes = torch.stack(negative_log_prob_changes)

    # Aggregate results
    avg_positive_log_prob_change = positive_log_prob_changes.mean(dim=0)  # Average over prompts
    avg_negative_log_prob_change = negative_log_prob_changes.mean(dim=0)

    overall_avg_positive_change = avg_positive_log_prob_change.mean().item()
    overall_avg_negative_change = avg_negative_log_prob_change.mean().item()

    results = {
        "positive_tokens": positive_tokens,
        "negative_tokens": negative_tokens,
        "avg_positive_log_prob_change": overall_avg_positive_change,
        "avg_negative_log_prob_change": overall_avg_negative_change,
        "positive_log_prob_changes": avg_positive_log_prob_change.tolist(),
        "negative_log_prob_changes": avg_negative_log_prob_change.tolist()
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


    prompts = [
        "Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy just lost her beloved ball. She felt very",
        "In the heart of the bustling city, there was a small café owned by Emily. The café had seen fewer customers lately, and Emily worried it might close. She felt very",
        "After a long day at work, John returned home to his quiet apartment. His coworkers had ignored his ideas all day, leaving him feeling insignificant. He felt very",
        "Under the starry night sky, the campers gathered around the fire. They had just realized they were lost, with no signal to call for help. They felt very",
        "In the middle of the desert, a lone traveler set up camp for the night. Her water supply was running dangerously low, and she hadn’t seen another person for days. She felt very",
        "During the summer festival, the town square was filled with music and laughter. But he couldn’t stop thinking about how his best friend had moved away the week before. He felt very",
        "At the edge of the ocean, a young boy watched the waves crash against the shore. His kite had torn in the strong wind, leaving him disappointed. He felt very",
        "In the old library, shelves upon shelves of books stood tall. She had spent hours searching for her favorite book, but it wasn’t there. She felt very",
        "Amidst the snowy mountains, a group of friends embarked on a hiking trip. They had just realized they’d forgotten their food supplies at the last campsite. They felt very",
        "In the cozy living room, the family gathered for movie night. But the youngest child had just spilled their popcorn all over the floor. She felt very",
        "On the sunny hillside, flowers bloomed in vibrant colors. He had been planning to take pictures for his portfolio, but his camera battery had died. He felt very",
        "Inside the ancient castle, mysteries awaited to be uncovered. She had just tripped over a loose stone and twisted her ankle. She felt very",
        "During the autumn harvest, the farmers worked tirelessly in the fields. But the early frost had ruined much of the crop, leaving them worried about the season. They felt very",
        "At the local park, children played on the swings and slides. He sat on a bench, watching, remembering how his son used to play there before they moved away. He felt very",
        "In the quiet meadow, butterflies fluttered among the wildflowers. She had come there to paint, but her canvas was ruined by an unexpected rainstorm. She felt very",
        "Under the bright lights of the theater, the actors prepared for the performance. But the lead actor had forgotten their lines during rehearsal, filling them with doubt. They felt very",
        "In the serene garden, birds sang melodies in the morning. He had just received bad news from work, overshadowing the peaceful moment. He felt very",
        "At the art studio, painters mixed colors on their palettes. Her favorite painting had been damaged accidentally by a falling easel. She felt very",
        "In the bustling marketplace, vendors called out to attract customers. But their stall had been knocked over in the chaos, scattering their goods. They felt very",
        "On the quiet beach, the sun began to set over the horizon. He had been waiting for someone who never showed up. He felt very"
    ]
  

    # Test the effect of steering
    results = test_steering_effect(model, activation_key, steering_vector, prompts)

    model_type = "adversarially trained" if load_adv_model else "normal"
    print(f"\nResults for the {model_type} model:")
    print(f"Average change in log probability for positive tokens: {results['avg_positive_log_prob_change']}")
    print(f"Average change in log probability for negative tokens: {results['avg_negative_log_prob_change']}")

    # Detailed token-wise changes
    print("\nPositive Tokens Average Log Probability Changes:")
    for token, change in zip(results['positive_tokens'], results['positive_log_prob_changes']):
        print(f"  Token '{token}': {change}")

    print("\nNegative Tokens Average Log Probability Changes:")
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

    print(f"Overall average change in log probability for positive tokens:")
    print(f"  Normal model: {results_normal['avg_positive_log_prob_change']}")
    print(f"  Adversarially trained model: {results_adv['avg_positive_log_prob_change']}")

    print(f"\nOverall average change in log probability for negative tokens:")
    print(f"  Normal model: {results_normal['avg_negative_log_prob_change']}")
    print(f"  Adversarially trained model: {results_adv['avg_negative_log_prob_change']}")

compare_results(results_normal, results_adv)
# %%
