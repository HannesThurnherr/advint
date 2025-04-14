# %%
"""
# 1) Train the SAE on the base model
python -m train_sae.py --sae_out models/sae_base_e2e.pth
# 2) Do the advesarial training step, via adversarial_training.py, save weights to models/lm_adv.pth
# 3) Train the SAE on the adv model
python -m train_sae.py --llm_in models/lm_adv.pth --sae_out models/sae_post_adv_e2e.pth
"""
import sys
import os
import argparse
import warnings
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils as utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

import transformer_lens
from transformer_lens import HookedTransformer
from SAE import TopKSparseAutoencoder
from datalib import load as load_tiny_stories_data

from dataclasses import dataclass, fields
from typing import Optional, Union

@dataclass
class TrainingArgs:
    # Model/Data parameters
    llm_in: Optional[str] = None  # Path to load LLM weights (optional, default uses base model)
    model_name: str = "tiny-stories-33M"
    dataset_name: str = "roneneldan/TinyStories"
    layer_num: int = 3  # Layer number for SAE intervention (1-indexed)

    # SAE parameters
    sae_mul: int = 10  # Multiplier for SAE latent dimension relative to residual dim
    k: int = 25  # Top-K value for SAE activation
    sae_out: str = "models/DUMMY_FILE.pth"  # Path to save the trained SAE

    # Training parameters
    sae_lr: float = 3e-4  # Learning rate for SAE
    batch_size: int = 16  # Training batch size (up to 32 or 64 if VRAM allows)
    epochs: int = 1  # Number of training epochs
    max_batches: int = 5000  # Maximum number of batches per epoch
    val_every: int = 200  # Validation/logging frequency (batches)
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping

    # Misc
    num_workers: int = 4  # Number of workers for data loading
    seed: int = 42  # Random seed

def parse_args() -> TrainingArgs:
    """Parse command line arguments and override defaults in TrainingArgs."""
    args = TrainingArgs()
    
    parser = argparse.ArgumentParser(description='Train SAE end-to-end on a frozen model using start/stop_at_layer')
    
    # Model/Data parameters
    parser.add_argument('--llm_in', type=str, default=args.llm_in, 
                        help=f'Path to load LLM weights (default: {args.llm_in})')
    parser.add_argument('--model_name', type=str, default=args.model_name,
                        help=f'Model architecture name (default: {args.model_name})')
    parser.add_argument('--dataset_name', type=str, default=args.dataset_name,
                        help=f'Dataset name (default: {args.dataset_name})')
    parser.add_argument('--layer_num', type=int, default=args.layer_num,
                        help=f'Layer number for SAE intervention, 1-indexed (default: {args.layer_num})')
    
    # SAE parameters
    parser.add_argument('--sae_mul', type=int, default=args.sae_mul,
                        help=f'Multiplier for SAE latent dimension (default: {args.sae_mul})')
    parser.add_argument('--k', type=int, default=args.k,
                        help=f'Top-K value for SAE activation (default: {args.k})')
    parser.add_argument('--sae_out', type=str, default=args.sae_out,
                        help=f'Path to save the trained SAE (default: {args.sae_out})')
    
    # Training parameters
    parser.add_argument('--sae_lr', type=float, default=args.sae_lr,
                        help=f'Learning rate for SAE (default: {args.sae_lr})')
    parser.add_argument('--batch_size', type=int, default=args.batch_size,
                        help=f'Training batch size (default: {args.batch_size})')
    parser.add_argument('--epochs', type=int, default=args.epochs,
                        help=f'Number of training epochs (default: {args.epochs})')
    parser.add_argument('--max_batches', type=int, default=args.max_batches,
                        help=f'Maximum number of batches per epoch (default: {args.max_batches})')
    parser.add_argument('--val_every', type=int, default=args.val_every,
                        help=f'Validation/logging frequency in batches (default: {args.val_every})')
    parser.add_argument('--max_grad_norm', type=float, default=args.max_grad_norm,
                        help=f'Maximum gradient norm for clipping (default: {args.max_grad_norm})')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=args.num_workers,
                        help=f'Number of workers for data loading (default: {args.num_workers})')
    parser.add_argument('--seed', type=int, default=args.seed,
                        help=f'Random seed (default: {args.seed})')
    
    # Parse arguments and update the dataclass
    parsed_args = parser.parse_args()
    
    # Update dataclass with parsed arguments
    for key, value in vars(parsed_args).items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    return args

def loss_ce(logits, labels, ignore_index= None):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_index)

# %%
# Allows for running as jupyter notebook, will use default arguments
try:
    print("Parsing arguments...")
    args = parse_args()
except SystemExit:
    raise
except Exception as e:
    print(f"Argument parsing failed ({e}), using default values.")
    args = TrainingArgs()

print(args)

# --- Setup --- (Seed, Device, Model Loading, Freezing, Data Loading) ---
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if not torch.cuda.is_available():
    warnings.warn("CUDA device not found. Using CPU. Training might be slow.")

print(f"Loading model architecture: {args.model_name}")
model = HookedTransformer.from_pretrained(args.model_name, device=device)

print(f"Loading model weights from: {args.llm_in}")
if args.llm_in is None:
    print("No llm_in provided, using base model weights.")
else:
    if not os.path.exists(args.llm_in):
        raise FileNotFoundError(f"Model weights not found at {args.llm_in}")
    model.load_state_dict(torch.load(args.llm_in, map_location=device))
print(f"Model loaded onto {next(model.parameters()).device}")

print("Compiling model...")
model = torch.compile(model)

print("Freezing main model parameters...")
for param in model.parameters():
    param.requires_grad = False
model.eval()

print(f"Loading dataset: {args.dataset_name}")
# Assuming load_tiny_stories_data returns tokenized data directly
train_tokens, _ = load_tiny_stories_data(model.tokenizer, args.dataset_name) # Only need train for this script
# Ensure correct format (e.g., dictionary with 'input_ids' and 'attention_mask')
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

# %%

# --- SAE Initialization ---
resid_dim = model.cfg.d_model
sae_latent_dim = resid_dim * args.sae_mul
print(f"Initializing TopK SAE: resid_dim={resid_dim}, latent_dim={sae_latent_dim}, k={args.k}")
SAE = TopKSparseAutoencoder(input_dim=resid_dim, latent_dim=sae_latent_dim, k=args.k).to(device)
print("Compiling SAE...")
SAE = torch.compile(SAE)
SAE.train()

# --- Optimizer, Scaler, Loss ---
sae_optimizer = Adam(SAE.parameters(), lr=args.sae_lr)
scaler = GradScaler()
padding_idx = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else -1

# --- Training Loop ---
print("Starting SAE training...")
batch_losses = []
# Define the layer index for splitting the model pass
# stop_at_layer is exclusive, start_at_layer is inclusive.
# To get output after block N (resid_post), we stop *before* block N+1.
# To start computation *after* block N (using its output), we start *at* block N+1.
stop_layer_idx = args.layer_num
start_layer_idx = args.layer_num
print(f"Intervening after layer {args.layer_num}: stop_at_layer={stop_layer_idx}, start_at_layer={start_layer_idx}")

# %%

for epoch in range(args.epochs):
    print(f"--- Epoch {epoch+1}/{args.epochs} ---")
    SAE.train()
    model.eval() # Keep model frozen

    progress_bar = tqdm(enumerate(train_loader), total=min(args.max_batches, len(train_loader)), desc=f"Epoch {epoch+1}")


    for batch_idx, (input_ids, attention_mask) in progress_bar:
        if batch_idx >= args.max_batches:
            break

        input_ids = input_ids.to(device)
        # attention_mask is not directly used by model forward pass but needed for loss targets
        attention_mask = attention_mask.to(device)

        # Prepare inputs and targets for language modeling
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        # current_attention_mask = attention_mask[:, :-1] # Match input length if needed by loss

        sae_optimizer.zero_grad()

        with autocast():
            # --- Forward pass in two stages ---
            # Stage 1: Run model up to the intervention point
            # The output 'activations' will be the residual stream state *after* layer_num
            activations = model(inputs, stop_at_layer=stop_layer_idx)
            # Stage 2: Pass activations through SAE
            reconstructed, _ = SAE(activations) # latent is not needed for loss here
            # Stage 3: Run the rest of the model starting from the intervention point,
            # using the reconstructed activations as input.
            logits = model(reconstructed, start_at_layer=start_layer_idx)
            # --- End Forward pass ---

            # Calculate Cross Entropy loss based on the final logits
            loss = loss_ce(logits, targets, ignore_index=model.tokenizer.pad_token_id)

        # Backward pass - gradients flow back to the SAE via the reconstructed activations
        scaler.scale(loss).backward()

        # Unscale and clip gradients for SAE
        scaler.unscale_(sae_optimizer)
        utils.clip_grad_norm_(SAE.parameters(), args.max_grad_norm)

        # Optimizer step for SAE
        scaler.step(sae_optimizer)
        scaler.update()

        # Logging
        batch_losses.append(loss.item())
        if (batch_idx + 1) % args.val_every == 0:
            avg_loss = sum(batch_losses[-args.val_every:]) / args.val_every
            print(f"\nBatch {batch_idx+1}/{min(args.max_batches, len(train_loader))} - Avg Loss (last {args.val_every}): {avg_loss:.4f}")
            # Optional: Add validation loop here if needed

        progress_bar.set_postfix({"loss": loss.item()})

    # No hook removal needed
    # model_hook.remove()

print("Training complete.")

# --- Save SAE ---
print(f"Saving trained SAE to {args.sae_out}")
os.makedirs(os.path.dirname(args.sae_out), exist_ok=True)

if hasattr(SAE, '_orig_mod'):
    state_dict = SAE._orig_mod.state_dict()
else:
    state_dict = SAE.state_dict()
torch.save(state_dict, args.sae_out)
print("SAE saved.")

# --- Plotting ---
plt.figure(figsize=(10, 5))
plt.plot(batch_losses)
plt.xlabel("Batch")
plt.ylabel("Cross Entropy Loss")
plt.title("End-to-End SAE Training Loss (Frozen LM)")
plt.yscale("log")
plt.grid(True)
# Ensure plot directory exists if saving automatically
# os.makedirs(os.path.dirname(args.sae_out), exist_ok=True) # Already done above
plot_save_path = os.path.splitext(args.sae_out)[0] + "_loss_plot.png"
plt.savefig(plot_save_path)
print(f"Loss plot saved to {plot_save_path}")
# plt.show() # Keep commented out for non-interactive runs

# # --- Validation Step ---
# print("Starting validation...")
# val_dataset = load_tiny_stories_data(model.tokenizer, args.dataset_name, split="validation")  # Load validation data
# val_loader = DataLoader(TensorDataset(val_dataset['input_ids'], val_dataset['attention_mask']), 
#                         batch_size=args.batch_size, shuffle=False)
# validate(model, SAE, val_loader)

# %%
