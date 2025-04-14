# %%
"""
Script for Adversarial Training of LLM and SAE.

Example Usage:

# 1. Train from base LLM and randomly initialized SAE:
python -m train_sae_adv.py \
    --sae_lambda 0.1 \
    --llm_lr 1e-5 \
    --sae_lr 3e-4 \
    --llm_out models/lm_adv_joint.pth \
    --sae_out models/sae_adv_joint.pth \
    --max_batches 10000

# 2. Train from fine-tuned LLM and pre-trained SAE:
python -m train_sae_adv.py \
    --llm_in models/lm_adv.pth \
    --sae_in models/sae_base_e2e.pth \
    --sae_lambda 0.1 \
    --llm_lr 1e-5 \
    --sae_lr 3e-4 \
    --llm_out models/lm_adv_joint_v2.pth \
    --sae_out models/sae_adv_joint_v2.pth \
    --max_batches 10000
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
from SAE import TopKSparseAutoencoder # Assuming SAE.py contains TopKSparseAutoencoder
from datalib import load as load_tiny_stories_data # Assuming datalib/load.py exists

print("Packages imported")

from dataclasses import dataclass, fields
from typing import Optional, Union
import argparse

@dataclass
class TrainingArgs:
    # Model/Data parameters
    llm_in: Optional[str] = None  # Path to load initial LLM weights (optional, default uses base)
    llm_out: str = "models/lm_adv_joint.pth" # Path to save the fine-tuned LLM
    model_name: str = "tiny-stories-33M"
    dataset_name: str = "roneneldan/TinyStories"
    layer_num: int = 3  # Layer number for SAE intervention (1-indexed)

    # SAE parameters
    sae_in: Optional[str] = None # Path to load initial SAE weights (optional, default is random init)
    sae_out: str = "models/sae_adv_joint.pth"  # Path to save the trained SAE
    sae_mul: int = 10  # Multiplier for SAE latent dimension relative to residual dim
    k: int = 25  # Top-K value for SAE activation

    # Training parameters
    llm_lr: float = 1e-5 # Learning rate for LLM fine-tuning
    sae_lr: float = 3e-4  # Learning rate for SAE
    sae_lambda: float = 0.1 # Coefficient for adversarial SAE loss term for LLM
    batch_size: int = 16  # Training batch size
    epochs: int = 1  # Number of training epochs
    max_batches: int = 5000  # Maximum number of batches per epoch
    val_every: int = 200  # Logging frequency (batches)
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping

    # Misc
    num_workers: int = 4  # Number of workers for data loading
    seed: int = 42  # Random seed

def parse_args() -> TrainingArgs:
    """Parse command line arguments and override defaults in TrainingArgs."""
    args = TrainingArgs()
    parser = argparse.ArgumentParser(description='Adversarially train LLM and SAE jointly')

    # Add arguments corresponding to TrainingArgs fields
    for field in fields(args): # Use fields() to iterate
        field_type = field.type
        # Handle optional types like Optional[str] which is Union[str, None]
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
             # Get the first type argument that isn't NoneType
             non_none_type = next((t for t in field_type.__args__ if t is not type(None)), None)
             if non_none_type:
                 field_type = non_none_type
             else: # Should not happen for Optional[str] but safety check
                 field_type = str
        elif field_type is type(None):
             field_type = str # Fallback if only None was specified somehow

        parser.add_argument(f'--{field.name}', type=field_type, default=getattr(args, field.name),
                            help=f'{field.name} (default: {getattr(args, field.name)})')

    # Parse arguments and update the dataclass
    parsed_args = parser.parse_args()

    # Update dataclass with parsed arguments
    for key, value in vars(parsed_args).items():
        if hasattr(args, key): # Ensure the argument exists in the dataclass
            setattr(args, key, value)

    return args

def loss_ce(logits, labels, ignore_index= None):
    """Calculates Cross Entropy loss."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)

# %%
# Allows for running as jupyter notebook, will use default arguments
try:
    print("Parsing arguments...")
    args = parse_args()
except SystemExit: # Catch SystemExit from parser.parse_args() on --help
     raise # Re-raise to exit cleanly
except Exception as e:
    print(f"Argument parsing failed ({e}), using default values.")
    args = TrainingArgs()

print(args)

# --- Setup --- (Seed, Device) ---
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if not torch.cuda.is_available():
    warnings.warn("CUDA device not found. Using CPU. Training might be slow.")

# --- Model Loading ---
print(f"Loading model architecture: {args.model_name}")
model = HookedTransformer.from_pretrained(args.model_name, device=device)

# Use llm_in argument
if args.llm_in:
    if not os.path.exists(args.llm_in):
        raise FileNotFoundError(f"Model weights not found at {args.llm_in}")
    print(f"Loading model weights from: {args.llm_in}")
    model.load_state_dict(torch.load(args.llm_in, map_location=device))
else:
    print("No llm_in provided, using base model weights.")
print(f"Model loaded onto {next(model.parameters()).device}")

# --- SAE Initialization & Loading ---
resid_dim = model.cfg.d_model
sae_latent_dim = resid_dim * args.sae_mul
print(f"Initializing TopK SAE: resid_dim={resid_dim}, latent_dim={sae_latent_dim}, k={args.k}")
SAE = TopKSparseAutoencoder(input_dim=resid_dim, latent_dim=sae_latent_dim, k=args.k).to(device)

# Use sae_in argument
if args.sae_in:
    if not os.path.exists(args.sae_in):
        raise FileNotFoundError(f"SAE weights not found at {args.sae_in}")
    print(f"Loading SAE weights from: {args.sae_in}")
    sae_state_dict = torch.load(args.sae_in, map_location=device)
    if hasattr(SAE, '_orig_mod'):
         SAE._orig_mod.load_state_dict(sae_state_dict)
    else:
         SAE.load_state_dict(sae_state_dict)
else:
    print("No sae_in provided, starting SAE training from scratch.")

# --- Compile Models ---
print("Compiling model...")
model = torch.compile(model)
print("Compiling SAE...")
SAE = torch.compile(SAE)

# --- Set Training Mode ---
# Model is fine-tuned, SAE is trained
model.train()
SAE.train()

# --- Data Loading ---
print(f"Loading dataset: {args.dataset_name}")
train_tokens, _ = load_tiny_stories_data(model.tokenizer, args.dataset_name) # Only need train
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

# --- Optimizers, Scaler ---
llm_optimizer = Adam(model.parameters(), lr=args.llm_lr)
sae_optimizer = Adam(SAE.parameters(), lr=args.sae_lr)
scaler = GradScaler()
padding_idx = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else -1

# --- Training Loop ---
print("Starting Adversarial SAE training...")
batch_losses_standard = []
batch_losses_e2e = []
batch_losses_llm_total = []

stop_layer_idx = args.layer_num
start_layer_idx = args.layer_num
print(f"Intervening after layer {args.layer_num}: stop_at_layer={stop_layer_idx}, start_at_layer={start_layer_idx}")

for epoch in range(args.epochs):
    print(f"--- Epoch {epoch+1}/{args.epochs} ---")
    model.train() # Ensure model is in train mode
    SAE.train()   # Ensure SAE is in train mode

    progress_bar = tqdm(enumerate(train_loader), total=min(args.max_batches, len(train_loader)), desc=f"Epoch {epoch+1}")

    for batch_idx, (input_ids, attention_mask) in progress_bar:
        if batch_idx >= args.max_batches:
            break

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) # Needed for loss calculation if using padding

        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        llm_optimizer.zero_grad()
        sae_optimizer.zero_grad()

        with autocast():
            # --- Forward pass: Run model up to the intervention point ---
            activations = model(inputs, stop_at_layer=stop_layer_idx)
            
            # --- Continue standard forward pass ---
            logits_standard = model(activations, start_at_layer=start_layer_idx)
            loss_ce_standard = loss_ce(logits_standard, targets, ignore_index=padding_idx)
            
            # --- SAE intervention for E2E pass ---
            # Ensure activations are on the same device as SAE weights
            sae_device = next(SAE.parameters()).device
            activations_for_sae = activations.to(sae_device)
            reconstructed, _ = SAE(activations_for_sae)
            reconstructed = reconstructed.to(activations.device) # Move back if necessary
            
            # --- Continue E2E forward pass with reconstructed activations ---
            logits_e2e = model(reconstructed, start_at_layer=start_layer_idx)
            loss_ce_e2e = loss_ce(logits_e2e, targets, ignore_index=padding_idx)
            
            # --- Define Loss for each component ---
            # LLM Loss: Standard CE - lambda * E2E CE (minimize standard, maximize E2E)
            loss_llm = loss_ce_standard - args.sae_lambda * loss_ce_e2e
            
            # SAE Loss: E2E CE (minimize E2E)
            loss_sae = loss_ce_e2e

        # --- Backward & Optimizer Steps ---
        # Scale and backward for LLM loss (affects LLM params)
        # Need retain_graph=True because the E2E forward pass graph is used by both loss_llm and loss_sae backward passes
        scaler.scale(loss_llm).backward(retain_graph=True)

        # Scale and backward for SAE loss (affects SAE params)
        # The gradients from loss_llm affecting SAE params via the (-lambda * loss_ce_e2e) term
        # are already computed. This backward call adds the gradients from loss_sae (+1 * loss_ce_e2e).
        # The net effect on SAE params is proportional to (1 - lambda) * d(loss_ce_e2e)/d(SAE).
        # If lambda > 1, this flips the sign, making SAE optimize *against* E2E loss too.
        scaler.scale(loss_sae).backward()

        # Unscale and clip gradients for LLM
        scaler.unscale_(llm_optimizer)
        utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Unscale and clip gradients for SAE
        scaler.unscale_(sae_optimizer)
        utils.clip_grad_norm_(SAE.parameters(), args.max_grad_norm)

        # Optimizer steps
        scaler.step(llm_optimizer)
        scaler.step(sae_optimizer)

        # Update scaler
        scaler.update()

        # Logging
        batch_losses_standard.append(loss_ce_standard.item())
        batch_losses_e2e.append(loss_ce_e2e.item())
        batch_losses_llm_total.append(loss_llm.item()) # Log the LLM's objective value

        if (batch_idx + 1) % args.val_every == 0:
            avg_loss_std = np.mean(batch_losses_standard[-args.val_every:])
            avg_loss_e2e = np.mean(batch_losses_e2e[-args.val_every:])
            avg_loss_llm = np.mean(batch_losses_llm_total[-args.val_every:])
            print(f"\nBatch {batch_idx+1}/{min(args.max_batches, len(train_loader))} - "
                  f"Avg Loss (Std): {avg_loss_std:.4f}, "
                  f"Avg Loss (E2E): {avg_loss_e2e:.4f}, "
                  f"Avg Loss (LLM Obj): {avg_loss_llm:.4f}")

        progress_bar.set_postfix({
            "loss_std": f"{loss_ce_standard.item():.3f}",
            "loss_e2e": f"{loss_ce_e2e.item():.3f}",
            "loss_llm": f"{loss_llm.item():.3f}"
        })


print("Training complete.")

# --- Save Models ---
# Ensure directories exist
os.makedirs(os.path.dirname(args.llm_out), exist_ok=True)
os.makedirs(os.path.dirname(args.sae_out), exist_ok=True)

# Save LLM
print(f"Saving fine-tuned LLM to {args.llm_out}")
if hasattr(model, '_orig_mod'):
    llm_state_dict = model._orig_mod.state_dict()
else:
    llm_state_dict = model.state_dict()
torch.save(llm_state_dict, args.llm_out)
print("LLM saved.")

# Save SAE
print(f"Saving trained SAE to {args.sae_out}")
if hasattr(SAE, '_orig_mod'):
    sae_state_dict = SAE._orig_mod.state_dict()
else:
    sae_state_dict = SAE.state_dict()
torch.save(sae_state_dict, args.sae_out)
print("SAE saved.")

# --- Plotting ---
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(batch_losses_standard, label='Standard CE Loss')
plt.plot(batch_losses_e2e, label='End-to-End CE Loss (SAE)')
plt.xlabel("Batch")
plt.ylabel("Cross Entropy Loss")
plt.title("Standard vs E2E Loss")
plt.yscale("log")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(batch_losses_llm_total, label=f'LLM Objective (Std - {args.sae_lambda}*E2E)')
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("LLM Objective Function Value")
plt.yscale("log" if min(batch_losses_llm_total) > 0 else "linear") # Log scale if possible
plt.grid(True)
plt.legend()

plt.tight_layout()
plot_save_path = os.path.splitext(args.sae_out)[0] + "_adv_loss_plot.png"
plt.savefig(plot_save_path)
print(f"Loss plot saved to {plot_save_path}")
# plt.show()

# %% 