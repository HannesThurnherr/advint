# %%
%load_ext autoreload
%autoreload 2


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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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

# Disable donated buffers for torch.compile to allow retain_graph=True

# Enable interactive plotting
plt.ion() #! IMPORTANT

import transformer_lens
from transformer_lens import HookedTransformer
from SAE import TopKSparseAutoencoder # Assuming SAE.py contains TopKSparseAutoencoder
from datalib import load as load_tiny_stories_data # Assuming datalib/load.py exists
from utils import validate, loss_ce, get_tokenized_datasets
print("Packages imported")

from dataclasses import dataclass, fields
from typing import Optional, Union
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from gptneo import forward

@dataclass
class TrainingArgs:
    # Model/Data parameters
    llm_in: Optional[str] = None  # Path to load initial LLM weights (optional, default uses base)
    llm_out: str = "models/lm_DUMMY.pth" # Path to save the fine-tuned LLM: models/lm_adv.pth?
    model_name: str = "roneneldan/TinyStories-33M"
    dataset_name: str = "roneneldan/TinyStories"
    layer_num: int = 3  # Layer number for SAE intervention (1-indexed)

    # SAE parameters
    sae_in: Optional[str] = None # Path to load initial SAE weights (optional, default is random init)
    sae_out: str = "models/sae_DUMMY.pth" #"models/sae_adv_joint.pth"  # Path to save the trained SAE
    sae_mul: int = 10  # Multiplier for SAE latent dimension relative to residual dim
    k: int = 25  # Top-K value for SAE activation

    # Training parameters
    llm_lr: float = 5e-5 # Learning rate for LLM fine-tuning
    sae_lr: float = 5e-4  # Learning rate for SAE
    sae_lambda: float = 0.2 # Coefficient for adversarial SAE loss term for LLM
    batch_size: int = 8  # Training batch size
    epochs: int = 1  # Number of training epochs
    max_batches: int = None  # Maximum number of batches per epoch
    val_every: int = 250  # Logging frequency (batches)
    val_batches: int = 50  # Number of validation batches
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    # n_devices: int = 1  # Number of devices to train on
    # Misc
    num_workers: int = 4  # Number of workers for data loading
    seed: int = 42  # Random seed
    compile: bool = True  # Whether to compile the model and SAE
    max_batches: int = 7500

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

# %%
# Allows for running as jupyter notebook, will use default arguments
# try:
#     print("Parsing arguments...")
#     args = parse_args()
# except Exception as e:
#     print(f"Argument parsing failed ({e}), using default values.")

args = TrainingArgs()

print(args)
# %%
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
#model = HookedTransformer.from_pretrained(args.model_name, device=device)
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model.tokenizer = tokenizer
model.tokenizer.pad_token = model.tokenizer.eos_token

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
resid_dim = model.config.hidden_size
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
# if args.compile:
#     print("Compiling model...")
#     model = torch.compile(model)
#     print("Compiling SAE...")
#     SAE = torch.compile(SAE)

# --- Set Training Mode ---
# Model is fine-tuned, SAE is trained
model.train()
SAE.train()

# --- Data Loading ---
# print(f"Loading dataset: {args.dataset_name}")
train_tokens, val_tokens = load_tiny_stories_data(model.tokenizer, args.dataset_name) 
train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# train_dataset = get_tokenized_datasets(model.tokenizer, args.dataset_name, seq_len=512, batch_size=args.batch_size, split="train")
# val_dataset = get_tokenized_datasets(model.tokenizer, args.dataset_name, seq_len=512, batch_size=args.batch_size, split="validation")
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
# %%

# --- Plotting Setup ---
fig, ax = plt.subplots(figsize=(10, 6))
lines = {}
colors = plt.cm.tab10(np.linspace(0, 1, 4)) # Get distinct colors

# Initialize lines for different losses
lines['train_llm'], = ax.plot([], [], label='Train LLM Total Loss (L_ce - λ*L_ce_e2e)', marker='.', linestyle='-', color=colors[0])
lines['train_sae'], = ax.plot([], [], label='Train SAE Loss (L_ce_e2e)', marker='.', linestyle='-', color=colors[1])
lines['val_lm'], = ax.plot([], [], label='Val LM Loss', marker='o', linestyle='--', color=colors[2])
lines['val_e2e'], = ax.plot([], [], label='Val E2E Loss', marker='o', linestyle='--', color=colors[3])

ax.set_xlabel("Training Batches")
ax.set_ylabel("Loss")
ax.set_title("Live Training Progress")
ax.legend()
ax.grid(True)
plt.tight_layout()
fig.canvas.draw() # Initial draw

# Data storage for plotting
plot_data = {
    'batches': [],
    'train_llm_loss': [],
    'train_sae_loss': [],
    'val_batches': [],
    'val_lm_loss': [],
    'val_e2e_loss': []
}

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

# Initialize the postfix dictionary before the loop
postfix_stats = {
    "L_ce": "N/A",
    "L_ce_e2e": "N/A",
    "L_llm": "N/A",
    "val_loss": "N/A",
    "val_loss_e2e": "N/A",
    "acc_lm": "N/A",
    "acc_e2e": "N/A",
    "sae_l2": "N/A",
    "sae_l1": "N/A",
}
# %%

for epoch in range(args.epochs):
    print(f"--- Epoch {epoch+1}/{args.epochs} ---")
    model.train() # Ensure model is in train mode
    SAE.train()   # Ensure SAE is in train mode

    if args.max_batches is None:
        args.max_batches = len(train_loader)
    # Pass the initial postfix stats to tqdm
    progress_bar = tqdm(enumerate(train_loader), total=min(args.max_batches, len(train_loader)), desc=f"Epoch {epoch+1}", postfix=postfix_stats)

    for batch_idx, (input_ids, attention_mask) in progress_bar:
        if batch_idx >= args.max_batches:
            break

        # trim to max_len to speed up training
        max_len = attention_mask.sum(dim=-1).max()
        input_ids = input_ids[:, :max_len]
        attention_mask = attention_mask[:, :max_len]
        
        # trim for next token prediction
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        attention_mask = attention_mask[:, :-1]

        inputs = inputs.to(device)
        targets = targets.to(device)
        attention_mask = attention_mask.to(device) # Needed for loss calculation if using padding
        assert inputs.shape == targets.shape == attention_mask.shape


        with autocast():
            # --- Forward pass: Run model up to the intervention point ---
            first_pass = forward(model, inputs, stop_at_layer=stop_layer_idx, attention_mask=attention_mask)
            activations, attention_mask = first_pass['activations'], first_pass['attention_mask']
            
            logits = forward(model, activations, start_at_layer=start_layer_idx, attention_mask=attention_mask)['logits']
            loss_vanilla = loss_ce(logits, targets, ignore_index=padding_idx)
            
            reconstructed, _ = SAE(activations)
            
            # --- Continue E2E forward pass with reconstructed activations ---
            logits_e2e = forward(model, reconstructed, start_at_layer=start_layer_idx, attention_mask=attention_mask)['logits']
            loss_e2e = loss_ce(logits_e2e, targets, ignore_index=padding_idx)
            
            loss_llm = loss_vanilla - args.sae_lambda * loss_e2e
            loss_sae = loss_e2e

            # Update training stats in the dictionary
            postfix_stats["L_ce"] = f"{loss_vanilla.item():.3f}"
            postfix_stats["L_ce_e2e"] = f"{loss_e2e.item():.3f}"
            postfix_stats["L_llm"] = f"{loss_llm.item():.3f}"

        # --- Gradient Calculation and Optimizer Steps ---

        # Zero gradients *before* backward passes
        llm_optimizer.zero_grad(set_to_none=True)
        sae_optimizer.zero_grad(set_to_none=True)

        # --- Backward Passes ---
        # Scale and backward for LLM loss. Retain graph needed for loss_sae backward.
        scaler.scale(loss_llm).backward(retain_graph=True)

        # Scale and backward for SAE loss. Uses the retained graph.
        # Gradients computed here will only affect SAE parameters during the sae_optimizer step.
        scaler.scale(loss_sae).backward()

        # --- Optimizer Steps ---
        # Unscale, clip, and step for LLM
        # The scaler checks for inf/NaN gradients in model.parameters()
        scaler.unscale_(llm_optimizer)
        utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # Clip LLM grads
        scaler.step(llm_optimizer) # Steps the LLM optimizer

        # Unscale, clip, and step for SAE
        # The scaler checks for inf/NaN gradients in SAE.parameters()
        scaler.unscale_(sae_optimizer)
        utils.clip_grad_norm_(SAE.parameters(), args.max_grad_norm) # Clip SAE grads
        scaler.step(sae_optimizer) # Steps the SAE optimizer

        # Update the scaler *once* after both optimizer steps
        scaler.update()

        # --- Plotting Data Storage ---
        # Calculate current global batch index
        current_batch_idx = epoch * args.max_batches + batch_idx

        # Store batch losses for potential plotting (optional, can make plot noisy if plotted per batch)
        # plot_data['batches'].append(current_batch_idx) # Uncomment if plotting train loss per batch
        # plot_data['train_llm_loss'].append(loss_llm.item())
        # plot_data['train_sae_loss'].append(loss_sae.item())


        # --- Validation Step ---
        if (batch_idx + 1) % args.val_every == 0:
            # Run validation
            stats = validate(model, SAE, val_loader,
                             names = ["adv", "adv"],
                             max_batches = args.val_batches, # Use arg for consistency
                             verbose = False,
                             hf_model = True)

            # Update validation stats in the dictionary
            postfix_stats["val_loss"] = f"{stats['lm_loss']:.3f}"
            postfix_stats["val_loss_e2e"] = f"{stats['e2e_loss']:.3f}"
            postfix_stats["acc_lm"] = f"{stats['lm_acc']:.3f}"
            postfix_stats["acc_e2e"] = f"{stats['e2e_acc']:.3f}"
            postfix_stats["sae_l2"] = f"{stats['recon_l2']:.3f}"
            postfix_stats["sae_l1"] = f"{stats['sparsity_l1']:.3f}"

            # --- Update Plot ---
            # Store data points from this validation step
            plot_data['val_batches'].append(current_batch_idx) # Use calculated index
            plot_data['val_lm_loss'].append(stats['lm_loss'])
            plot_data['val_e2e_loss'].append(stats['e2e_loss'])

            # Store recent training losses (corresponding to this validation point)
            plot_data['batches'].append(current_batch_idx) # Use calculated index
            plot_data['train_llm_loss'].append(loss_llm.item()) # Last batch train loss
            plot_data['train_sae_loss'].append(loss_sae.item()) # Last batch train loss

            # Update plot lines
            lines['train_llm'].set_data(plot_data['batches'], plot_data['train_llm_loss'])
            lines['train_sae'].set_data(plot_data['batches'], plot_data['train_sae_loss'])
            lines['val_lm'].set_data(plot_data['val_batches'], plot_data['val_lm_loss'])
            lines['val_e2e'].set_data(plot_data['val_batches'], plot_data['val_e2e_loss'])

            # Rescale axes and redraw
            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events() # Process plot events

        # Set the postfix with the updated dictionary *once* per iteration
        progress_bar.set_postfix(postfix_stats)


print("Training complete.")

# --- Keep plot window open ---
plt.ioff() # Turn off interactive mode
plt.show() # Display the final plot

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

# %% 