# %%
"""
Script for Adversarial Training of LLM and SAE.

Example Usage:

# Basic run with wandb:
python train_sae_adv.py --sae_lambda 0.1 --llm_lr 1e-5 --sae_lr 3e-4 --llm_out models/lm_adv_joint.pth --sae_out models/sae_adv_joint.pth --max_batches 10000 --wandb_project your_wandb_project_name

# Run from pre-trained models:
python train_sae_adv.py --llm_in models/lm_adv.pth --sae_in models/sae_base_e2e.pth --sae_lambda 0.1 --llm_lr 1e-5 --sae_lr 3e-4 --llm_out models/lm_adv_joint_v2.pth --sae_out models/sae_adv_joint_v2.pth --max_batches 10000 --wandb_project your_wandb_project_name
"""
import sys
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Keep or remove as needed
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import warnings
import time
import wandb # Import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils as utils
from tqdm import tqdm
# import matplotlib.pyplot as plt # No longer needed for live plotting
import numpy as np

import transformer_lens
from transformer_lens import HookedTransformer
from SAE import TopKSparseAutoencoder # Assuming SAE.py contains TopKSparseAutoencoder
from datalib import load as load_tiny_stories_data # Assuming datalib/load.py exists
from utils import validate, loss_ce, get_tokenized_datasets
print("Packages imported")

from dataclasses import dataclass, fields, asdict
from typing import Optional, Union
import argparse

@dataclass
class TrainingArgs:
    # Model/Data parameters
    llm_in: Optional[str] = None
    llm_out: str = "models/lm_DUMMY.pth"
    model_name: str = "tiny-stories-33M"
    dataset_name: str = "roneneldan/TinyStories"
    layer_num: int = 3

    # SAE parameters
    sae_in: Optional[str] = None
    sae_out: str = "models/sae_DUMMY.pth"
    sae_mul: int = 10
    k: int = 25

    # Training parameters
    llm_lr: float = 5e-5 # Learning rate for LLM fine-tuning
    sae_lr: float = 5e-4  # Learning rate for SAE
    sae_lambda: float = 0.2 # Coefficient for adversarial SAE loss term for LLM
    batch_size: int = 8 # Training batch size
    epochs: int = 1  # Number of training epochs
    max_batches: int = 7500  # Maximum number of batches per epoch
    val_every: int = 250  # Logging frequency (batches)
    val_batches: int = 100  # Number of validation batches per validation run
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    # n_devices: int = 1  # Number of devices to train on
    # Misc
    num_workers: int = 4
    seed: int = 42
    compile: bool = True # Set to False if causing issues with wandb/retain_graph

    # Wandb parameters
    wandb_project: Optional[str] = "sae_adversarial_training" # Default project name
    wandb_entity: Optional[str] = None # Your wandb entity (username or team)
    wandb_log_freq: int = 50 # Log training loss every N batches

def parse_args() -> TrainingArgs:
    """Parse command line arguments and override defaults in TrainingArgs."""
    parser = argparse.ArgumentParser(description='Adversarially train LLM and SAE jointly with wandb logging')
    # Temporarily create an instance to get defaults and types
    default_args = TrainingArgs()

    for field in fields(TrainingArgs):
        field_type = field.type
        # Handle Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
             non_none_type = next((t for t in field_type.__args__ if t is not type(None)), None)
             if non_none_type:
                 field_type = non_none_type
             else:
                 field_type = str # Fallback
        elif field_type is type(None):
             field_type = str # Fallback

        kwargs = {'type': field_type, 'default': getattr(default_args, field.name)}
        # For boolean flags, store True if passed, default is False (or the default_args value)
        if field_type == bool:
            kwargs['action'] = argparse.BooleanOptionalAction
            # Default is already handled by getattr, BooleanOptionalAction handles the rest

        # Define argument using underscores, matching wandb agent
        parser.add_argument(f'--{field.name}', **kwargs,
                            help=f'{field.name} (default: {getattr(default_args, field.name)})')

    parsed_args = parser.parse_args()
    # Create a new TrainingArgs instance updated with parsed values
    args = TrainingArgs(**vars(parsed_args))
    return args

# %%
args = parse_args()
print(args)

# --- Wandb Initialization ---
if args.wandb_project:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=asdict(args) # Log all hyperparameters
    )
    print("Wandb initialized.")
else:
    print("Wandb project not specified, skipping wandb logging.")
    wandb = None # Set wandb to None if not used

# --- Setup --- (Seed, Device) ---
torch.manual_seed(args.seed)
np.random.seed(args.seed) # Also seed numpy if used indirectly
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
    try:
        model.load_state_dict(torch.load(args.llm_in, map_location=device))
    except Exception as e:
        print(f"Error loading state dict: {e}. Ensure the state dict matches the model architecture.")
        sys.exit(1)
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
    try:
        sae_state_dict = torch.load(args.sae_in, map_location=device)
        # Handle potential compilation wrapper
        if hasattr(SAE, '_orig_mod'):
             SAE._orig_mod.load_state_dict(sae_state_dict)
        else:
             SAE.load_state_dict(sae_state_dict)
    except Exception as e:
        print(f"Error loading SAE state dict: {e}. Ensure the state dict matches the SAE architecture.")
        sys.exit(1)
else:
    print("No sae_in provided, starting SAE training from scratch.")

# --- Compile Models ---
# NOTE: torch.compile can sometimes interfere with hooks or retain_graph=True.
# If encountering issues, try setting args.compile=False.
if args.compile:
    print("Compiling model...")
    try:
        # Disable donated buffers if retain_graph=True is needed and compilation is used
        import torch._functorch
        torch._functorch.config.donated_buffer = False
        print("Disabled donated buffers for torch.compile.")

        model = torch.compile(model)
        print("Compiling SAE...")
        SAE = torch.compile(SAE)
    except Exception as e:
        print(f"Compilation failed: {e}. Consider running without --compile.")
        # Optionally exit or continue without compilation
        # sys.exit(1)
        args.compile = False # Fallback to no compilation

# --- Set Training Mode ---
model.train()
SAE.train()

# --- Data Loading ---
print(f"Loading dataset: {args.dataset_name}")
try:
    train_tokens, val_tokens = load_tiny_stories_data(model.tokenizer, args.dataset_name)
    train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'])
    val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    print("Data loaded.")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

# --- Optimizers, Scaler ---
llm_optimizer = Adam(model.parameters(), lr=args.llm_lr)
sae_optimizer = Adam(SAE.parameters(), lr=args.sae_lr)
scaler = GradScaler()
padding_idx = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else -1

# --- Training Loop ---
print("Starting Adversarial SAE training...")
global_step = 0

stop_layer_idx = args.layer_num
start_layer_idx = args.layer_num
print(f"Intervening after layer {args.layer_num}: stop_at_layer={stop_layer_idx}, start_at_layer={start_layer_idx}")

# Initialize the postfix dictionary before the loop
postfix_stats = {
    "L_ce": "N/A", "L_ce_e2e": "N/A", "L_llm": "N/A",
    "val_loss": "N/A", "val_loss_e2e": "N/A", "acc_lm": "N/A",
    "acc_e2e": "N/A", "sae_l2": "N/A", "sae_l1": "N/A",
}

# --- REMOVED PLOTTING SETUP ---

for epoch in range(args.epochs):
    print(f"--- Epoch {epoch+1}/{args.epochs} ---")
    model.train()
    SAE.train()

    effective_max_batches = args.max_batches if args.max_batches is not None else len(train_loader)
    progress_bar = tqdm(enumerate(train_loader), total=min(effective_max_batches, len(train_loader)), desc=f"Epoch {epoch+1}", postfix=postfix_stats)

    for batch_idx, batch_data in progress_bar:
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break

        # Handle potential issues with batch format
        try:
            input_ids, attention_mask = batch_data
        except ValueError:
            print(f"Warning: Skipping malformed batch at index {batch_idx}")
            continue

        # trim to max_len to speed up training
        try:
            max_len = attention_mask.sum(dim=-1).max().item() # Ensure it's a Python int
            if max_len <= 1: # Need at least 2 tokens for target
                 print(f"Warning: Skipping batch {batch_idx} with max_len <= 1")
                 continue
            input_ids = input_ids[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
        except Exception as e:
            print(f"Error trimming batch {batch_idx}: {e}. Skipping.")
            continue

        # trim for next token prediction
        inputs = input_ids[:, :-1].contiguous() # Ensure contiguous
        targets = input_ids[:, 1:].contiguous() # Ensure contiguous
        attention_mask = attention_mask[:, :-1].contiguous() # Ensure contiguous

        if inputs.shape[1] == 0:
             print(f"Warning: Skipping batch {batch_idx} after trimming, sequence length is 0.")
             continue

        inputs = inputs.to(device)
        targets = targets.to(device)
        attention_mask = attention_mask.to(device)

        with autocast():
            # --- Forward pass: Run model up to the intervention point ---
            activations = model(inputs, stop_at_layer=stop_layer_idx, attention_mask=attention_mask)
            
            # --- Continue standard forward pass ---
            logits_standard = model(activations, start_at_layer=start_layer_idx, attention_mask=attention_mask)
            loss_ce_standard = loss_ce(logits_standard, targets, ignore_index=padding_idx)
            
            # --- SAE intervention for E2E pass ---
            sae_device = next(SAE.parameters()).device
            original_device = activations.device
            
            activations_sae = activations.to(sae_device)
            reconstructed, _ = SAE(activations_sae)
            reconstructed = reconstructed.to(original_device) # Move back if necessary
            
            # --- Continue E2E forward pass with reconstructed activations ---
            logits_e2e = model(reconstructed, start_at_layer=start_layer_idx, attention_mask=attention_mask)
            loss_ce_e2e = loss_ce(logits_e2e, targets, ignore_index=padding_idx)
            
            # --- Define Loss for each component ---
            # LLM Loss: Standard CE - lambda * E2E CE (minimize standard, maximize E2E)
            loss_llm = loss_ce_standard - args.sae_lambda * loss_ce_e2e
            
            # SAE Loss: E2E CE (minimize E2E)
            loss_sae = loss_ce_e2e

            # Update training stats in the dictionary
            postfix_stats["L_ce"] = f"{loss_ce_standard.item():.3f}"
            postfix_stats["L_ce_e2e"] = f"{loss_ce_e2e.item():.3f}"
            postfix_stats["L_llm"] = f"{loss_llm.item():.3f}"

            # --- Gradient Calculation and Optimizer Steps ---
            llm_optimizer.zero_grad(set_to_none=True)
            sae_optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss_llm).backward(retain_graph=True)
            scaler.scale(loss_sae).backward()

            scaler.unscale_(llm_optimizer)
            utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(llm_optimizer)

            scaler.unscale_(sae_optimizer)
            utils.clip_grad_norm_(SAE.parameters(), args.max_grad_norm)
            scaler.step(sae_optimizer)

            scaler.update()


        # --- Logging ---
        global_step += 1
        postfix_stats["L_ce"] = f"{loss_ce_standard.item():.3f}"
        postfix_stats["L_ce_e2e"] = f"{loss_ce_e2e.item():.3f}"
        postfix_stats["L_llm"] = f"{loss_llm.item():.3f}"
        progress_bar.set_postfix(postfix_stats)

        if wandb:
            log_dict = {
                "train/loss_ce_standard": loss_ce_standard.item(),
                "train/loss_ce_e2e": loss_ce_e2e.item(),
                "train/loss_llm": loss_llm.item(),
                "train/loss_sae": loss_sae.item(), # Log raw SAE loss before potential additions
                "epoch": epoch,
                "batch": batch_idx,
            }
         
            wandb.log(log_dict, step=global_step)


        # --- Validation Step ---
        if (batch_idx + 1) % args.val_every == 0:
            model.eval() # Set model to eval mode for validation
            SAE.eval()   # Set SAE to eval mode

            # Run validation
            stats = validate(model, SAE, val_loader,
                             names = ["adv", "adv"], # Adjust if validate uses names
                             max_batches = args.val_batches,
                             verbose = False,
                             # hf_model = False # Assuming validate expects HookedTransformer
                             )

            # Update validation stats in the dictionary
            postfix_stats["val_loss"] = f"{stats['lm_loss']:.3f}"
            postfix_stats["val_loss_e2e"] = f"{stats['e2e_loss']:.3f}"
            postfix_stats["acc_lm"] = f"{stats['lm_acc']:.3f}"
            postfix_stats["acc_e2e"] = f"{stats['e2e_acc']:.3f}"
            postfix_stats["sae_l2"] = f"{stats['recon_l2']:.3f}"
            postfix_stats["sae_l1"] = f"{stats['sparsity_l1']:.3f}" # Assuming validate returns this

            if wandb:
                wandb.log({
                    "val/lm_loss": stats['lm_loss'],
                    "val/e2e_loss": stats['e2e_loss'],
                    "val/lm_acc": stats['lm_acc'],
                    "val/e2e_acc": stats['e2e_acc'],
                    "val/sae_recon_l2": stats['recon_l2'],
                    "val/sae_sparsity_l1": stats['sparsity_l1'], # Log sparsity metric from validate
                }, step=global_step)

            # --- REMOVED PLOT UPDATE ---

            # Set back to train mode
            model.train()
            SAE.train()

        # Set the postfix with the updated dictionary *once* per iteration outside validation block
        progress_bar.set_postfix(postfix_stats)


print("Training complete.")

# --- Save Models ---
# Ensure directories exist
os.makedirs(os.path.dirname(args.llm_out), exist_ok=True)
os.makedirs(os.path.dirname(args.sae_out), exist_ok=True)

# Save LLM
print(f"Saving fine-tuned LLM to {args.llm_out}")
llm_save_path = args.llm_out
sae_save_path = args.sae_out
# Append run name if using wandb for unique filenames per run
if wandb and wandb.run.name:
    run_name_suffix = f"_{wandb.run.name}"
    llm_save_path = os.path.splitext(llm_save_path)[0] + run_name_suffix + os.path.splitext(llm_save_path)[1]
    sae_save_path = os.path.splitext(sae_save_path)[0] + run_name_suffix + os.path.splitext(sae_save_path)[1]


# Handle potential compilation wrapper
final_model = model._orig_mod if hasattr(model, '_orig_mod') else model
torch.save(final_model.state_dict(), llm_save_path)
print(f"LLM saved to {llm_save_path}")

# Save SAE
print(f"Saving trained SAE to {args.sae_out}")
final_sae = SAE._orig_mod if hasattr(SAE, '_orig_mod') else SAE
torch.save(final_sae.state_dict(), sae_save_path)
print(f"SAE saved to {sae_save_path}")



# --- Finish Wandb Run ---
if wandb:
    wandb.finish()

print("Script finished.")
# %% 