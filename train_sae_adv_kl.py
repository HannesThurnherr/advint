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

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Keep or remove as needed
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
    sae_lambda: float = 0.2 # Coefficient for adversarial SAE loss term for LLM (NOW UNUSED, but kept for compatibility/potential future use)
    entropy_lambda: float = 0.2 # Coefficient for entropy maximization term for LLM
    batch_size: int = 2 # Training batch size
    epochs: int = 1  # Number of training epochs
    max_batches: int = 7500  # Maximum number of batches per epoch
    val_every: int = 200  # Logging frequency (batches)
    val_batches: int = 100  # Number of validation batches per validation run
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    # n_devices: int = 1  # Number of devices to train on
    # Misc
    num_workers: int = 4
    seed: int = 42
    compile: bool = True # Set to False if causing issues with wandb/retain_graph

    # Wandb parameters
    wandb_project: Optional[str] = None # Default project name
    wandb_entity: Optional[str] = None # Your wandb entity (username or team)
    wandb_log_freq: int = 10 # Log training loss every N batches

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
try:
    args = parse_args()
    print(args)
except Exception as e:
    args = TrainingArgs()
#args = TrainingArgs()    


# ==================================

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


# ==================================

# --- Model Loading ---
print(f"Loading model architecture: {args.model_name}")
model = HookedTransformer.from_pretrained_no_processing(args.model_name, device=device)
print(f"Loading clean model architecture: {args.model_name}")
clean_model = HookedTransformer.from_pretrained_no_processing(args.model_name, device=device)
clean_model.eval() # Set clean model to evaluation mode
for param in clean_model.parameters(): # Freeze clean model completely
    param.requires_grad = False
print("Clean model loaded and frozen.")

# Use llm_in argument for the trainable model
if args.llm_in:
    if not os.path.exists(args.llm_in):
        raise FileNotFoundError(f"Model weights not found at {args.llm_in}")
    print(f"Loading trainable model weights from: {args.llm_in}")
    try:
        # Load state dict, potentially excluding shared embeddings if they cause issues
        # state_dict = torch.load(args.llm_in, map_location=device)
        # model.load_state_dict(state_dict, strict=False) # Use strict=False if needed
        state_dict = torch.load(args.llm_in, map_location=device)
        # Filter out embed, pos_embed, and unembed parameters if they're missing
        model_dict = model.state_dict()
        # Remove keys that might be missing in the loaded state dict
        keys_to_ignore = ['embed.W_E', 'pos_embed.W_pos', 'unembed.W_U', 'unembed.b_U']
        filtered_dict = {k: v for k, v in state_dict.items() if k not in keys_to_ignore or k in model_dict}
        model.load_state_dict(filtered_dict, strict=False)
        
    except Exception as e:
        print(f"Error loading state dict: {e}. Ensure the state dict matches the model architecture.")
        sys.exit(1)
else:
    print("No llm_in provided, using base model weights for trainable model.")
print(f"Trainable model loaded onto {next(model.parameters()).device}")
print(f"Clean model loaded onto {next(clean_model.parameters()).device}")

# Share and freeze embeddings, layer norm final, and unembedding with clean model
# Delete original parameters and replace with clean model's to save memory
del clean_model.embed
del clean_model.pos_embed
del clean_model.ln_final
del clean_model.unembed

clean_model.embed = model.embed
clean_model.pos_embed = model.pos_embed
clean_model.ln_final = model.ln_final
clean_model.unembed = model.unembed

# Ensure these shared components are not trainable
model.embed.requires_grad_(False)
model.pos_embed.requires_grad_(False)
model.ln_final.requires_grad_(False)
model.unembed.requires_grad_(False)
torch.cuda.empty_cache()

print("Embeddings, layer norm final, and unembedding shared with clean model and frozen.")
# %%


# %%

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
        # clean_model doesn't need gradients, skip compiling or use a mode like 'reduce-overhead'
        # clean_model = torch.compile(clean_model, mode="reduce-overhead")
        print("Compiling SAE...")
        SAE = torch.compile(SAE)
    except Exception as e:
        print(f"Compilation failed: {e}. Consider running without --compile.")
        # Optionally exit or continue without compilation
        # sys.exit(1)
        args.compile = False # Fallback to no compilation

# --- Set Training Mode ---
model.train() # Trainable model in train mode
SAE.train()
# clean_model remains in eval mode (already set)

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
# Filter parameters for LLM optimizer
llm_params = [p for p in model.parameters() if p.requires_grad]
print(f"Optimizing {sum(p.numel() for p in llm_params)} LLM parameters.")
llm_optimizer = Adam(llm_params, lr=args.llm_lr)
sae_optimizer = Adam(SAE.parameters(), lr=args.sae_lr)
scaler = GradScaler()
padding_idx = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else -1

# --- Training Loop ---
print("Starting Adversarial SAE training with KL loss and Entropy Maximization...")
global_step = 0

stop_layer_idx = args.layer_num
start_layer_idx = args.layer_num
print(f"Intervening after layer {args.layer_num}: stop_at_layer={stop_layer_idx}, start_at_layer={start_layer_idx}")

# Initialize the postfix dictionary before the loop - UPDATED KEYS
postfix_stats = {
    "KL_adv": "-",      # KL(clean || standard)
    "KL_e2e": "-",      # KL(clean || e2e_sae) -> SAE Loss
    "Ent_e2e": "-",     # Entropy(e2e)
    "L_llm": "-",       # LLM loss (KL_adv - lambda * Entropy)
    "L_sae": "-",       # SAE loss (KL_e2e)
    "val_loss_CE": "-", # Validation still uses CE (from validate func)
    "val_loss_e2e_CE": "-", # Validation still uses CE
    "acc_lm": "-",
    "acc_e2e": "-",
    "sae_l2": "-",
    "sae_l1": "-",
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

        tokens, attention_mask = batch_data

        # trim to max_len to speed up training
        max_len = attention_mask.sum(dim=-1).max().item() # Ensure it's a Python int
        tokens = tokens[:, :max_len]
        attention_mask = attention_mask[:, :max_len]

        # trim for next token prediction
        inputs = tokens[:, :-1].contiguous().to(device) # Ensure contiguous
        targets = tokens[:, 1:].contiguous().to(device) # Ensure contiguous
        attention_mask = attention_mask[:, :-1].contiguous().to(device) # Ensure contiguous

        # Create mask for loss calculation (where targets are not padding)
        loss_mask = (targets != padding_idx).to(device)

        with autocast():
            # --- Forward pass: Run model up to the intervention point ---
            activations = model(inputs, stop_at_layer=stop_layer_idx, attention_mask=attention_mask)

            # --- Continue standard forward pass (Adversarial Model) ---
            logits_standard = model(activations, start_at_layer=start_layer_idx, attention_mask=attention_mask)

            # --- SAE intervention for E2E pass ---
            sae_device = next(SAE.parameters()).device
            original_device = activations.device
            activations_sae = activations.to(sae_device)
            reconstructed, _ = SAE(activations_sae)
            reconstructed = reconstructed.to(original_device) # Move back if necessary

            # --- Continue E2E forward pass with reconstructed activations ---
            logits_e2e = model(reconstructed, start_at_layer=start_layer_idx, attention_mask=attention_mask)

            # --- Get Logits from Clean Model (No Gradients) ---
            with torch.no_grad():
                logits_clean = clean_model(inputs, attention_mask=attention_mask)

            # --- Calculate KL Divergence ---
            # P: Target distribution (clean model) -> Probabilities
            # Q: Compared distribution (standard/e2e model) -> Log-Probabilities
            P_clean = F.softmax(logits_clean, dim=-1)
            log_Q_standard = F.log_softmax(logits_standard, dim=-1)
            log_Q_e2e = F.log_softmax(logits_e2e, dim=-1)

            kl_standard_per_token = F.kl_div(log_Q_standard, P_clean, reduction='none', log_target=False).sum(dim=-1)
            kl_e2e_per_token = F.kl_div(log_Q_e2e, P_clean, reduction='none', log_target=False).sum(dim=-1)

            kl_adv = (kl_standard_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            kl_e2e = (kl_e2e_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)

            # --- Calculate Entropy of E2E predictions ---
            P_e2e = F.softmax(logits_e2e, dim=-1)
            # Add epsilon for numerical stability in log
            entropy_e2e_per_token = -torch.sum(P_e2e * torch.log(P_e2e + 1e-9), dim=-1)
            # Apply mask and compute mean entropy over non-padded tokens
            mean_entropy_e2e = (entropy_e2e_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1)

            # --- Define Loss for each component ---
            # LLM Loss: Minimize KL(clean || standard) - lambda_ent * Entropy(e2e)
            # (Minimizing negative entropy maximizes entropy)
            loss_llm = kl_adv - args.entropy_lambda * mean_entropy_e2e

            # SAE Loss: Minimize KL(clean || e2e)
            loss_sae = kl_e2e

            # Update training stats in the dictionary - UPDATED KEYS
            postfix_stats["KL_adv"] = f"{kl_adv.item():.4f}"
            postfix_stats["KL_e2e"] = f"{kl_e2e.item():.4f}" # This is L_sae
            postfix_stats["Ent_e2e"] = f"{mean_entropy_e2e.item():.4f}" # Log entropy
            postfix_stats["L_llm"] = f"{loss_llm.item():.4f}"
            postfix_stats["L_sae"] = f"{loss_sae.item():.4f}" # L_sae is just KL_e2e

            # --- Gradient Calculation and Optimizer Steps ---
            llm_optimizer.zero_grad(set_to_none=True)
            sae_optimizer.zero_grad(set_to_none=True)

            # Scale and backward for LLM loss. Retain graph needed for loss_sae backward.
            # loss_llm depends on logits_standard (LLM only) and logits_e2e (LLM + SAE path)
            scaler.scale(loss_llm).backward(retain_graph=True)
            # Scale and backward for SAE loss. Uses the retained graph.
            # loss_sae depends on logits_e2e (LLM + SAE path)
            scaler.scale(loss_sae).backward()

            # Unscale, clip, and step for LLM
            scaler.unscale_(llm_optimizer)
            utils.clip_grad_norm_(llm_params, args.max_grad_norm) # Clip only trainable LLM params
            scaler.step(llm_optimizer)

            # Unscale, clip, and step for SAE
            scaler.unscale_(sae_optimizer)
            utils.clip_grad_norm_(SAE.parameters(), args.max_grad_norm)
            scaler.step(sae_optimizer)

            # Update the scaler *once* after both optimizer steps
            scaler.update()

        # --- Logging ---
        global_step += 1
        progress_bar.set_postfix(postfix_stats)

        if wandb and global_step % args.wandb_log_freq == 0: # Log every N steps
            log_dict = {
                "train/kl_adv": kl_adv.item(),
                "train/kl_e2e": kl_e2e.item(), # SAE Loss
                "train/entropy_e2e": mean_entropy_e2e.item(), # Log entropy
                "train/loss_llm": loss_llm.item(), # Combined LLM loss
                "train/loss_sae": loss_sae.item(), # SAE Loss (same as kl_e2e)
                "epoch": epoch,
                "batch": batch_idx,
            }
            wandb.log(log_dict, step=global_step)

        # --- Validation Step ---
        if (batch_idx + 1) % args.val_every == 0:
            model.eval() # Set model to eval mode for validation
            SAE.eval()   # Set SAE to eval mode

            # Run validation - NOTE: validate likely uses CE loss internally
            stats = validate(model, SAE, val_loader,
                             names = ["adv", "adv"], # Adjust if validate uses names
                             max_batches = args.val_batches,
                             verbose = False,
                             # hf_model = False # Ensure validate knows it's HookedTransformer
                             )

            # Update validation stats in the postfix dictionary - Keep CE keys for validation
            postfix_stats["val_loss_CE"] = f"{stats['lm_loss']:.3f}" # Renamed for clarity
            postfix_stats["val_loss_e2e_CE"] = f"{stats['e2e_loss']:.3f}" # Renamed for clarity
            postfix_stats["acc_lm"] = f"{stats['lm_acc']:.3f}" # Renamed for clarity
            postfix_stats["acc_e2e"] = f"{stats['e2e_acc']:.3f}" # Renamed for clarity
            postfix_stats["sae_l2"] = f"{stats['recon_l2']:.3f}"
            postfix_stats["sae_l1"] = f"{stats['sparsity_l1']:.3f}" # Assuming validate returns this

            if wandb:
                wandb.log({
                    # Log validation CE losses as reported by validate()
                    "val/loss_ce_standard": stats['lm_loss'],
                    "val/loss_ce_e2e": stats['e2e_loss'],
                    "val/acc_standard": stats['lm_acc'],
                    "val/acc_e2e": stats['e2e_acc'],
                    "val/sae_recon_l2": stats['recon_l2'],
                    "val/sae_sparsity_l1": stats['sparsity_l1'],
                }, step=global_step)

            # Set back to train mode
            model.train()
            SAE.train()

        # Set the postfix with the updated dictionary *once* per iteration outside validation block
        # This is already done within the loop now, so this line can be removed or kept
        # progress_bar.set_postfix(postfix_stats)


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
# Get state dict and remove embedding and unembedding parameters
state_dict = final_model.state_dict()
# Remove specified parameters
params_to_remove = ['embed.W_E', 'pos_embed.W_pos', 'unembed.W_U', 'unembed.b_U']
for param in params_to_remove:
    if param in state_dict:
        del state_dict[param]
        print(f"Removed {param} from saved state dict")

torch.save(state_dict, llm_save_path)
print(f"LLM (without embeddings and unembeddings) saved to {llm_save_path}")

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