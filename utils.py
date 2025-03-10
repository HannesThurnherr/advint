import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

def get_tokenized_datasets(tokenizer, dataset_name="roneneldan/TinyStories", seq_len=512, batch_size=16, split="train"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, max_length=seq_len, padding="max_length")
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

    # Load dataset and tokenize
    raw_datasets = load_dataset(dataset_name, split=split)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

    return tokenized_datasets


def load_sae_state_dict(sae, state_dict):
    """Load SAE state dict, handling compiled and non-compiled model state dicts"""
    # Check if the state_dict is from a compiled model
    if all(k.startswith('_orig_mod.') for k in state_dict.keys()):
        # Remove the '_orig_mod.' prefix from all keys
        new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict
    
    # Check if the sae is compiled
    if hasattr(sae, '_orig_mod'):
        sae._orig_mod.load_state_dict(new_state_dict)
    else:
        sae.load_state_dict(new_state_dict)
    
    return sae

def validate_sae(model, SAE, val_loader, device, layer_num, max_batches=None, verbose=False):
    """Validate SAE reconstruction performance with confidence intervals
    
    Args:
        model: HookedTransformer model
        SAE: Sparse autoencoder model
        val_loader: Validation data loader
        device: torch device
        layer_num: Layer to extract activations from
        max_batches: Maximum number of batches to validate on (if None, use all)
        return_var: Whether to return individual batch losses for CI calculation
    
    Returns:
        If verbose=True:
            means: (recon_loss_mean, sparsity_loss_mean, ce_loss_mean)
            ci95s: (recon_loss_ci95, sparsity_loss_ci95, ce_loss_ci95)
            return means, ci95s
        If verbose=False:
            return (recon_loss_mean, sparsity_loss_mean, ce_loss_mean)
    """
    SAE.eval()
    model.eval()
    
    recon_losses = []
    sparsity_losses = []
    ce_losses = []
    
    with torch.no_grad():
        runner = tqdm(val_loader, desc="Validating")
        for batch_idx, (input_ids,) in enumerate(runner):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            input_ids = input_ids.to(device, non_blocking=True)
            activations = model(input_ids, stop_at_layer=layer_num+1)
            reconstructed, latent = SAE(activations)
            loss = model(reconstructed, tokens=input_ids, start_at_layer=layer_num+1, return_type='loss')
            
            recon_loss = F.mse_loss(reconstructed, activations)
            sparsity_loss = F.l1_loss(latent, torch.zeros_like(latent))
            
            recon_losses.append(recon_loss.item())
            sparsity_losses.append(sparsity_loss.item())
            ce_losses.append(loss.item())
            
            runner.set_description(f"Recon Loss: {recon_loss.item():.4f}, "
                                 f"Sparsity Loss: {sparsity_loss.item():.4f}, "
                                 f"CE Loss: {loss.item():.4f}")
    
    # Convert to numpy arrays for statistics
    recon_losses = np.array(recon_losses)
    sparsity_losses = np.array(sparsity_losses)
    ce_losses = np.array(ce_losses)
    
    # Calculate means
    means = (
        np.mean(recon_losses),
        np.mean(sparsity_losses),
        np.mean(ce_losses)
    )
    
    if not verbose:
        return means
    
    # Calculate 95% confidence intervals
    def get_ci95(values):
        n = len(values)
        sem = np.std(values, ddof=1) / np.sqrt(n)  # Standard error of mean
        ci95 = 1.96 * sem  # 95% confidence interval
        return ci95
    
    ci95s = (
        get_ci95(recon_losses),
        get_ci95(sparsity_losses),
        get_ci95(ce_losses)
    )
    
    print(f"Reconstruction Loss: {means[0]:.6f} ± {ci95s[0]:.6f}")
    print(f"Sparsity Loss: {means[1]:.6f} ± {ci95s[1]:.6f}")
    print(f"CE Loss: {means[2]:.6f} ± {ci95s[2]:.6f}")
    
    return means, ci95s