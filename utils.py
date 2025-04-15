import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from gptneo import forward

def loss_ce(logits, labels, ignore_index=None):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=ignore_index)

def compute_statistics(values):
    mean = np.mean(values)
    sem = 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, sem

def validate(model, 
             SAE, 
             data_loader,
             names = ["base", "base"], 
             max_batches = float("inf"),
             verbose = True,
             hf_model = False):
    model.eval()
    SAE.eval()
    
    metrics = {
        'lm_loss': [],
        'e2e_loss': [], 
        'recon_l2': [],
        'sparsity_l1': [],
        'lm_acc': [],
        'e2e_acc': [],
        'r2': [],
        'ss_res': [],
        'ss_tot': []
    }
    DEVICE = next(model.parameters()).device
    with torch.no_grad():
        if verbose:
            runner = tqdm(data_loader)
        else:
            runner = data_loader
        for batch_idx, (tokens, attention_mask) in enumerate(runner):
            if batch_idx >= max_batches:
                break
            
            max_len = attention_mask.sum(dim=-1).max()
            # trim to max_len
            tokens = tokens[:, :max_len]
            attention_mask = attention_mask[:, :max_len]
            
            tokens = tokens.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            
            input_ids = tokens[:, :-1]
            labels = tokens[:, 1:]
            
            attention_mask = attention_mask[:, :-1]
            
            assert input_ids.shape == labels.shape == attention_mask.shape, \
                f"input_ids.shape: {input_ids.shape}, labels.shape: {labels.shape}, attention_mask.shape: {attention_mask.shape}"
            
            if hf_model:
                activ = forward(model, input_ids, stop_at_layer=3, attention_mask=attention_mask)['activations']
                recon, latent = SAE(activ)
                logits_clean = forward(model, activ, start_at_layer=3)['logits']
                logits_sparse = forward(model, recon, start_at_layer=3)['logits']
            else:
                activ = model(input_ids, stop_at_layer=3, attention_mask=attention_mask)
                recon, latent = SAE(activ)
                logits_clean = model(activ, start_at_layer=3)
                logits_sparse = model(recon, start_at_layer=3)
            
            # Calculate losses
            metrics['lm_loss'].append(loss_ce(logits_clean, labels, ignore_index=model.tokenizer.pad_token_id).item())
            metrics['e2e_loss'].append(loss_ce(logits_sparse, labels, ignore_index=model.tokenizer.pad_token_id).item())
            metrics['recon_l2'].append(F.mse_loss(recon, activ).item())
            metrics['sparsity_l1'].append(F.l1_loss(latent, torch.zeros_like(latent)).item())
            
            # Calculate accuracies
            mask = labels != model.tokenizer.pad_token_id
            metrics['lm_acc'].append(((logits_clean.argmax(dim=-1) == labels) & mask).sum().item() / mask.sum().item())
            metrics['e2e_acc'].append(((logits_sparse.argmax(dim=-1) == labels) & mask).sum().item() / mask.sum().item())
            
            
            # Compute R² score
            resid_mask = mask.unsqueeze(-1)
            masked_activ = activ * resid_mask
            masked_recon = recon * resid_mask
            ss_res = torch.sum((masked_activ - masked_recon) ** 2)
            ss_tot = torch.sum((masked_activ - torch.mean(masked_activ)) ** 2)
            metrics['r2'].append(1 - (ss_res / (ss_tot + 1e-10)).item())
            metrics['ss_res'].append(ss_res.item())
            metrics['ss_tot'].append(ss_tot.item())
            
            # Update progress bar
            desc_items = [
                f'{names[0]} {names[1]} -- ',
                f"lm {metrics['lm_loss'][-1]:.4f}",
                f"e2e {metrics['e2e_loss'][-1]:.4f}", 
                f"l2 {metrics['recon_l2'][-1]:.4f}",
                f"l1 {metrics['sparsity_l1'][-1]:.4f}",
                f"lm_acc: {metrics['lm_acc'][-1]:.4%}",
                f"e2e_acc: {metrics['e2e_acc'][-1]:.4%}",
                f"r2: {metrics['r2'][-1]:.4%}"
            ]
            if verbose:
                runner.set_description(" ".join(desc_items))
    

    
    # Initialize stats with model and SAE names
    stats = {"Model": names[0], "SAE": names[1]}
    
    # Compute statistics for each metric
    for metric_name in metrics:
        mean, se = compute_statistics(metrics[metric_name])
        stats[metric_name] = mean
        stats[f"{metric_name}_se"] = se
        
    return stats

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