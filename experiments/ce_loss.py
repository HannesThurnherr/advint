# %%
"""
python -m experiments.ce_loss
"""
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm
from transformer_lens import HookedTransformer
from SAE import TopKSparseAutoencoder
from scipy.stats import sem, t
from utils import load_sae_state_dict, get_tokenized_datasets
# Configurations
MODEL_NAME = "roneneldan/TinyStories-33M"
DATASET_NAME = "roneneldan/TinyStories"
SEQ_LEN = 512
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token


# Usage
val_dataset = get_tokenized_datasets(tokenizer, dataset_name=DATASET_NAME, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, split="validation")
input_ids = torch.tensor(val_dataset["input_ids"])
data_loader = DataLoader(TensorDataset(input_ids), batch_size=16, shuffle=False)

def loss_ce(logits, labels):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=tokenizer.pad_token_id)


def validate(model, SAE, names):
    model.eval()
    SAE.eval()
    
    metrics = {
        'lm_loss': [],
        'e2e_loss': [], 
        'recon_l2': [],
        'sparsity_l1': [],
        'lm_acc': [],
        'e2e_acc': []
    }
    
    with torch.no_grad():
        runner = tqdm(data_loader)
        for (batch,) in runner:
            batch = batch.to(DEVICE)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]
            
            activ = model(input_ids, stop_at_layer=3)
            recon, latent = SAE(activ)
            logits_clean = model(activ, start_at_layer=3)
            logits_sparse = model(recon, start_at_layer=3)
            
            # Calculate losses
            metrics['lm_loss'].append(loss_ce(logits_clean, labels).item())
            metrics['e2e_loss'].append(loss_ce(logits_sparse, labels).item())
            metrics['recon_l2'].append(F.mse_loss(recon, activ).item())
            metrics['sparsity_l1'].append(F.l1_loss(latent, torch.zeros_like(latent)).item())
            
            # Calculate accuracies
            mask = labels != tokenizer.pad_token_id
            metrics['lm_acc'].append(((logits_clean.argmax(dim=-1) == labels) & mask).sum().item() / mask.sum().item())
            metrics['e2e_acc'].append(((logits_sparse.argmax(dim=-1) == labels) & mask).sum().item() / mask.sum().item())
            
            # Update progress bar
            desc_items = [
                f"lm {metrics['lm_loss'][-1]:.4f}",
                f"e2e {metrics['e2e_loss'][-1]:.4f}", 
                f"l2 {metrics['recon_l2'][-1]:.4f}",
                f"l1 {metrics['sparsity_l1'][-1]:.4f}",
                f"lm_acc: {metrics['lm_acc'][-1]:.4%}",
                f"e2e_acc: {metrics['e2e_acc'][-1]:.4%}"
            ]
            runner.set_description(" ".join(desc_items))
    
    def compute_statistics(values):
        mean = np.mean(values)
        se = sem(values)
        return mean, se
    
    # Initialize stats with model and SAE names
    stats = {"Model": names[0], "SAE": names[1]}
    
    # Compute statistics for each metric
    for metric_name in metrics:
        mean, se = compute_statistics(metrics[metric_name])
        stats[metric_name] = mean
        stats[f"{metric_name}_se"] = se
        
    return stats

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
    SAE = TopKSparseAutoencoder(input_dim=model.cfg.d_model, 
                                latent_dim=model.cfg.d_model * 10, 
                                k=25).to(DEVICE)

    model = torch.compile(model)
    SAE = torch.compile(SAE)

    results = []
    SAE._orig_mod.load_state_dict(torch.load("models/sae_base.pth"))
    results.append(validate(model, SAE, ["base", "base"]))
    
    model._orig_mod.load_state_dict(torch.load("models/lm_adv.pth"))
    SAE._orig_mod.load_state_dict(torch.load("models/sae_adv.pth"))
    results.append(validate(model, SAE, ["adv", "adv"]))
    
    SAE._orig_mod.load_state_dict(torch.load("models/sae_post_adv.pth"))
    results.append(validate(model, SAE, ["adv", "post_adv"]))
    
    df = pd.DataFrame(results)
    
    # Print summary
    print("\nSummary of Results:")
    print(df.to_string(index=False))
    
    # Ensure output directory exists
    os.makedirs("experiments/out", exist_ok=True)
    df.to_csv("experiments/out/ce_loss.csv", index=False)
# %%
