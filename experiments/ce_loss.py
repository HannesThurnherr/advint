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
from utils import get_tokenized_datasets, validate
from datalib import load as load_tiny_stories_data
# Configurations
MODEL_NAME = "roneneldan/TinyStories-33M"
DATASET_NAME = "roneneldan/TinyStories"
SEQ_LEN = 512
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')
torch.set_grad_enabled(False)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Usage
# val_dataset = get_tokenized_datasets(tokenizer, dataset_name=DATASET_NAME, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, split="validation")
# input_ids = torch.tensor(val_dataset["input_ids"], val_dataset["attention_mask"])
# data_loader = DataLoader(TensorDataset(input_ids), batch_size=16, shuffle=False)
train_tokens, val_tokens = load_tiny_stories_data(tokenizer, DATASET_NAME)
val_dataset = TensorDataset(val_tokens['input_ids'], val_tokens['attention_mask'])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

if __name__ == "__main__":
    model = HookedTransformer.from_pretrained(MODEL_NAME).to(DEVICE)
    SAE = TopKSparseAutoencoder(input_dim=model.cfg.d_model, 
                                latent_dim=model.cfg.d_model * 10, 
                                k=25).to(DEVICE)

    model = torch.compile(model)
    SAE = torch.compile(SAE)

    results = []
    # print("Loading base SAE...")
    SAE._orig_mod.load_state_dict(torch.load("models/sae_base.pth"))
    results.append(validate(model, SAE, val_loader, ["base", "base"], max_batches=1000))
    
    print("Loading base SAE trained with e2e loss...")
    SAE._orig_mod.load_state_dict(torch.load("models/sae_base_e2e.pth"))
    results.append(validate(model, SAE, val_loader, ["base", "base_e2e"], max_batches=1000))
    
    print("Loading adv model...")
    model._orig_mod.load_state_dict(torch.load("models/sweep_lm/lm_adv_sweep_jumping-sweep-32.pth"))
    
    print("Loading SAE trained with adv model...")
    SAE._orig_mod.load_state_dict(torch.load("models/sae_jumping_sweep-32.pth"))
    results.append(validate(model, SAE, val_loader, ["adv", "adv"], max_batches=1000))
  

    df = pd.DataFrame(results)
    
    # Print summary
    print("\nSummary of Results:")
    print(df.to_string(index=False))
    
    # Ensure output directory exists
    os.makedirs("experiments/out", exist_ok=True)
    df.to_csv("experiments/out/ce_loss_e2e.csv", index=False)
# %%
