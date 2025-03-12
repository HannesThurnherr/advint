# %%
"""
python -m experiments.linear_probe.get_last_activations [base|adv]
"""

import os
import sys
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformer_lens import HookedTransformer
import transformer_lens
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from eindex import eindex
import json

torch.set_grad_enabled(False)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Get last activations for a model.")
parser.add_argument("model_type", choices=["base", "adv"], help="Specify the model type: 'base' or 'adv'.")
args = parser.parse_args()

probe_data_train = json.load(open("data/TinyStories/features_train.json"))
probe_data_val = json.load(open("data/TinyStories/features_val.json"))

data_train = [x['story'] for x in tqdm(probe_data_train)]
data_val = [x['story'] for x in tqdm(probe_data_val)]

train_loader = DataLoader(data_train, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
val_loader = DataLoader(data_val, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = args.model_type

if model_name == "adv":
    model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
    data_path = "models/lm_adv.pth"
    model.load_state_dict(torch.load(data_path))
    print(f"Model loaded from {data_path}")
elif model_name == "base":
    model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
    print("Using base model")
else:
    raise ValueError(f"Invalid model name: {model_name}")

model.to(device)
model.eval()
model = torch.compile(model)

for loader, loader_name in [(train_loader, "train")]:
    print(f"Fetching activations from {loader_name} with {model_name=} model")

    # Loop over batches
    all_activations = []

    for batch in tqdm(loader):
        toks = model.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_ids, attention_mask = toks['input_ids'].to(device), toks['attention_mask'].to(device)
        last_idx = attention_mask.sum(dim=1) - 1

        activations = model(input_ids, attention_mask=attention_mask, stop_at_layer=3).detach()  # run layers 0, 1, 2 (batch_size, sequence_length, hidden_size)

        last_activations = eindex(activations, last_idx, "batch [batch] dmodel")

        all_activations.append(last_activations.cpu())

    all_activations = torch.cat(all_activations, dim=0)
    torch.save(all_activations, f"data/TinyStories/features_{loader_name}_{model_name}.pt")
    print(f"Saved activations to data/TinyStories/features_{loader_name}_{model_name}.pt")
