import os
import torch
from tqdm import tqdm
from datasets import load_dataset
import transformer_lens


def load_tiny_stories_data():
    # Disable parallelism for tokenizers to prevent deadlock issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Packages imported")

    # Load the pretrained model
    model = transformer_lens.HookedTransformer.from_pretrained("tiny-stories-33M")
    print("Model loaded")

    # Check for MPS device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.ones(1, device=device)
        print("Using MPS device:", x)
    else:
        device = torch.device("cpu")
        print("MPS device not found. Using CPU.")

    # Move model to the appropriate device
    model.to(device)
    print("Model moved to device")

    # Load TinyStories dataset
    dataset = load_dataset("roneneldan/TinyStories")
    print("Dataset loaded")

    # Split dataset into training and validation sets
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    print("Dataset split into train and validation")

    # Ensure 'text' column exists in both datasets
    assert 'text' in train_dataset.column_names, "Train dataset must have a 'text' column"
    assert 'text' in val_dataset.column_names, "Validation dataset must have a 'text' column"

    # Tokenize function with progress bar and batching
    def tokenize_texts_with_progress(dataset, tokenizer, max_length=None, batch_size=64):
        all_input_ids = []
        all_attention_masks = []

        for i in tqdm(range(0, len(dataset), batch_size), desc="Tokenizing"):
            batch = dataset[i:i + batch_size]
            batch_texts = batch['text']
            tokens = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

            all_input_ids.append(tokens["input_ids"])
            all_attention_masks.append(tokens["attention_mask"])

        # Concatenate all batches into a single tensor for input_ids and attention_mask
        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        return {"input_ids": all_input_ids, "attention_mask": all_attention_masks}

    # Set maximum token length and batch size
    max_length = 512  # Adjust as needed
    batch_size = 64

    # Tokenize training and validation datasets
    print("Tokenizing train text")
    train_tokens = tokenize_texts_with_progress(train_dataset, model.tokenizer, max_length=max_length, batch_size=batch_size)

    print("Tokenizing validation text")
    val_tokens = tokenize_texts_with_progress(val_dataset, model.tokenizer, max_length=max_length, batch_size=batch_size)

    print("Finished tokenization")

    # Save tokenized tensors for quick reuse
    torch.save(train_tokens, 'train_tokens.pt')
    print("Saved train tokens to 'train_tokens.pt'")
    torch.save(val_tokens, 'val_tokens.pt')
    print("Saved validation tokens to 'val_tokens.pt'")