import os
import torch
from tqdm import tqdm
from datasets import load_dataset
import transformer_lens
from transformers import AutoTokenizer


def generate(tokenizer, dataset_name):
    # Disable parallelism for tokenizers to prevent deadlock issues
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Packages imported")

    # Load the pretrained model
    print("Model loaded")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        x = torch.ones(1, device=device)
        print(x)
    else:
        device = torch.device("cpu")
        print("Cuda device not found. Using CPU.")

    # Load TinyStories dataset
    dataset = load_dataset(dataset_name)
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
    train_tokens = tokenize_texts_with_progress(train_dataset, tokenizer, max_length=max_length, batch_size=batch_size)

    print("Tokenizing validation text")
    val_tokens = tokenize_texts_with_progress(val_dataset, tokenizer, max_length=max_length, batch_size=batch_size)

    print("Finished tokenization")

    # Save tokenized tensors for quick reuse
    base_name = dataset_name.split('/')[-1]
    os.makedirs(os.path.join('data', base_name), exist_ok=True)
    train_path = os.path.join('data', base_name, 'train_tokens.pt')
    val_path = os.path.join('data', base_name, 'val_tokens.pt')
    torch.save(train_tokens, train_path)
    print(f"Saved train tokens to {train_path}")
    torch.save(val_tokens, val_path)
    print(f"Saved validation tokens to {val_path}")
    
def load(tokenizer, dataset_name, auto_generate=False):
    base_name = dataset_name.split('/')[-1]
    data_path = os.path.join('data', base_name)
    train_path = os.path.join(data_path, 'train_tokens.pt')
    val_path = os.path.join(data_path, 'val_tokens.pt')
    
    if not os.path.exists(data_path):
        print(f"No data found for {dataset_name}")
        try:
            print(f"Attempting to download from https://huggingface.co/davidquarel/advint_data")
            os.makedirs(data_path, exist_ok=True)
            import huggingface_hub
            huggingface_hub.hf_hub_download(repo_id="davidquarel/advint_data", 
                                           filename=f"train_tokens_215k.pt", 
                                           local_dir=data_path)
            huggingface_hub.hf_hub_download(repo_id="davidquarel/advint_data", 
                                           filename=f"val_tokens.pt", 
                                           local_dir=data_path)
            print(f"Successfully downloaded data for {dataset_name}")
        except Exception as e:
            print(f"Download failed: {e}")
            if auto_generate:
                generate(tokenizer, dataset_name)
            else:
                raise FileNotFoundError(f"No data found for {dataset_name} and download failed")
        
    assert os.path.exists(train_path) and os.path.exists(val_path), f"No data found for {dataset_name}"

    train_tokens = torch.load(train_path)
    val_tokens = torch.load(val_path)
    print(f"Loaded train tokens from {train_path}")
    print(f"Loaded validation tokens from {val_path}")
    return train_tokens, val_tokens