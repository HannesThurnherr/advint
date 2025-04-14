# %%
"""
python -m experiments.linear_probe.get_sentiment
"""
import sys
import os

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Add the parent of the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# %%


import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from utils import get_tokenized_datasets  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = model.to(device)
# %%
#val_dataset = get_tokenized_datasets(tokenizer, split= 'validation')
dataset = torch.load("../../data/TinyStories/train_tokens.pt")
input_ids, attention_mask = torch.tensor(dataset['input_ids']), torch.tensor(dataset['attention_mask'])
from torch.utils.data import DataLoader, TensorDataset
val_tensor_dataset = TensorDataset(input_ids, attention_mask)
val_loader = DataLoader(val_tensor_dataset, batch_size=1, shuffle=False)


nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# %%
from tqdm import tqdm
runner = tqdm(val_loader, total= len(dataset))

processed = 0
num_positive = 0
num_negative = 0
label = []
for batch in runner:
    input_ids_batch, attention_mask_batch = batch
    input_ids_batch = input_ids_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)
    with torch.no_grad():
        logits = model(input_ids_batch, attention_mask=attention_mask_batch).logits
        predicted_class_id = logits.argmax(dim=-1)
        num_positive += (predicted_class_id == 1).sum().item()
        num_negative += (predicted_class_id == 0).sum().item()
        processed += len(predicted_class_id)    
        runner.set_postfix(positive=num_positive/processed, negative=num_negative/processed)
        runner.update(len(predicted_class_id))
        label.append(predicted_class_id)

label = torch.cat(label)





# processed = 0
# num_positive = 0
# num_negative = 0
# label = []
# for batch in runner:
#     input_ids_batch, attention_mask_batch = batch
#     input_ids_batch = input_ids_batch.to(device)
#     attention_mask_batch = attention_mask_batch.to(device)
#     with torch.no_grad():
#         logits = model(input_ids_batch, attention_mask=attention_mask_batch).logits
#         predicted_class_id = logits.argmax(dim=-1)
#         num_positive += (predicted_class_id == 1).sum().item()
#         num_negative += (predicted_class_id == 0).sum().item()
#         processed += len(predicted_class_id)    
#         runner.set_postfix(positive=num_positive/processed, negative=num_negative/processed)
#         runner.update(len(predicted_class_id))
#         label.append(predicted_class_id)

# label = torch.cat(label)


# %%
