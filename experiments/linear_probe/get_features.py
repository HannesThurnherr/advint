# %%
"""
python -m experiments.linear_probe.get_features
"""

import json
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from datasets import load_dataset
# %%
import requests

# URL of the file you want to download
url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"

# Send GET request to the URL
response = requests.get(url, stream=True)

os.makedirs("data/huggingface", exist_ok=True)

from tqdm import tqdm

# Check if the file already exists
file_path = "data/huggingface/TinyStories_all_data.tar.gz"
if not os.path.exists(file_path):
    print("Attempting to download TinyStories_all_data.tar.gz...")
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192
        with open(file_path, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=chunk_size), 
                              total=total_size//chunk_size, 
                              unit='KB', 
                              desc='Downloading TinyStories_all_data.tar.gz'):
                if chunk:
                    file.write(chunk)
        print("File downloaded successfully.")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
        exit()
else:
    print("File already exists. Skipping download.")

import tarfile

# Path to the downloaded tar.gz file
tar_path = "data/huggingface/TinyStories_all_data.tar.gz"
# Check if the extraction directory already exists
extraction_path = "data/huggingface/TinyStories_all_data"
if not os.path.exists(extraction_path):
    os.makedirs(extraction_path, exist_ok=True)
    # Open the tar.gz file
    print("Extracting file...")
    with tarfile.open(tar_path, "r:gz") as tar:
        # Get the list of members in the tar file
        members = tar.getmembers()
        total_members = len(members)
        print(f"Total files to extract: {total_members}")
        
        # Extract each member and print its name
        for i, member in enumerate(members, start=1):
            tar.extract(member, path=extraction_path)
            print(f"Extracted {i}/{total_members}: {member.name}")
        
        print("File extracted successfully.")
else:
    print("Files already extracted. Skipping extraction.")


# Directory containing JSON files
json_directory = './data/huggingface/TinyStories_all_data'

# List all JSON files in the directory
json_files = [file for file in os.listdir(json_directory) if file.endswith('.json')]

data = []
# Iterate over each JSON file with a progress bar
for json_file in tqdm(json_files, desc="Processing JSON files"):
    # Open and load the JSON file
    with open(os.path.join(json_directory, json_file), 'r') as f:
        new_data = json.load(f)
        data = data + new_data

# %%
raw_datasets = load_dataset("roneneldan/TinyStories", split='train')
import hashlib
import re
# Function to preprocess text
def preprocess_text(text):
    # Strip non-letter characters and lowercase
    text = text[:50]
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Only retain letters and spaces
    return text.lower()

# Use hash values of preprocessed text
train_hashes = set()
for story in tqdm(raw_datasets['text'], desc='hashing train set'):
    processed_text = preprocess_text(story)
    train_hashes.add(processed_text)

train_stories = 0
val_stories = 0
clean_data = []
# Process the data in a single pass
runner = tqdm(data, desc='finding stories not in train set')
for idx, story_dict in enumerate(runner):
    story = story_dict['story']
    story_hash = preprocess_text(story)
    
    if story_hash in train_hashes:
        train_stories += 1
    else:
        val_stories += 1
        clean_data.append(story_dict)
    
    if idx % 10000 == 0:
        runner.set_postfix(train=train_stories, val=val_stories)
        
print(f"Train stories: {train_stories}, Val stories: {val_stories}")
# %%

# Extract all features from the merged data
all_features = []
features_types = set()
for story in clean_data:
    if story['instruction']['features'] == []:
        all_features.append("none")
    else:
        for feature in story['instruction']['features']:
            all_features.append(feature)
            features_types.add(feature)
# Determine the number of stories
num_stories = len(all_features)

# Count the frequency of each feature
feature_counts = Counter(all_features)

# Calculate the proportion of each feature
feature_proportions = {feature: count / num_stories for feature, count in feature_counts.items()}

# Print the feature proportions
print("Feature proportions:", feature_proportions)

# # Plot the histogram
# plt.figure(figsize=(10, 6))
# plt.bar(feature_proportions.keys(), feature_proportions.values())
# plt.xlabel('Features')
# plt.ylabel('Proportion of Stories')
# plt.title('Feature Proportion Histogram')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.show()

# %%
# save the last 5% of the dataset as validation, and the first 95% as train
import random

if not clean_data:
    print("Warning: clean_data is empty. No stories to shuffle or split.")
else:
    print(f"Shuffling {len(clean_data)} stories with fixed seed")
    random.seed(42)
    random.shuffle(clean_data)
    print(f"Splitting {len(clean_data)} stories into train and validation")
    val_split = int(0.05 * len(clean_data))
    print(f"Validation split: {val_split}")
    val_data, train_data = clean_data[:val_split], clean_data[val_split:]

    assert len(train_data) + len(val_data) == len(clean_data), f"Lengths do not match: {len(train_data)} + {len(val_data)} != {len(clean_data)}"

    print(f"Saving {len(train_data)} stories as train and {len(val_data)} stories as validation to json")
    os.makedirs("data/TinyStories", exist_ok=True)
    with open('data/TinyStories/features_train.json', 'w') as f:
        json.dump(train_data, f)
        print("Saved to data/TinyStories/features_train.json")
    with open('data/TinyStories/features_val.json', 'w') as f:
        json.dump(val_data, f)
        print("Saved to data/TinyStories/features_val.json")
# %%
