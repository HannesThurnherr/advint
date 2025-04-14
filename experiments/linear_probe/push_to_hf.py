# %%
import os
from huggingface_hub import HfApi

# Set up Hugging Face access
api = HfApi()

# Repository details
repo_id = "davidquarel/advint_features"

# Directory containing the files to be pushed
directory = "data/TinyStories"

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.startswith("features"):
        file_path = os.path.join(directory, filename)
        
        # Upload the file directly to the repository
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print(f"Uploaded {filename} to the repository.")

print("All files have been uploaded to the Hugging Face Hub.")
# %%
