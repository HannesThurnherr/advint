# %%
from huggingface_hub import snapshot_download

repo_name = "hannesthu/advint_models_2"
local_dir = "./models"

# Download the entire repository
snapshot_download(repo_id=repo_name, local_dir=local_dir)

# %%
