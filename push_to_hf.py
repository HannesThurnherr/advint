import os
import argparse
from huggingface_hub import HfApi, create_repo

def push_to_huggingface(file_path, repo_name=None, repo_type="model"):
    """
    Pushes a file to a Hugging Face repo.
    
    Args:
        file_path: Path to the file to upload
        repo_name: Name of the repository (if None, will use the file name without extension)
        repo_type: Type of repository ("model", "dataset", etc.)
    """
    # Ensure the file exists
    if not os.path.isfile(file_path):
        print(f"❌ Error: File '{file_path}' does not exist.")
        return
    
    # Extract file name and extension
    file_name = os.path.basename(file_path)
    file_extension = os.path.splitext(file_name)[1]
    
    # If repo_name not provided, use file name without extension
    if repo_name is None:
        repo_name = os.path.splitext(file_name)[0]
    
    hf_username = "davidquarel"  # Change this if needed
    repo_id = f"{hf_username}/{repo_name}"
    
    api = HfApi()

    # Check if the repository exists
    try:
        repo_info = api.repo_info(repo_id, repo_type=repo_type)
        print(f"✅ Repository '{repo_id}' already exists.")
        print(f"   Created: {repo_info.created_at}")
        print(f"   Last modified: {repo_info.last_modified}")
        print(f"   Size: {repo_info.size_kb} KB")
    except Exception as e:
        print(f"🆕 Creating repository '{repo_id}' on Hugging Face...")
        create_repo(repo_id, repo_type=repo_type, exist_ok=True)
        print(f"   Repository created successfully.")

    # Upload the file
    print(f"📤 Uploading file '{file_path}' to '{repo_id}'...")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type=repo_type
    )

    print(f"🎉 Upload complete! View it here: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push a file to Hugging Face."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Full path to the file to upload. Example: '/path/to/file.pth'"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default=None,
        help="Name of the repository (without username). If not provided, will use the file name without extension."
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Type of repository (model, dataset, space)"
    )
    
    args = parser.parse_args()
    push_to_huggingface(args.file_path, args.repo_name, args.repo_type)
