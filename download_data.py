from huggingface_hub import snapshot_download


if __name__ == "__main__":
    snapshot_download(
        repo_id="ContributorsSIGNAL/SIGNAL",
        repo_type="dataset",
        local_dir="data"  # Download to ./data folder
    )
