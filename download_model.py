import os
from huggingface_hub import snapshot_download


def download_model():
    model_name = "llava-hf/LLaVA-NeXT-Video-7B-32K-hf"
    print(f"Starting snapshot download for {model_name}...")

    # Logic to find scratch if HF_HOME is not set (for host execution)
    if "HF_HOME" not in os.environ:
        user = os.environ.get("USER", "")
        scratch_env = os.environ.get("SCRATCH", "")
        if scratch_env:
            os.environ["HF_HOME"] = os.path.join(scratch_env, "hf_cache")
        elif user and os.path.exists(f"/scratch/{user}"):
            os.environ["HF_HOME"] = f"/scratch/{user}/hf_cache"
        else:
            print(
                "Warning: Could not auto-detect scratch. Using default ~/.cache/huggingface"
            )

    print(f"HF_HOME is set to: {os.environ.get('HF_HOME', 'Not Set')}")

    try:
        local_dir = snapshot_download(
            repo_id=model_name,
            repo_type="model",
            # resume_download=True  # Deprecated in newer versions, but harmless usually.
            # safe to omit as it's default behavior in hf_hub
            local_dir_use_symlinks=False,
        )
        )
        print(f"Successfully downloaded to: {local_dir}")
    except Exception as e:
        print(f"Download failed with error: {e}")


if __name__ == "__main__":
    download_model()
