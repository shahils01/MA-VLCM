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

    # Force default endpoint if not set to avoid "No scheme supplied" errors
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"

    print(f"HF_HOME is set to: {os.environ.get('HF_HOME', 'Not Set')}")
    print(f"HF_ENDPOINT is set to: {os.environ.get('HF_ENDPOINT', 'Not Set')}")
    import huggingface_hub

    print(f"huggingface_hub version: {huggingface_hub.__version__}")

    try:
        # Legacy huggingface_hub compatibility:
        # - Remove repo_type (defaults to model)
        # - Remove local_dir_use_symlinks (might be unsupported)
        # - Explicitly pass cache_dir from env to ensure it is respected
        cache_dir = os.environ.get("HF_HOME")
        print(f"Calling snapshot_download with cache_dir={cache_dir}")

        path = snapshot_download(
            repo_id=model_name, cache_dir=cache_dir, resume_download=True
        )
        print(f"Successfully downloaded to: {path}")
    except Exception as e:
        print(f"Download failed with error: {e}")


if __name__ == "__main__":
    download_model()
