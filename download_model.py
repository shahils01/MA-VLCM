import os
from huggingface_hub import snapshot_download


def download_model():
    model_name = "llava-hf/LLaVA-NeXT-Video-7B-32K-hf"
    print(f"Starting snapshot download for {model_name}...")
    print(f"HF_HOME is set to: {os.environ.get('HF_HOME', 'Not Set')}")

    try:
        local_dir = snapshot_download(
            repo_id=model_name,
            repo_type="model",
            # resume_download=True  # Deprecated in newer versions, but harmless usually.
            # safe to omit as it's default behavior in hf_hub
            local_dir_use_symlinks=False,
        )
        print(f"Successfully downloaded to: {local_dir}")
    except Exception as e:
        print(f"Download failed with error: {e}")


if __name__ == "__main__":
    download_model()
