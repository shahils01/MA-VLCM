import os
import torch
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


def download_model():
    model_name = "llava-hf/LLaVA-NeXT-Video-7B-32K-hf"
    print(f"Starting download for {model_name}...")
    print(f"HF_HOME is set to: {os.environ.get('HF_HOME', 'Not Set')}")

    print("Downloading Processor...")
    processor = LlavaNextVideoProcessor.from_pretrained(model_name)
    print(f"Processor downloaded successfully: {type(processor)}")

    print("Downloading Model (this may take a while)...")
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    print(f"Model downloaded successfully: {type(model)}")


if __name__ == "__main__":
    download_model()
