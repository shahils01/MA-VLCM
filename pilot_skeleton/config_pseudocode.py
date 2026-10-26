"""Defaults for the image-language value-model pilot."""

from dataclasses import asdict, dataclass


@dataclass
class PilotConfig:
    dataset: str = "shulnak09/formation-vlcm-polygon-3to6"
    dataset_revision: str = "main"
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    model_revision: str = "main"
    output_dir: str = "pilot_runs/iteration1"
    feature_cache: str = "pilot_runs/iteration1/features"
    clip_length: int = 8
    clip_stride: int = 2
    horizon: int = 0  # 0 means discount all rewards through episode termination
    image_hz: float = 5.0
    gamma: float = 0.99  # per 20 Hz control-step discount
    temporal_width: int = 256
    temporal_heads: int = 4
    temporal_layers: int = 2
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 0.001
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42
    num_workers: int = 0

    def to_dict(self):
        return asdict(self)
