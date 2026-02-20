import torch
from torch.utils.data import Dataset


class RandomMultimodalDataset(Dataset):
    """Placeholder dataset. Replace with your real data loader."""

    def __init__(
        self,
        num_samples: int,
        video_frames: int,
        video_channels: int,
        video_height: int,
        video_width: int,
        num_robots: int,
        robot_obs_dim: int,
        text_dim: int,
    ):
        self.num_samples = num_samples
        self.video_frames = video_frames
        self.video_channels = video_channels
        self.video_height = video_height
        self.video_width = video_width
        self.num_robots = num_robots
        self.robot_obs_dim = robot_obs_dim
        self.text_dim = text_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video = torch.randn(
            self.video_frames,
            self.video_channels,
            self.video_height,
            self.video_width,
        )
        robot_obs = torch.randn(self.video_frames, self.num_robots, self.robot_obs_dim)
        adj = torch.randint(0, 2, (self.video_frames, self.num_robots, self.num_robots)).float()
        text_emb = torch.randn(self.text_dim)
        value = torch.randn(1).squeeze(0)
        return {
            "video": video,
            "robot_obs": robot_obs,
            "adj": adj,
            "text_emb": text_emb,
            "value": value,
        }
