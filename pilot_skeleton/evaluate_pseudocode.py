"""Write held-out predictions for the image-language value model."""

import csv

import numpy as np
import torch


def evaluate(model, episode_split, device, output_path):
    model.eval()
    predictions = []
    targets = []
    rows = []

    with torch.inference_mode():
        for clip in episode_split:
            features = clip["features"].unsqueeze(0).to(device=device, dtype=torch.float32)
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
            prediction = model(features, mask).item()
            target = float(clip["target"])
            predictions.append(prediction)
            targets.append(target)
            rows.append({
                "episode_id": clip["episode_id"],
                "frame_id": clip["frame_id"],
                "timestamp": clip["timestamp"],
                "target_return": target,
                "predicted_value": prediction,
            })

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    errors = np.asarray(predictions) - np.asarray(targets)
    return {
        "count": len(rows),
        "mse": float(np.mean(errors ** 2)),
        "mae": float(np.mean(np.abs(errors))),
        "pearson": float(np.corrcoef(predictions, targets)[0, 1]) if len(rows) > 1 else 0.0,
    }
