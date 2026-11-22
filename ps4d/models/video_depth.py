"""Metric Video Depth Anything (VDA-L) on a clip: the official model and inference call (input size 518, fp16),
depth resized to the clip resolution and stored as float16."""
import os
import sys

import numpy as np


def load_vda(repo, checkpoint, encoder="vitl", device="cuda"):
    import torch
    sys.path.insert(0, os.path.abspath(repo))
    from video_depth_anything.video_depth import VideoDepthAnything
    cfg = {"vits": (64, [48, 96, 192, 384]), "vitb": (128, [96, 192, 384, 768]), "vitl": (256, [256, 512, 1024, 1024])}
    features, channels = cfg[encoder]
    model = VideoDepthAnything(encoder=encoder, features=features, out_channels=channels, metric=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval()


def vda_depth(model, frames, fps=15, device="cuda"):
    """(T, H, W) float16 metric depth of every frame."""
    import cv2
    T, H, W = frames.shape[:3]
    depths, _ = model.infer_video_depth(frames, fps, input_size=518, device=device, fp32=False)
    depths = np.stack([cv2.resize(np.asarray(dd, np.float32), (W, H), interpolation=cv2.INTER_LINEAR) for dd in depths])
    return depths.astype(np.float16)
