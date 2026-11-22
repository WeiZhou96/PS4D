"""SAM 2.1, CoTracker3 and MoGe-2, called exactly as PS4D uses them.

SAM 2.1     facebook/sam2.1-hiera-large (Hugging Face), video predictor, bfloat16 autocast
CoTracker3  scaled_offline.pth, offline predictor with window_len 60
MoGe-2      Ruicheng/moge-2-vitl-normal (Hugging Face), fov_x from the intrinsics, fp32
"""
import os
import shutil
import tempfile

import numpy as np

from ..utils import checkpoint_dir

SAM2_MODEL = "facebook/sam2.1-hiera-large"
MOGE_MODEL = "Ruicheng/moge-2-vitl-normal"


def load_cotracker(checkpoint=None, device="cuda"):
    from cotracker.predictor import CoTrackerPredictor
    checkpoint = checkpoint or os.path.join(checkpoint_dir(), "cotracker3", "scaled_offline.pth")
    return CoTrackerPredictor(checkpoint=checkpoint, offline=True, window_len=60).to(device)


def cotrack(model, frames, queries, device="cuda"):
    """frames (T, H, W, 3) uint8; queries (Q, 3) of (t, x, y).  Returns tracks (T, Q, 2) and visibility (T, Q)."""
    import torch
    vid = torch.tensor(frames, device=device).permute(0, 3, 1, 2)[None].float()
    with torch.no_grad():
        tr, vis = model(vid, queries=torch.tensor(np.asarray(queries), dtype=torch.float32, device=device)[None])
    return tr[0].cpu().numpy(), vis[0].cpu().numpy()


def load_sam2_video(device="cuda"):
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    return SAM2VideoPredictor.from_pretrained(SAM2_MODEL, device=device)


def load_sam2_image(device="cuda"):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor.from_pretrained(SAM2_MODEL, device=device)


def sam2_propagate(predictor, frames, start, box, workdir=None):
    """Masks (T, H, W) bool propagated forward from a box (x0, y0, x1, y1) on frame `start`; frames before it are
    empty.  The frames are written as JPEG (quality 95) for the video predictor, as in the experiments."""
    import cv2
    import torch
    d = tempfile.mkdtemp(prefix="ps4d_sam2_", dir=workdir)
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(d, f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        M = np.zeros(frames.shape[:3], bool)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = predictor.init_state(video_path=d)
            predictor.add_new_points_or_box(st, frame_idx=int(start), obj_id=1, box=np.asarray(box, np.float32))
            for f, _, logits in predictor.propagate_in_video(st):
                M[f] = (logits[0, 0] > 0).float().cpu().numpy() > 0
            predictor.reset_state(st)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return M


def load_moge(device="cuda"):
    from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(MOGE_MODEL).to(device).eval()


def moge_depths(model, frames_rgb, K, device="cuda"):
    """{frame: (H, W) metric depth, NaN where MoGe-2 masks the pixel} for a dict {frame: RGB image}."""
    import torch
    out = {}
    for k, img in frames_rgb.items():
        fov_x = float(np.degrees(2 * np.arctan(img.shape[1] / (2 * K[0, 0]))))
        with torch.no_grad():
            o = model.infer(torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1),
                            fov_x=fov_x, use_fp16=False)
        d = o["depth"].float().cpu().numpy()
        d[~o["mask"].cpu().numpy().astype(bool)] = np.nan
        out[k] = d
    return out


def stored_mono_depth(depths, frames_shape, frames=None):
    """Monocular depth in the stored form used by every experiment: (T, H, W) float16 with 0 where invalid
    (frames not in `depths` are 0)."""
    T, H, W = frames_shape[:3]
    out = np.zeros((T, H, W), np.float16)
    for k, d in depths.items():
        out[k] = np.nan_to_num(d, nan=0.0).astype(np.float16)
    return out
