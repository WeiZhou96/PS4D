"""Per-frame depth from 4D reconstruction models (Pi3X, pi3, VGGT, CUT3R).

Each model runs on the selected frames of a clip at once (at most 100, uniformly subsampled otherwise; the
experiments used all frames on the hand datasets and every third frame on DROID).  The depth of a frame is the z of
the frame's own-camera point map, resized back to the clip resolution (bilinear) and stored as float16; frames not run
are NaN.  The sequence-level scale of the depth does not matter for depth motion.

  pi3x / pi3  images resized to <= 255,000 px with sides multiples of 14 (as the repository's loader), bfloat16;
              local_points[..., 2].  Pi3X is built with use_multimodal=False.
  vggt        VGGT-1B; width 518, height rounded to a multiple of 14 (no crop for landscape or square inputs); depth head.
  cut3r       cut3r_512_dpt_4_64; the repository's own loader (long side 512, centre crop to multiples of 16, square
              inputs cropped to 4:3); pts3d_in_self_view[..., 2]; pixels outside the crop are NaN.
"""
import math
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

MODELS = ("pi3x", "pi3", "vggt", "cut3r")


def _resize(frames, W1, H1, method):
    from PIL import Image
    return np.stack([np.asarray(Image.fromarray(f).resize((W1, H1), method)) for f in frames])


def run_vggt(model, frames):
    import torch
    from PIL import Image
    T, H, W = frames.shape[:3]
    W1 = 518
    H1 = int(round(H * W1 / W / 14)) * 14
    x = torch.tensor(_resize(frames, W1, H1, Image.Resampling.BICUBIC), device="cuda").permute(0, 3, 1, 2).float() / 255.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x)
    return pred["depth"][0, ..., 0].float().cpu().numpy()


def run_pi3(model, frames, limit=255000):
    import torch
    from PIL import Image
    T, H, W = frames.shape[:3]
    s = math.sqrt(limit / (W * H))
    k, m = round(W * s / 14), round(H * s / 14)
    while (k * 14) * (m * 14) > limit:
        if k / m > W / H:
            k -= 1
        else:
            m -= 1
    W1, H1 = max(1, k) * 14, max(1, m) * 14
    x = _resize(frames, W1, H1, Image.Resampling.LANCZOS)
    x = torch.tensor(x, device="cuda").permute(0, 3, 1, 2).float()[None] / 255.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        res = model(imgs=x)
    return res["local_points"][0, ..., 2].float().cpu().numpy()


def run_cut3r(model, frames, inference, load_images):
    import torch
    from PIL import Image
    T, H, W = frames.shape[:3]
    tmp = tempfile.mkdtemp(prefix="ps4d_cut3r_")
    try:
        paths = []
        for i, f in enumerate(frames):
            p = os.path.join(tmp, f"{i:05d}.png")
            Image.fromarray(f).save(p)
            paths.append(p)
        imgs = load_images(paths, size=512, verbose=False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    views = []
    for i, im in enumerate(imgs):
        views.append({"img": im["img"],
                      "ray_map": torch.full((im["img"].shape[0], 6, im["img"].shape[-2], im["img"].shape[-1]), torch.nan),
                      "true_shape": torch.from_numpy(im["true_shape"]), "idx": i, "instance": str(i),
                      "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                      "img_mask": torch.tensor(True).unsqueeze(0), "ray_mask": torch.tensor(False).unsqueeze(0),
                      "update": torch.tensor(True).unsqueeze(0), "reset": torch.tensor(False).unsqueeze(0)})
    with torch.no_grad():
        out, _ = inference(views, model, "cuda")
    D = np.stack([o["pts3d_in_self_view"][0, ..., 2].float().cpu().numpy() for o in out["pred"]])
    # undo the loader: long side resized to 512 (as PIL in the repository), then the centred crop
    s = 512 / max(W, H)
    W1, H1 = int(round(W * s)), int(round(H * s))
    cx, cy = W1 // 2, H1 // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if W1 == H1:
        halfh = int(3 * halfw / 4)
    assert D.shape[1:] == (2 * halfh, 2 * halfw), (D.shape, halfh, halfw)
    canvas = np.full((T, H1, W1), np.nan, np.float32)
    canvas[:, cy - halfh:cy + halfh, cx - halfw:cx + halfw] = D
    return canvas


def load(name, repo, checkpoint, device="cuda"):
    """Returns a function frames (T, H, W, 3) uint8 -> depth (T, h, w) at the model's resolution."""
    import torch
    sys.path.insert(0, os.path.abspath(repo))
    if name == "vggt":
        from vggt.models.vggt import VGGT
        model = VGGT()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        fn = run_vggt
    elif name in ("pi3", "pi3x"):
        from safetensors.torch import load_file
        if name == "pi3":
            from pi3.models.pi3 import Pi3
            model = Pi3()
        else:
            from pi3.models.pi3x import Pi3X
            model = Pi3X(use_multimodal=False)
        model.load_state_dict(load_file(checkpoint), strict=False)
        fn = run_pi3
    elif name == "cut3r":
        sys.path.insert(0, os.path.join(os.path.abspath(repo), "src"))
        from src.dust3r.inference import inference
        from src.dust3r.model import ARCroco3DStereo
        from src.dust3r.utils.image import load_images
        model = ARCroco3DStereo.from_pretrained(checkpoint)
        fn = lambda m, f: run_cut3r(m, f, inference, load_images)  # noqa: E731
    else:
        raise ValueError(name)
    model = model.to(device).eval()
    return lambda frames: fn(model, frames)


def select_frames(frame_ids, max_frames=100):
    """At most max_frames of frame_ids, uniformly subsampled."""
    frame_ids = np.asarray(frame_ids)
    if len(frame_ids) <= max_frames:
        return frame_ids
    return frame_ids[np.round(np.linspace(0, len(frame_ids) - 1, max_frames)).astype(int)]


def clip_depth(run, frames, frame_ids, max_frames=100):
    """(T, H, W) float16 depth at the clip resolution on the selected frames, NaN elsewhere."""
    T, H, W = frames.shape[:3]
    sel = select_frames(frame_ids, max_frames)
    Dm = run(frames[sel])
    depth = np.full((T, H, W), np.nan, np.float32)
    for i, t in enumerate(sel):
        depth[t] = cv2.resize(Dm[i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return depth.astype(np.float16), sel
