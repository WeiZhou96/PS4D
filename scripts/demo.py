"""PS4D on your own video: depth motion of a manipulated object along the camera's line of sight.

Input: a video file or a directory of frames, the camera intrinsics and a box around the object on the first frame.
Output (in --out): depth_motion.csv (per frame: log depth of the projective-scale branch, the 4D branch and PS4D,
centred, i.e. up to an additive constant; exp of a difference is a depth ratio), depth_motion.png (plot) and
overlay.mp4 (mask outline and the PS4D curve).

Example:
  python scripts/demo.py --video clip.mp4 --intrinsics 600 600 320 240 --box 210 150 330 260 --out demo_out \
      --depth4d-repo third_party/Pi3 --depth4d-checkpoint checkpoints/pi3x/model.safetensors
Use --depth4d none for the projective-scale branch alone (no 4D model).  Intrinsics refer to the original
resolution; frames are resized so that the long side is at most --max-side and the intrinsics are scaled with them.
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ps4d.utils import checkpoint_dir, default_reliability_model, third_party_dir  # noqa: E402


def read_frames(a):
    if a.video:
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        out = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        files = sorted(p for p in glob.glob(os.path.join(a.frames, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png")))
        out = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in files]
        fps = a.fps
    if not out:
        sys.exit("no frames read")
    frames = np.stack(out[a.start:a.start + a.max_frames if a.max_frames else None])
    return frames, fps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="video file")
    src.add_argument("--frames", help="directory of frames (sorted by name)")
    ap.add_argument("--intrinsics", nargs=4, type=float, required=True, metavar=("FX", "FY", "CX", "CY"))
    ap.add_argument("--box", nargs=4, type=float, required=True, metavar=("X0", "Y0", "X1", "Y1"),
                    help="box around the object on the first used frame (original pixels)")
    ap.add_argument("--start", type=int, default=0, help="first frame to use")
    ap.add_argument("--max-frames", type=int, default=0, help="number of frames to use (0 = all)")
    ap.add_argument("--fps", type=float, default=30.0, help="frame rate of a frame directory (for the overlay)")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--depth4d", default="pi3x", choices=("pi3x", "pi3", "vggt", "cut3r", "none"))
    ap.add_argument("--depth4d-repo", default=os.path.join(third_party_dir(), "Pi3"),
                    help="code of the 4D model (the Pi3 repository for pi3x and pi3)")
    ap.add_argument("--depth4d-checkpoint", default=os.path.join(checkpoint_dir(), "pi3x", "model.safetensors"),
                    help="checkpoint of the 4D model")
    ap.add_argument("--cotracker-checkpoint", default=None)
    ap.add_argument("--model", default=default_reliability_model(), help="reliability model")
    ap.add_argument("--stride-4d", type=int, default=1, help="run the 4D model on every n-th frame")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="demo_out")
    a = ap.parse_args()

    from ps4d.pipeline import Models, estimate
    from ps4d.reliability import VarianceModel
    frames, fps = read_frames(a)
    H, W = frames.shape[1:3]
    s = min(1.0, a.max_side / max(H, W))
    fx, fy, cx, cy = a.intrinsics
    box = np.array(a.box, np.float32)
    if s < 1.0:
        W1, H1 = int(round(W * s)), int(round(H * s))
        frames = np.stack([cv2.resize(f, (W1, H1), interpolation=cv2.INTER_AREA) for f in frames])
        sx, sy = W1 / W, H1 / H
        fx, fy, cx, cy = fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5
        box = ((box.reshape(2, 2) + 0.5) * [sx, sy] - 0.5).ravel().astype(np.float32)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    models = Models(VarianceModel.load(a.model), None if a.depth4d == "none" else a.depth4d, a.depth4d_repo,
                    a.depth4d_checkpoint, a.cotracker_checkpoint, a.device)
    T = len(frames)
    res = estimate(models, frames, K, box, start=0, frames_4d=np.arange(0, T, a.stride_4d))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "depth_motion.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "log_depth_ps4d", "log_depth_projective_scale", "log_depth_4d"])
        for t in range(T):
            w.writerow([t + a.start] + [None if not np.isfinite(res[k][t]) else f"{res[k][t]:.6f}"
                                        for k in ("log_depth", "log_depth_ps", "log_depth_4d")])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(T) + a.start
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=150)
    ax.plot(t, res["log_depth_ps"], color="#4D9A96", lw=1.0, label="projective-scale branch")
    if a.depth4d != "none":
        ax.plot(t, res["log_depth_4d"], color="#9A9EA6", lw=1.0, label=f"4D branch ({a.depth4d})")
        ax.plot(t, res["log_depth"], color="#A3203B", lw=1.8, label="PS4D")
    ax.set_xlabel("frame")
    ax.set_ylabel("log depth (centred)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "depth_motion.png"))
    plt.close(fig)
    curve = res["log_depth"] if a.depth4d != "none" else res["log_depth_ps"]
    ok = np.isfinite(curve)
    lo, hi = (np.nanmin(curve), np.nanmax(curve)) if ok.any() else (-0.1, 0.1)
    span = max(hi - lo, 1e-3)
    h, w = frames.shape[1:3]
    ph = max(80, h // 4)
    vw = cv2.VideoWriter(os.path.join(a.out, "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + ph))
    for k in range(T):
        img = cv2.cvtColor(frames[k], cv2.COLOR_RGB2BGR)
        cnts, _ = cv2.findContours(res["masks"][k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, (59, 32, 163), 2)
        panel = np.full((ph, w, 3), 255, np.uint8)
        pts = [(int(i / max(T - 1, 1) * (w - 1)), int((1 - (curve[i] - lo) / span) * (ph - 11)) + 5)
               for i in range(T) if np.isfinite(curve[i])]
        for p, q in zip(pts[:-1], pts[1:]):
            cv2.line(panel, p, q, (59, 32, 163), 1, cv2.LINE_AA)
        cv2.line(panel, (int(k / max(T - 1, 1) * (w - 1)), 0), (int(k / max(T - 1, 1) * (w - 1)), ph - 1), (60, 60, 60), 1)
        cv2.putText(panel, "log depth (PS4D)", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)
        vw.write(np.vstack([img, panel]))
    vw.release()
    print(f"{T} frames; {len(res['measurements'])} measurements; outputs in {a.out}")


if __name__ == "__main__":
    main()
