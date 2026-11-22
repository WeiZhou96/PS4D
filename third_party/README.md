# Third-party code and checkpoints

PS4D calls these models through thin wrappers in `ps4d/models/`; nothing here is redistributed. The versions below
are the ones used for every result of the paper. Where a code commit was not recorded, the field says so.

| Component | Code | Weights | sha256 of the checkpoint |
|---|---|---|---|
| SAM 2.1 Hiera-L | facebookresearch/sam2 (commit not recorded) | HF `facebook/sam2.1-hiera-large`, revision `665f8e2ad61cf5f53d65644ff27c8ee525124610` | (Hub) |
| CoTracker3 | facebookresearch/co-tracker `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` | `scaled_offline.pth` | `2670d4562ed69326dda775a26e54883925cd11b6fc9b24cb7aa9f8078bce7834` |
| MoGe-2 ViT-L (normal) | microsoft/MoGe (commit not recorded) | HF `Ruicheng/moge-2-vitl-normal`, revision `cb0e8bbd6b1e243589717c78e750b1ba4c093acf` | (Hub) |
| Pi3X | yyfz/Pi3 `9fa3ddb` (short hash); `Pi3X(use_multimodal=False)` | `pi3x/model.safetensors` | `69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a` |
| pi3 | same repository | `pi3/model.safetensors` | `33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d` |
| VGGT-1B | facebookresearch/vggt (commit not recorded) | `model.pt` | `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0` |
| CUT3R | CUT3R (commit not recorded) | `cut3r_512_dpt_4_64.pth` | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` |
| Video Depth Anything, metric Large | DepthAnything/Video-Depth-Anything (commit not recorded) | `metric_video_depth_anything_vitl.pth` | `24eba25342e7ee0f054be25da6a852ebd7cfa40cd6bf41a353ecb6abfe24e620` |
| SpatialTrackerV2 | henry123-boy/SpaTrackerV2 `7e12274c52077860cebfe007a6290777db43b63c` + `patches/spatrackerv2_keep_queries.patch` | HF `Yuxihenry/SpatialTrackerV2_Front` rev `2e7164a93770cb934861d7de579bb81f58ad0b03`, `Yuxihenry/SpatialTrackerV2-Offline` rev `76e275b00f9c57dab71d46544df5255d4538106d` | (Hub) |
| Any4D, Depth Anything 3 (Large) | not recorded | not recorded | supplementary table only |

Python environments of the experiments: Python 3.11, PyTorch 2.5.1, numpy 2.0.1, scikit-learn 1.8.0, OpenCV
4.13.0 (SAM 2.1, CoTracker3, MoGe-2 and all CPU stages); PyTorch 2.4.1+cu124, numpy 1.26.4, transformers 4.46.3
(pi3, Pi3X, VGGT, CUT3R).

Apply the SpatialTrackerV2 patch inside its repository with `git apply ../patches/spatrackerv2_keep_queries.patch`.
It keeps every user query: the stock code drops low-confidence queries and pads with random duplicates, which breaks
the correspondence between queries and outputs that the baseline needs.
