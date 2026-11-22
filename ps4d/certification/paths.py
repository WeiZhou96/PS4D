"""Locations of the DROID data and assets, under $PS4D_DROID_ROOT (default ./data/droid):

    datasets/droid_raw/<uuid>/                 DROID raw release 1.0.1 episodes (metadata, trajectory.h5, MP4s)
    external/droid_vp/tables/                  DROID-VP tables (camera intrinsics by serial, episode ids; CC BY 4.0)
    external/droid_vp/points/droidvp_gripper_model.npz   gripper points in the flange frame
                                               (a copy is shipped in assets/ and used when this file is absent)
    external/droid3d/                          DROID-3D calibration release (extrinsics.jsonl, camera_intrinsics.jsonl)
    external/depthworld/gripper2wrist_transforms.json    DepthWorld wrist asset (MIT)
    external/menagerie/franka_emika_panda, franka_fr3     MuJoCo Menagerie robot models (Apache-2.0)
"""
import os

DATA_ROOT = os.environ.get("PS4D_DROID_ROOT", os.path.join(os.getcwd(), "data", "droid"))
RAW_DIR = os.path.join(DATA_ROOT, "datasets/droid_raw")
DROIDVP = os.path.join(DATA_ROOT, "external/droid_vp/tables")
D3D_DIR = os.path.join(DATA_ROOT, "external/droid3d")
G2W_ASSET = os.path.join(DATA_ROOT, "external/depthworld/gripper2wrist_transforms.json")
MENAGERIE = os.path.join(DATA_ROOT, "external/menagerie")
GRIPPER_MODEL = os.path.join(DATA_ROOT, "external/droid_vp/points/droidvp_gripper_model.npz")
if not os.path.exists(GRIPPER_MODEL):      # the copy shipped with the repository
    GRIPPER_MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "assets", "droidvp_gripper_model.npz")
