"""PS4D: projective scale complements 4D reconstruction of manipulated objects.

The package estimates the depth motion of a manipulated object (its log depth along the camera's line of sight, up to
an additive constant) from a monocular video, the camera intrinsics and a box around the object in the first frame.

Modules
    masks         SAM 2 mask propagation from a box
    tracks        re-anchored point tracking on zoomed crops (CoTracker3)
    measurements  size and anchored-rigid measurements of log-depth changes, and their geometric statistics
    reliability   learned measurement variances (gradient-boosted trees on eleven geometric statistics)
    graph         robust weighted least squares on the log-depth graph
    fusion        the 4D branch (median depth of a 4D model inside the masks) and the equal-weight average
    metrics       residual ratio, absolute error, amplitude and paired bootstrap comparisons
    evaluation    per-clip scoring against reference log depth
    pipeline      end-to-end estimation on a video
    certification robot-kinematics certified object depth labels on DROID (training labels of the reliability model)
"""

__version__ = "1.0.0"
