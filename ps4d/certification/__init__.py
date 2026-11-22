"""Certified object-depth labels from robot kinematics on DROID (Sec. 3.4 of the paper).

While a gripper holds an object, the object is rigid in the gripper frame, and a static exterior camera expressed in
that frame through the robot's forward kinematics becomes a camera moving around the object.  Object points tracked in
this camera (C_g, DROID's first exterior camera) are triangulated in the gripper frame from many virtual views;
rigidity residuals, ray span and covariance certify each point, and rendering the robot model gives its visibility.
The certified points give the kinematic label of every frame.  The second exterior camera (C_a) never enters the
labels; on the evaluation data it gives the independent reference by triangulating the same points from its own
views (cross-camera closure).

Modules
    paths          locations of the DROID raw release and the calibration / robot assets ($PS4D_DROID_ROOT)
    droid_raw      DROID raw episodes (trajectory.h5, metadata, MP4)
    calibration    DROID-3D optimized calibration (read-only) and the factory fallback
    fk             Franka Panda / FR3 forward kinematics
    geometry       SE(3) helpers and projection
    grasps         grasp segmentation from the gripper signal
    robot_render   ray-cast robot model (arm meshes, gripper hulls)
    gripper        gripper-frame helpers (between-finger box, prompts, capture-time poses)
    triangulation  gripper-frame triangulation and its certificate
    labels         the certification of one hold segment
"""
