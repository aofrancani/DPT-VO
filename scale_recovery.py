"""
Code adapted and modified from DF-VO:
    -https://github.com/Huangying-Zhan/DF-VO
"""

import numpy as np
from sklearn import linear_model
import cv2


def image_shape(img):
    """Return image shape

    Args:
        img (array, [HxWx(c) or HxW]): image

    Returns:
        a tuple containing
            - **h** (int) : image height
            - **w** (int) : image width
            - **c** (int) : image channel
    """
    if len(img.shape) == 3:
        return img.shape
    elif len(img.shape) == 2:
        h, w = img.shape
        return h, w, 1


def find_scale_from_depth(cam_intrinsics, kp1, kp2, T_21, depth2, ransac_method="depth_ratio"):
    """Compute VO scaling factor for T_21

    Args:
        kp1 (array, [Nx2]): reference kp
        kp2 (array, [Nx2]): current kp
        T_21 (array, [4x4]): relative pose; from view 1 to view 2
        depth2 (array, [HxW]): depth 2

    Returns:
        scale (float): scaling factor
    """
    # Triangulation
    img_h, img_w, _ = image_shape(depth2)
    kp1_norm = kp1.copy()
    kp2_norm = kp2.copy()

    kp1_norm[:, 0] = \
        (kp1[:, 0] - cam_intrinsics.cx) / cam_intrinsics.fx
    kp1_norm[:, 1] = \
        (kp1[:, 1] - cam_intrinsics.cy) / cam_intrinsics.fy
    kp2_norm[:, 0] = \
        (kp2[:, 0] - cam_intrinsics.cx) / cam_intrinsics.fx
    kp2_norm[:, 1] = \
        (kp2[:, 1] - cam_intrinsics.cy) / cam_intrinsics.fy

    # triangulation
    _, _, X2_tri = triangulation(kp1_norm, kp2_norm, np.eye(4), T_21)

    # Triangulation outlier removal
    depth2_tri = convert_sparse3D_to_depth(kp2, X2_tri, img_h, img_w)
    depth2_tri[depth2_tri < 0] = 0
    # self.timers.end('triangulation')

    # common mask filtering
    non_zero_mask_pred2 = (depth2 > 0)
    non_zero_mask_tri2 = (depth2_tri > 0)
    valid_mask2 = non_zero_mask_pred2 * non_zero_mask_tri2

    depth_pred_non_zero = np.concatenate([depth2[valid_mask2]])
    depth_tri_non_zero = np.concatenate([depth2_tri[valid_mask2]])
    depth_ratio = depth_tri_non_zero / depth_pred_non_zero


    # Estimate scale (ransac)
    if valid_mask2.sum() > 10:
        # RANSAC scaling solver
        # self.timers.start('scale ransac', 'scale_recovery')
        ransac = linear_model.RANSACRegressor(
            base_estimator=linear_model.LinearRegression(
                fit_intercept=False),
            min_samples=3,  # minimum number of min_samples
            max_trials=100, # maximum number of trials
            stop_probability=0.99, # the probability that the algorithm produces a useful result
            residual_threshold=0.1,  # inlier threshold value
        )
        if ransac_method == "depth_ratio":
            ransac.fit(
                depth_ratio.reshape(-1, 1),
                np.ones((depth_ratio.shape[0], 1))
            )
        elif ransac_method == "abs_diff":
            ransac.fit(
                depth_tri_non_zero.reshape(-1, 1),
                depth_pred_non_zero.reshape(-1, 1),
            )
        scale = ransac.estimator_.coef_[0, 0]

    else:
        scale = -1.0
    return scale


def triangulation(kp1, kp2, T_1w, T_2w):
    """Triangulation to get 3D points

    Args:
        kp1 (array, [Nx2]): keypoint in view 1 (normalized)
        kp2 (array, [Nx2]): keypoints in view 2 (normalized)
        T_1w (array, [4x4]): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (array, [4x4]): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)

    Returns:
        a tuple containing
            - **X** (array, [3xN]): 3D coordinates of the keypoints w.r.t world coordinate
            - **X1** (array, [3xN]): 3D coordinates of the keypoints w.r.t view1 coordinate
            - **X2** (array, [3xN]): 3D coordinates of the keypoints w.r.t view2 coordinate
    """
    kp1_3D = np.ones((3, kp1.shape[0]))
    kp2_3D = np.ones((3, kp2.shape[0]))
    kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
    kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()
    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    X /= X[3]
    X1 = T_1w[:3] @ X
    X2 = T_2w[:3] @ X
    return X[:3], X1, X2


def convert_sparse3D_to_depth(kp, XYZ, height, width):
    """Convert sparse 3D keypoint to depth map

    Args:
        kp (array, [Nx2]): keypoints
        XYZ (array, [3xN]): 3D coorindates for the keypoints
        height (int): image height
        width (int): image width

    Returns:
        depth (array, [HxW]): depth map
    """
    # initialize depth map
    depth = np.zeros((height, width))
    kp_int = kp.astype(np.int)

    # remove out of region keypoints
    y_idx = (kp_int[:, 0] >= 0) * (kp_int[:, 0] < width)
    kp_int = kp_int[y_idx]
    x_idx = (kp_int[:, 1] >= 0) * (kp_int[:, 1] < height)
    kp_int = kp_int[x_idx]

    XYZ = XYZ[:, y_idx]
    XYZ = XYZ[:, x_idx]

    depth[kp_int[:, 1], kp_int[:, 0]] = XYZ[2]
    return depth
