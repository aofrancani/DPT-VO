import numpy as np
import cv2
from util.gric import *
from scale_recovery import find_scale_from_depth


class VisualOdometry:
    """
    Monocular Visual Odometry:
        1) Capture new frame I_k
        2) Extract and match features between I_{k-1} and I_k
        3) Compute essential matrix for image pair I_{k-1}, I_k
        4) Decompose essential matrix into R_k and t_k, and form T_k
        5) Compute relative scale and rescale tk accordingly
        6) Concatenate transformation by computing Ck ¼ Ck1Tk
        7) Repeat from 1).

    Main theory source:
        D. Scaramuzza and F. Fraundorfer, "Visual Odometry [Tutorial]"
        https://rpg.ifi.uzh.ch/visual_odometry_tutorial.html

    Code ref.:
        https://github.com/uoip/monoVO-python
    """

    def __init__(self, cam, depth_model):
        self.cam = cam  # camera object
        self.prev_frame = None  # previous frame
        self.feat_ref = None  # reference features (first frame)
        self.feat_curr = None  # features from current image
        self.detector_method = "FAST"  # detector method
        self.matching_method = "OF_PyrLK"  # feature matching method
        self.min_num_features = 2500
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.depth_model = depth_model
        self.prev_depth = None
        self.pose = np.zeros((4, 4))  # pose matrix [R | t; 0  1]

    def detect_features(self, frame):
        """
        Point-feature detector: search for salient keypoints that are likely to match well in other image frames.
            - corner detectors: Moravec, Forstner, Harris, Shi-Tomasi, and FAST.
            - blob detectors: SIFT, SURF, and CENSURE.

        Args:
            frame {ndarray}: frame to be processed
        """
        if self.detector_method == "FAST":
            detector = cv2.FastFeatureDetector_create()  # threshold=25, nonmaxSuppression=True)
            return detector.detect(frame)

        elif self.detector_method == "ORB":
            detector = cv2.ORB_create(nfeatures=2000)
            kp1, des1 = detector.detectAndCompute(frame, None)
            return kp1

    def feature_matching(self, frame):
        """
        The feature-matching: looks for corresponding features in other images.

        Args:
            frame {ndarray}: frame to be processed
        """
        if self.matching_method == "OF_PyrLK":
            # Calculate optical flow for a sparse feature set using the iterative Lucas-Kanade method with pyramids
            kp2, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame,
                                                    self.feat_ref, None,
                                                    winSize=(21, 21),
                                                    criteria=(
                                                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

            st = st.reshape(st.shape[0])  # status of points from frame to frame
            # Keypoints
            kp1 = self.feat_ref[st == 1]
            kp2 = kp2[st == 1]
        return kp1, kp2

    def motion_estimation_init(self, frame):
        """
        Processes first frame to initialize the reference features and the matrix R and t.
        Only for frame_id == 0.

        Args:
            frame {ndarray}: frame to be processed
            frame_id {int}: integer corresponding to the frame id
        """
        feat_ref = self.detect_features(frame)
        self.feat_ref = np.array([x.pt for x in feat_ref], dtype=np.float32)

    def motion_estimation(self, frame, frame_id):
        """
        Estimates the motion from current frame and computed keypoints

        Args:
            frame {ndarray}: frame to be processed
            frame_id {int}: integer corresponding to the frame id
            pose {list}: list with ground truth pose [x, y, z]
        """
        self.feat_ref, self.feat_curr = self.feature_matching(frame)
        # # Estimating an essential matrix (E): geometric relations between two images
        R, t = compute_pose_2d2d(self.feat_ref, self.feat_curr, self.cam)

        # estimate depth from a single image frame
        depth = self.depth_model.compute_depth(frame)
        depth = preprocess_depth(depth, crop=[[0.3, 1], [0, 1]], depth_range=[0, 50])

        if frame_id == 1:
            self.R = R
            self.t = t

        else:
            E_pose = np.eye(4)
            E_pose[:3, :3] = R
            E_pose[: 3, 3:] = t

            # estimate scale
            scale = find_scale_from_depth(
                self.cam,
                self.feat_ref,
                self.feat_curr,
                np.linalg.inv(E_pose),
                depth
            )

            if np.linalg.norm(t) == 0 or scale == -1.0:
                R, t = compute_pose_3d2d(
                    self.feat_ref,
                    self.feat_curr,
                    self.prev_depth,
                    self.cam
                )  # pose: from cur->ref
                scale = 1.0

            # estimate camera motion
            self.t = self.t + scale * self.R.dot(t)
            self.R = self.R.dot(R)

        self.prev_depth = depth

        # check if number of features is enough (some features are lost in time due to the moving scene)
        if self.feat_ref.shape[0] < self.min_num_features:
            self.feat_curr = self.detect_features(frame)
            self.feat_curr = np.array([x.pt for x in self.feat_curr], dtype=np.float32)

        # update reference features
        self.feat_ref = self.feat_curr

    def update(self, frame, frame_id):
        """
        Computes the camera motion between the current image and the previous one.
        """
        # Process first frame to get reference features
        if frame_id == 0:
            self.motion_estimation_init(frame)
        else:
            self.motion_estimation(frame, frame_id)

        self.prev_frame = frame

        # Pose matrix
        pose = np.eye(4)
        pose[:3, :3] = self.R
        pose[: 3, 3:] = self.t
        self.pose = pose


def preprocess_depth(depth, crop, depth_range):
    """
    Preprocess depth map with cropping and capping range
    Code adapted from DF-VO:
        -https://github.com/Huangying-Zhan/DF-VO

    Args:
        depth (array, [HxW]): depth map
        crop (list): normalized crop regions [[y0, y1], [x0, x1]]. non-cropped regions set to 0.
        depth_range (list): a list with float numbers [min_depth, max_depth]

    Returns:
        depth (array, [HxW]): processed depth map
    """
    # normalize depth
    # depth_min = depth.min()
    # depth_max = depth.max()
    # max_val = (2 ** (8 * 1)) - 1
    # if depth_max - depth_min > np.finfo("float").eps:
    #     depth = max_val * (depth - depth_min) / (depth_max - depth_min)
    # else:
    #     depth = np.zeros(depth.shape, dtype=depth.dtype)

    # print("depth_max: ", depth.max())
    # print("depth_min: ", depth.min())

    # set cropping region
    min_depth, max_depth = depth_range
    h, w = depth.shape
    y0, y1 = int(h * crop[0][0]), int(h * crop[0][1])
    x0, x1 = int(w * crop[1][0]), int(w * crop[1][1])
    depth_mask = np.zeros((h, w))
    depth_mask[y0:y1, x0:x1] = 1

    # set range mask
    depth_range_mask = (depth < max_depth) * (depth > min_depth)

    # set invalid pixel to zero depth
    valid_mask = depth_mask * depth_range_mask
    depth = depth * valid_mask
    return depth


def compute_pose_3d2d(kp1, kp2, depth_1, cam_intrinsics):
    """
    Compute pose from 3d-2d correspondences
    Code adapted from DF-VO:
        -https://github.com/Huangying-Zhan/DF-VO

    Args:
        cam_intrinsics: camera intrinsics
        kp1 (array, [Nx2]): keypoints for view-1
        kp2 (array, [Nx2]): keypoints for view-2
        depth_1 (array, [HxW]): depths for view-1

    Returns:
        R (array, [3x3]): rotation matrix
        t (array, [3x1]): translation vector
    """
    max_depth = 50
    min_depth = 0

    outputs = {}
    height, width = depth_1.shape

    # Filter keypoints outside image region
    x_idx = (kp1[:, 0] >= 0) * (kp1[:, 0] < width)
    kp1 = kp1[x_idx]
    kp2 = kp2[x_idx]
    x_idx = (kp2[:, 0] >= 0) * (kp2[:, 0] < width)
    kp1 = kp1[x_idx]
    kp2 = kp2[x_idx]
    y_idx = (kp1[:, 1] >= 0) * (kp1[:, 1] < height)
    kp1 = kp1[y_idx]
    kp2 = kp2[y_idx]
    y_idx = (kp2[:, 1] >= 0) * (kp2[:, 1] < height)
    kp1 = kp1[y_idx]
    kp2 = kp2[y_idx]

    # Filter keypoints outside depth range
    kp1_int = kp1.astype(np.int)
    kp_depths = depth_1[kp1_int[:, 1], kp1_int[:, 0]]
    non_zero_mask = (kp_depths != 0)
    depth_range_mask = (kp_depths < max_depth) * (kp_depths > min_depth)
    valid_kp_mask = non_zero_mask * depth_range_mask

    kp1 = kp1[valid_kp_mask]
    kp2 = kp2[valid_kp_mask]

    # Get 3D coordinates for kp1
    XYZ_kp1 = unprojection_kp(kp1, kp_depths[valid_kp_mask], cam_intrinsics)

    # initialize ransac setup
    best_rt = []
    best_inlier = 0
    max_ransac_iter = 3

    for _ in range(max_ransac_iter):
        # shuffle kp (only useful when random seed is fixed)
        new_list = np.arange(0, kp2.shape[0], 1)
        np.random.shuffle(new_list)
        new_XYZ = XYZ_kp1.copy()[new_list]
        new_kp2 = kp2.copy()[new_list]

        if new_kp2.shape[0] > 4:
            # PnP solver
            flag, r, t, inlier = cv2.solvePnPRansac(
                objectPoints=new_XYZ,
                imagePoints=new_kp2,
                cameraMatrix=cam_intrinsics.mat,
                distCoeffs=None,
                iterationsCount=100,  # number of iteration
                reprojectionError=1,  # inlier threshold value
            )

            # save best pose estimation
            if flag and inlier.shape[0] > best_inlier:
                best_rt = [r, t]
                best_inlier = inlier.shape[0]

    # format pose
    R = np.eye(3)
    t = np.zeros((3, 1))
    if len(best_rt) != 0:
        r, t = best_rt
        R = cv2.Rodrigues(r)[0]
    E_pose = np.eye(4)
    E_pose[:3, :3] = R
    E_pose[: 3, 3:] = t
    E_pose = np.linalg.inv(E_pose)
    R = E_pose[:3, :3]
    t = E_pose[: 3, 3:]
    return R, t


def unprojection_kp(kp, kp_depth, cam_intrinsics):
    """
    Convert kp to XYZ
    Code from DF-VO:
        -https://github.com/Huangying-Zhan/DF-VO

    Args:
        kp (array, [Nx2]): [x, y] keypoints
        kp_depth (array, [Nx2]): keypoint depth
        cam_intrinsics (Intrinsics): camera intrinsics

    Returns:
        XYZ (array, [Nx3]): 3D coordinates
    """
    N = kp.shape[0]
    # initialize regular grid
    XYZ = np.ones((N, 3, 1))
    XYZ[:, :2, 0] = kp

    inv_K = np.ones((1, 3, 3))
    inv_K[0] = np.linalg.inv(cam_intrinsics.mat)  # cam_intrinsics.inv_mat
    inv_K = np.repeat(inv_K, N, axis=0)

    XYZ = np.matmul(inv_K, XYZ)[:, :, 0]
    XYZ[:, 0] = XYZ[:, 0] * kp_depth
    XYZ[:, 1] = XYZ[:, 1] * kp_depth
    XYZ[:, 2] = XYZ[:, 2] * kp_depth
    return XYZ


def compute_pose_2d2d(kp_ref, kp_cur, cam_intrinsics):
    """
    Compute the pose from view2 to view1
    Code adapted from DF-VO:
        -https://github.com/Huangying-Zhan/DF-VO

    Args:
        kp_ref (array, [Nx2]): keypoints for reference view
        kp_cur (array, [Nx2]): keypoints for current view
        cam_intrinsics (Intrinsics): camera intrinsics
        is_iterative (bool): is iterative stage

    Returns:
        a dictionary containing
            - **pose** (SE3): relative pose from current to reference view
            - **best_inliers** (array, [N]): boolean inlier mask
    """
    principal_points = (cam_intrinsics.cx, cam_intrinsics.cy)

    # initialize ransac setup
    R = np.eye(3)
    t = np.zeros((3, 1))
    best_Rt = [R, t]
    best_inlier_cnt = 0
    max_ransac_iter = 3
    best_inliers = np.ones((kp_ref.shape[0], 1)) == 1

    # method GRIC of validating E-tracker
    if kp_cur.shape[0] > 10:
        H, H_inliers = cv2.findHomography(
            kp_cur,
            kp_ref,
            method=cv2.RANSAC,
            confidence=0.99,
            ransacReprojThreshold=1,
        )

        H_res = compute_homography_residual(H, kp_cur, kp_ref)
        H_gric = calc_GRIC(
            res=H_res,
            sigma=0.8,
            n=kp_cur.shape[0],
            model="HMat"
        )
        valid_case = True
    else:
        valid_case = False

    if valid_case:
        num_valid_case = 0
        for i in range(max_ransac_iter):  # repeat ransac for several times for stable result
            # shuffle kp_cur and kp_ref (only useful when random seed is fixed)
            new_list = np.arange(0, kp_cur.shape[0], 1)
            np.random.shuffle(new_list)
            new_kp_cur = kp_cur.copy()[new_list]
            new_kp_ref = kp_ref.copy()[new_list]

            E, inliers = cv2.findEssentialMat(
                new_kp_cur,
                new_kp_ref,
                focal=cam_intrinsics.fx,
                pp=principal_points,
                method=cv2.RANSAC,
                prob=0.99,
                threshold=0.2,
            )

            # get F from E
            K = cam_intrinsics.mat
            F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
            E_res = compute_fundamental_residual(F, new_kp_cur, new_kp_ref)

            E_gric = calc_GRIC(
                res=E_res,
                sigma=0.8,
                n=kp_cur.shape[0],
                model='EMat'
            )
            valid_case = H_gric > E_gric

            # inlier check
            inlier_check = inliers.sum() > best_inlier_cnt

            # save best_E
            if inlier_check:
                best_E = E
                best_inlier_cnt = inliers.sum()

                revert_new_list = np.zeros_like(new_list)
                for cnt, i in enumerate(new_list):
                    revert_new_list[i] = cnt
                best_inliers = inliers[list(revert_new_list)]
            num_valid_case += (valid_case * 1)

        major_valid = num_valid_case > (max_ransac_iter / 2)
        if major_valid:
            cheirality_cnt, R, t, _ = cv2.recoverPose(best_E, kp_cur, kp_ref,
                                                      focal=cam_intrinsics.fx,
                                                      pp=principal_points,
                                                      )

            # cheirality_check
            if cheirality_cnt > kp_cur.shape[0] * 0.1:
                best_Rt = [R, t]

    R, t = best_Rt
    return R, t
