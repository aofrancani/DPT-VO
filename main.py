import numpy as np
import cv2
import argparse
from tqdm import tqdm
from dataloader.kitti import KITTI
from camera_model import CameraModel
from depth_model import DepthModel
from visual_odometry import VisualOdometry
from traj_utils import plot_trajectory, save_trajectory
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_path",
        default=r"dataset\sequences_jpg",
        help="path to dataset"
    )
    parser.add_argument(
        "-s", "--sequence",
        default=00,
        help="sequence to be evaluated",
    )
    parser.add_argument(
        "-p",
        "--pose_path",
        default=r"dataset\poses",
        help="path to ground truth poses",
    )
    parser.add_argument(
        "-m", "--model_weights",
        default=None,
        help="path to model weights"
    )
    parser.add_argument(
        "-t", "--model_type",
        default="dpt_hybrid_kitti",
        help="model type [dpt_large|dpt_hybrid|dpt_hybrid_kitti]",
    )
    parser.add_argument(
        "-disp", "--display_traj",
        default=False,
        help="display trajectory during motion estimation if True",
    )
    parser.add_argument(
        "-seed", "--SEED",
        default=2,
        help="Random seed (int)",
    )

    parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
    parser.add_argument("--absolute_depth", dest="absolute_depth", action="store_true")
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--no-optimize", dest="optimize", action="store_false")
    parser.set_defaults(optimize=True)
    parser.set_defaults(kitti_crop=False)
    parser.set_defaults(absolute_depth=False)

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.SEED)
    torch.cuda.manual_seed(args.SEED)
    torch.manual_seed(args.SEED)

    # Create KITTI dataloader
    dataloader = KITTI(
                      data_path=args.data_path,
                      pose_path=args.pose_path,
                      sequence=args.sequence,
                      )

    # Create camera model object
    cam = CameraModel(params=dataloader.cam_params)

    # Create network model to estimate depth
    depth_model = DepthModel(model_type=args.model_type)

    # Initialize VO with camera model and depth model
    vo = VisualOdometry(cam, depth_model)

    # Initialize graph trajectory
    trajectory = 255 + np.zeros((700, 700, 3), dtype=np.uint8)

    # Initialize lists
    estimated_trajectory = []
    gt_trajectory = []
    poses = []

    for _ in tqdm(range(len(dataloader)), desc="Sequence {}: ".format(args.sequence)):
        # Get frame, ground truth pose and frame_id from dataset
        frame, pose, frame_id = dataloader.get_next_data()

        # Apply VO motion estimation algorithm
        vo.update(frame, frame_id)

        # Get estimated translation
        estimated_t = vo.t.flatten()
        [x, y, z] = estimated_t
        [x_true, y_true, z_true] = [pose[0], pose[1], pose[2]]

        # Store all estimated poses (4x4)
        poses.append(vo.pose)

        # Store trajectories
        estimated_trajectory.append(estimated_t)
        gt_trajectory.append(pose)

        # Draw trajectory
        if args.display_traj:
            cv2.circle(trajectory, (int(x)+350, int(-z)+610), 1, (255, 0, 0), 1)
            cv2.circle(trajectory, (int(x_true)+350, int(-z_true)+610), 1, (0, 0, 255), 2)
            cv2.rectangle(trajectory, (10, 20), (600, 81), (255, 255, 255), -1)  # background to display MSE
            cv2.putText(trajectory, "Ground truth (RED)", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, 8)
            cv2.putText(trajectory, "Estimated (BLUE)", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1, 8)
            # compute and display distance
            MSE = np.linalg.norm(np.array([x, z]) - np.array([x_true, z_true]))
            cv2.putText(trajectory, "Frobenius Norm: {:.2f}".format(MSE), (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, 8)

            cv2.imshow("Camera", frame)
            cv2.imshow("Visual Odometry", trajectory)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Save predicted poses
    save_trajectory(poses, args.sequence, save_dir="results")

    # Save image map
    if args.display_traj:
        cv2.imwrite("results/maps/map_{}.png".format(args.sequence), trajectory)

    # Plot estimated trajectory
    plot_trajectory(gt_trajectory, estimated_trajectory,
                    save_name="results/plots/plot_{}.png".format(args.sequence))