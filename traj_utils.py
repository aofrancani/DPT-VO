import os
import matplotlib.pyplot as plt


def plot_trajectory(gt_poses, estimated_poses, save_name):
    """
    Plot estimated and ground truth trajectory.

    Args:
        gt_poses {list}: list with ground truth poses of trajectory [x_true, y_true, z_true]
        estimated_poses {list}: list with estimated poses of trajectory [x, y, z]
    """
    plt.figure()
    # Plot estimated trajectory
    plt.plot([x[0] for x in estimated_poses], [z[2] for z in estimated_poses], "b")
    # Plot ground truth trajectory
    plt.plot([x[0] for x in gt_poses], [z[2] for z in gt_poses], "r")

    plt.grid()
    plt.title("Visual Odometry")
    plt.xlabel("Translation in x direction [m]")
    plt.ylabel("Translation in z direction [m]")
    plt.legend(["estimated", "ground truth"])
    plt.savefig(save_name)


def save_trajectory(poses, sequence, save_dir):
    """
    Save predicted poses in .txt file

    Args:
        poses {ndarray}: list with all 4x4 pose matrix
        sequence {str}: sequence of KITTI dataset
        save_dir {str}: path to save pose
    """
    # create directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_filename = os.path.join(save_dir, "{}.txt".format(sequence))
    with open(output_filename, "w") as f:
        for pose in poses:
            pose = pose.flatten()[:12]
            line = " ".join([str(x) for x in pose]) + "\n"
            # line = f"{pose[0]:.4f}" + " " + f"{pose[1]:.4f}" + " " + f"{pose[2]:.4f}" + "\n"
            f.write(line)
