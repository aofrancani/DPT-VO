import glob
import os
import cv2
import argparse


class KITTI:
    def __init__(self,
                 data_path=r"dataset\sequences_jpg",
                 pose_path=r"dataset\poses",
                 sequence="00",
                 camera_id="0",
                 ):
        """
        Dataloader for KITTI Visual Odometry Dataset
            http://www.cvlibs.net/datasets/kitti/eval_odometry.php

        Arguments:
            data_path {str}: path to data sequences
            pose_path {str}: path to poses
            sequence {str}: sequence to be tested (default: "00")
        """
        self.data_path = data_path
        self.sequence = sequence
        self.camera_id = camera_id
        self.frame_id = 0

        # Read ground truth poses
        with open(os.path.join(pose_path, sequence+".txt")) as f:
            self.poses = f.readlines()

        # Get frames list
        frames_dir = os.path.join(data_path, sequence, "image_{}".format(camera_id), "*.jpg")
        self.frames = sorted(glob.glob(frames_dir))

        # Camera Parameters
        self.cam_params = {}
        frame = cv2.imread(self.frames[self.frame_id], 0)
        self.cam_params["width"] = frame.shape[0]
        self.cam_params["height"] = frame.shape[1]
        self.read_intrinsics_param()

    def __len__(self):
        return len(self.frames)

    def get_next_data(self):
        """
        Returns:
            frame {ndarray}: image frame at index self.frame_id
            pose {list}: list containing the ground truth pose [x, y, z]
            frame_id {int}: integer representing the frame index
        """
        # Read frame as grayscale
        frame = cv2.imread(self.frames[self.frame_id], 0)
        self.cam_params["width"] = frame.shape[0]
        self.cam_params["height"] = frame.shape[0]

        # Read poses
        pose = self.poses[self.frame_id]
        pose = pose.strip().split()
        pose = [float(pose[3]), float(pose[7]), float(pose[11])]  # coordinates for the left camera
        frame_id = self.frame_id
        self.frame_id = self.frame_id + 1
        return frame, pose, frame_id

    # def get_limits(self):
    #     """
    #     Returns the limits to draw max and min poses in trajectory image
    #
    #     Returns:
    #         x_lim {list}: float representing max and min poses of x
    #         y_lim {list}: float representing max and min poses of y
    #         z_lim {list}: float representing max and min poses of z
    #     """
    #     poses = [pose.strip().split() for pose in self.poses]
    #     x_lim = [max([pose[0] for pose in poses]), min([pose[0] for pose in poses])]
    #     y_lim = [max([pose[1] for pose in poses]), min([pose[1] for pose in poses])]
    #     z_lim = [max([pose[2] for pose in poses]), min([pose[2] for pose in poses])]
    #     return x_lim, y_lim, z_lim

    def read_intrinsics_param(self):
        """
        Reads camera intrinsics parameters

        Returns:
            cam_params {dict}: dictionary with focal lenght and principal point
        """
        calib_file = os.path.join(self.data_path, self.sequence, "calib.txt")
        with open(calib_file, 'r') as f:
            lines = f.readlines()
            line = lines[int(self.camera_id)].strip().split()
            [fx, cx, fy, cy] = [float(line[1]), float(line[3]), float(line[6]), float(line[7])]

            # focal length of camera
            self.cam_params["fx"] = fx
            self.cam_params["fy"] = fy
            # principal point (optical center)
            self.cam_params["cx"] = cx
            self.cam_params["cy"] = cy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--data_path",
        default=r"dataset\sequences_jpg",
        help="path to dataset"
    )
    parser.add_argument(
        "-s",
        "--sequence",
        default="03",
        help="sequence to be evaluated",
    )
    parser.add_argument(
        "-p",
        "--pose_path",
        default=r"dataset\poses",
        help="path to ground truth poses",
    )

    args = parser.parse_args()

    # Create dataloader
    dataloader = KITTI(
        data_path=args.data_path,
        pose_path=args.pose_path,
        sequence=args.sequence,
    )

    dataloader.read_intrinsics_param()


