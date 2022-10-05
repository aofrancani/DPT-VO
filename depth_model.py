"""
Build DPT depth model
 - modified from https://github.com/isl-org/DPT
"""

import os
import torch
import cv2
import argparse
import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


class DepthModel(object):
    """
    Build DPT network and compute depth maps
    """

    def __init__(self, model_type="dpt_hybrid", optimize=True):
        """
        Build MonoDepthNN to compute depth maps.

        Arguments:
            model_path (str): path to saved model
        """
        default_models = {
            "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
            "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
            "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
        }
        model_path = default_models[model_type]
        self.model_type = model_type
        self.optimize = optimize

        # select device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)

        # load network
        if model_type == "dpt_large":  # DPT-Large
            net_w = net_h = 384
            model = DPTDepthModel(
                path=model_path,
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "dpt_hybrid":  # DPT-Hybrid
            net_w = net_h = 384
            model = DPTDepthModel(
                path=model_path,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        elif model_type == "dpt_hybrid_kitti":
            net_w = 1216
            net_h = 352

            model = DPTDepthModel(
                path=model_path,
                scale=0.00006016,
                shift=0.00579,
                invert=True,
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )

            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        else:
            assert (
                False
            ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        model.eval()

        if optimize and self.device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)
            model = model.half()

        self.model = model.to(self.device)

    @torch.no_grad()
    def compute_depth(self, img, kitti_crop=False):
        """
        Computes depth map

        Arguments:
            img (array): image (0-255)
        """

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        if kitti_crop is True:
            height, width, _ = img.shape
            top = height - 352
            left = (width - 1216) // 2
            img = img[top : top + 352, left : left + 1216, :]

        img_input = self.transform({"image": img})["image"]

        # with torch.no_grad():
        sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)

        if self.optimize and self.device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        # if self.model_type == "dpt_hybrid_kitti":
        #     prediction *= 256

        return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model_weights", default=None, help="path to model weights"
    )

    parser.add_argument(
        "-t",
        "--model_type",
        default="dpt_hybrid",
        help="model type [dpt_large|dpt_hybrid|midas_v21]",
    )

    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.set_defaults(optimize=True)

    args = parser.parse_args()

    # default_models = {
    #     "midas_v21": "weights/midas_v21-f6b98070.pt",
    #     "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
    #     "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    #     "dpt_hybrid_kitti": "weights/dpt_hybrid_kitti-cb926ef4.pt",
    #     "dpt_hybrid_nyu": "weights/dpt_hybrid_nyu-2ce69ec7.pt",
    # }
    #
    # if args.model_weights is None:
    #     args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # build model
    model = DepthModel(
        args.model_type,
        args.optimize,
    )
    # print(model)

    # read img
    img_path = r"dataset\sequences_jpg\00\image_0\000000.jpg"
    img = cv2.imread(img_path)

    # compute depth
    depth = model.compute_depth(img, kitti_crop=False)

    filename = os.path.join(
        "temp", os.path.splitext(os.path.basename(img_path))[0]
    )
    util.io.write_depth(filename.replace(".jpg", "_depth.jpg"), depth, bits=2, absolute_depth=True)



