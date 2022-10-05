# DPT-VO: Dense Prediction Transformer for Scale Estimation in Monocular Visual Odometry

[![arXiv](https://img.shields.io/badge/cs.CV-arXiv%3A2210.01723-B31B1B.svg)](https://arxiv.org/abs/2210.01723)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/aofrancani/DPT-VO/blob/main/LICENSE)

Official repository of the paper "[Dense Prediction Transformer for Scale Estimation in Monocular Visual Odometry](https://arxiv.org/abs/2210.01723)"

<img src="seq_00.gif" width=1000>

## Abstract
*Monocular visual odometry consists of the estimation of the position of an agent through images of a single camera, and it is applied in autonomous vehicles, medical robots, and augmented reality. However, monocular systems suffer from the scale ambiguity problem due to the lack of depth information in 2D frames. This paper contributes by showing an application of the dense prediction transformer model for scale estimation in monocular visual odometry systems. Experimental results show that the scale drift problem of monocular systems can be reduced through the accurate estimation of the depth map by this model, achieving competitive state-of-the-art performance on a visual odometry benchmark.*


## Contents
1. [Dataset](#1-dataset)
2. [Download the DPT Model](#2-download-the-dpt-model)
3. [Setup](#3-setup)
4. [Usage](#4-usage)
5. [Evaluation](#5-evaluation)


## 1. Dataset
Download the [KITTI odometry dataset (grayscale).](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

In this work, we use the `.jpg` format. You can convert the dataset to `.jpg` format with [png_to_jpg.py.](https://github.com/aofrancani/DPT-VO/blob/main/util/png_to_jpg.py)

Create a simbolic link (Windows) or a softlink (Linux) to the dataset in the `dataset` folder:

- On Windows:
```mklink /D <path_to_your_project>\DPT-VO\dataset <path_to_your_downloaded_dataset>```
- On Linux: 
```ln -s <path_to_your_downloaded_dataset> <path_to_your_project>/DPT-VO/dataset```

Then, the data structure should be as follows:
```
|---DPT-VO
    |---dataset
        |---sequences_jpg
            |---00
                |---image_0
                    |---000000.png
                    |---000001.png
                    |---...
                |---image_1
                    |...
                |---image_2
                    |---...
                |---image_3
                    |---...
            |---01
            |---...
```

## 2. Download the DPT Model
Download the [DPT trained weights](https://drive.google.com/file/d/1-oJpORoJEdxj4LTV-Pc17iB-smp-khcX/view) and save it in the `weights` folder.
- [dpt_hybrid_kitti-cb926ef4.pt](https://drive.google.com/file/d/1-oJpORoJEdxj4LTV-Pc17iB-smp-khcX/view)

For more details please check the [original DPT repository](https://github.com/isl-org/DPT).


## 3. Setup
- Create a virtual environment using Anaconda and activate it:
```
conda create -n dpt-vo python==3.8.0
conda activate dpt-vo
```
- Install dependencies (with environment activated):
```
pip install -r requirements.txt
```

## 4. Usage
Run the `main.py` code with the following command:
```
python main.py  -s <sequence_number>
```
You can also use a different path to dataset by changing the arguments ``--data_path`` and ``--pose_path``:
```
python main.py -d <path_to_dataset> -p <path_to_gt_poses> -s <sequence_number>
```

## 5. Evaluation
The evalutaion is done with the [KITTI odometry evaluation toolbox](https://github.com/Huangying-Zhan/kitti-odom-eval). Please go to the [evaluation repository](https://github.com/Huangying-Zhan/kitti-odom-eval) to see more details about the evaluation metrics and how to run the toolbox.


## Citation
Please cite our paper you find this research useful in your work:

```bibtex
@INPROCEEDINGS{Francani2022,
    title={Dense Prediction Transformer for Scale Estimation in Monocular Visual Odometry},
    author={André O. Françani and Marcos R. O. A. Maximo},
    booktitle={2022 Latin American Robotics Symposium (LARS), 2022 Brazilian Symposium on Robotics (SBR), and 2022 Workshop on Robotics in Education (WRE)},
    days={18-21},
    month={oct},
    year={2022},
}
```

## References
Some of the functions were borrowed and adapted from three amazing works, which are: [DPT](https://github.com/isl-org/DPT), [DF-VO](https://github.com/Huangying-Zhan/DF-VO), and [monoVO](https://github.com/uoip/monoVO-python).
