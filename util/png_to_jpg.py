import os
from glob import glob
from PIL import Image
from tqdm import tqdm

""" Convert png to jpg images for KITTI dataset

Data structure:
|---dataset
    |---sequences
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
"""

dataset_dir = f"..\dataset\sequences"
new_root_dir = f"..\dataset\sequences_jpg"
image_nb = 0  # number of image folder (image_0, image_1, ... image_3)

# create new directory
if not os.path.exists(new_root_dir):
    os.makedirs(new_root_dir)

for seq_nb in range(22): # list all sequences ["00", ... "21"]
    # sequence as 2-digit string
    seq = "{:02d}".format(seq_nb)
    
    # create seq in save directory
    seq_dir = os.path.join(new_root_dir, seq)
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)
        
    # create image_nb directory
    img_nb_dir = os.path.join(seq_dir, "image_{}".format(image_nb))
    if not os.path.exists(img_nb_dir):
        os.makedirs(img_nb_dir)
    
    images_list = glob(os.path.join(dataset_dir, seq, "image_{}".format(image_nb), "*"))  # paths to png images in seq
    
    for i in tqdm(range(len(images_list)), desc="Sequence {}: ".format(seq)):
        img_path = images_list[i]
        img_name = os.path.basename(img_path)
        
        # read png image and convert to jpg using PIL
        img = Image.open(img_path)
        save_dir = os.path.join(img_nb_dir, img_name.replace(".png", ".jpg"))
        img.save(save_dir)
