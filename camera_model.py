import numpy as np


class CameraModel(object):
    """
    Class that represents a pin-hole camera model (or projective camera model).
    In the pin-hole camera model, light goes through the camera center (cx, cy) before its projection
    onto the image plane.
    """
    def __init__(self, params):
        """
        Creates a camera model

        Arguments:
            params {dict} -- Camera parameters
        """

        # Image resolution
        self.width = params['width']
        self.height = params['height']
        # Focal length of camera
        self.fx = params['fx']
        self.fy = params['fy']
        # Optical center (principal point)
        self.cx = params['cx']
        self.cy = params['cy']
        # Distortion coefficients.
        # k1, k2, and k3 are the radial coefficients.
        # p1 and p2 are the tangential distortion coefficients.
        #self.distortion_coeff = [params['k1'], params['k2'], params['p1'], params['p2'], params['k3']]
        self.mat = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]])