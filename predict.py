import os

import matplotlib.pyplot as plt
import pydicom
from matplotlib.pyplot import gray

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pydicom as dicom
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
import scipy
from scipy import ndimage
import numpy as np
import cv2
import os
""" Creating a directory """


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

"""
def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2  # use the double slash // operator to perform floor division
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    return hu_image
"""
if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("test_2")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset """
    test_x = glob("data/test/*/*.dcm")
    print(f"Test: {len(test_x)}")

    """ Loop over the data """
    for x in tqdm(test_x):
        """ Extract the names """
        dir_name = x.split("/")[-2]
        name = dir_name + "_" + x.split("/")[-1].split(".")[0]

        """ Read the image """
        #medical_image = dicom.dcmread(x)
        medical_image = pydicom.read_file(x)
        image = medical_image.pixel_array
        image = np.expand_dims(image, axis=-1)
        image = image / np.max(image) * 255.0
        x = image / 255.0
        x = np.concatenate([x, x, x], axis=-1)
        x = np.expand_dims(x, axis=0)

        """ Prediction """
        mask = model.predict(x)[0]
        mask = mask > 0.5
        mask = mask.astype(np.int32)
        mask = mask * 255
        print("image", image.shape)
        print("mask", mask.shape)

        new_image = image.reshape(image.shape[0], (image.shape[1]) * image.shape[2])
        print("new image", new_image.shape)
        new_mask = mask.reshape(mask.shape[0], (mask.shape[1]) * mask.shape[2])
        print("new mask", new_mask.shape)

# hu transform
        intercept = medical_image.RescaleIntercept
        slope = medical_image.RescaleSlope
        hu_image = image * slope + intercept
        print("hu", hu_image.shape)

# windowing
        window_image = hu_image.copy()
        window_center = 40
        window_width = 80
        img_min = window_center - window_width // 2  # use the double slash // operator to perform floor division
        img_max = window_center + window_width // 2

        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max
        #window_image = (window_image - img_min) / (img_max - img_min)*255.0
        print("window", window_image.shape)

        brain_img = window_image
        """hu_image = transform_to_hu(medical_image, image)
        brain_img = window_image(hu_image, 40, 80)"""
        print("brain", brain_img.shape)

# min max pixels of images
        smallest_image = image.min(axis=(0, 1))
        smallest_hu = hu_image.min(axis=(0, 1))
        smallest_window = window_image.min(axis=(0, 1))
        smallest_brain = brain_img.min(axis=(0, 1))
        largest_image = image.max(axis=(0, 1))
        largest_hu = hu_image.max(axis=(0, 1))
        largest_window = window_image.max(axis=(0, 1))
        largest_brain = brain_img.max(axis=(0, 1))

        print(smallest_image, smallest_hu, smallest_window, smallest_brain)
        print(largest_image, largest_hu, largest_window, largest_brain)

# write images
        segmented = mask * brain_img
        cv2.imwrite(f"test_2/{name} seg.png", segmented)
        cat_images = np.concatenate([image, mask, hu_image, brain_img, window_image], axis=1)
        cv2.imwrite(f"test_2/{name}.png", cat_images)
