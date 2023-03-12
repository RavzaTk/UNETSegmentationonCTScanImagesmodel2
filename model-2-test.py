import os

import pydicom

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
import pydicom as dicom
from pydicom import dcmread
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

""" Creating a directory """


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("test")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Load the dataset """
    test_x = glob("data/test/patient188/*.dcm")
    print(f"Test: {len(test_x)}")


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


    """ Loop over the data """
    for x in tqdm(test_x):
        """ Extract the names """
        dir_name = x.split("/")[-3]
        name = dir_name + "_" + x.split("/")[-1].split(".")[0]

        """ Read the image """
        medical_image = pydicom.read_file(x)
        image = medical_image.pixel_array

        print("image", image.shape)

        hu_image = transform_to_hu(medical_image, image)
        #hu_image = np.expand_dims(hu_image, axis=-1)


        brain_img = window_image(hu_image, 40, 80)
        print("brain", brain_img.shape)

        image = np.expand_dims(image, axis=-1)
        print("image", image.shape)
        image = image / np.max(image) * 255.0
        x = image / 255.0
        x = np.concatenate([x, x, x], axis=-1)
        x = np.expand_dims(x, axis=0)
        print("x", x.shape)

        """ Prediction """
        mask = model.predict(x)[0]
        print("mask", mask.shape)
        mask = mask > 0.5
        print("mask", mask.shape)
        mask = mask.astype(np.int32)
        mask = mask * 255
        print("mask", mask.shape)
        segmented_image = mask * brain_img
        cat_images = np.concatenate([hu_image, brain_img], axis=1)
        cv2.imwrite(f"test/{name}.png", cat_images)
