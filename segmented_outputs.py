import pydicom
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from skimage import morphology
import tensorflow as tf
from tqdm import tqdm
from glob import glob
import pydicom as dicom
""" Creating a directory """


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("segmented_outputs")


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


    def remove_noise(medical_image):
        image = medical_image.pixel_array
        hu_image = transform_to_hu(medical_image, image)
        brain_img = window_image(hu_image, 40, 80)

        segmentation = morphology.dilation(brain_img, np.ones((1, 1)))
        labels, label_nb = ndimage.label(segmentation)
        label_count = np.bincount(labels.ravel().astype(int))
        label_count[0] = 0
        mask = labels == label_count.argmax()

        mask = morphology.dilation(mask, np.ones((1, 1)))
        mask = scipy.ndimage.binary_fill_holes(mask)
        mask = morphology.dilation(mask, np.ones((3, 3)))
        masked_image = mask * brain_img
        return mask, masked_image

    original_x = glob("data/test/patient40/*.dcm")
    print(f"Test: {len(original_x)}")


    for x in tqdm(original_x):
        """ Extract the names """
        dir_name = x.split("/")[-2]
        name = dir_name + "_" + x.split("/")[-1].split(".")[0]

        medical_image = dicom.dcmread(x)

        mask, masked_image = remove_noise(medical_image)


        print(masked_image.shape)
        print(medical_image.shape)
        cat_images = np.concatenate([medical_image, masked_image], axis=1)
        cv2.imwrite(f"test_2/{name}.png", cat_images)