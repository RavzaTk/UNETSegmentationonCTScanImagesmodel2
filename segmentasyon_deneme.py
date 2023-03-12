import pydicom
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from skimage import morphology


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2  # use the double slash // operator to perform floor division
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    smallest_window = window_image.min(axis=(0, 1))
    largest_window = window_image.max(axis=(0, 1))
    print("wmin / wmax",  smallest_window, largest_window)
    print("window", window_image.shape)
    return window_image


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope
    hu_image = image * slope + intercept
    print("hu", hu_image.shape)
    return hu_image


def remove_noise(medical_image):
    image = medical_image.pixel_array
    smallest_image = image.min(axis=(0, 1))
    largest_image = image.max(axis=(0, 1))
    print("imin / imax", smallest_image, largest_image)
    print("image", image.shape)
    hu_image = transform_to_hu(medical_image, image)
    smallest_hu = hu_image.min(axis=(0, 1))
    largest_hu = hu_image.max(axis=(0, 1))
    print("hmin / hmax", smallest_hu, largest_hu)
    brain_img = window_image(hu_image, 40, 80)
    smallest_brain = brain_img.min(axis=(0, 1))
    largest_brain = brain_img.max(axis=(0, 1))
    print("bmin / bmax", smallest_brain, largest_brain)
    print("brain", brain_img.shape)
    segmentation = morphology.dilation(brain_img, np.ones((1, 1)))
    labels, label_nb = ndimage.label(segmentation)
    label_count = np.bincount(labels.ravel().astype(int))
    label_count[0] = 0
    mask = labels == label_count.argmax()

    mask = morphology.dilation(mask, np.ones((1, 1)))
    mask = scipy.ndimage.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    masked_image = mask * brain_img
    print("masked", masked_image.shape)
    return mask, masked_image, brain_img


path = "data/test/patient40/CT000001.dcm"
medical_image = pydicom.read_file(path)
plt.subplot(1,4,1)
plt.title('original')
plt.style.use('grayscale')
plt.imshow(medical_image.pixel_array)
plt.plot()

mask, masked_image, brain_img = remove_noise(medical_image)
plt.subplot(1,4,2)
plt.style.use('grayscale')
plt.title('mask')
plt.imshow(mask)
plt.plot()

plt.subplot(1,4,3)
plt.imshow(masked_image)
plt.title('segmented image')
plt.plot()

plt.subplot(1,4,4)
plt.imshow(brain_img)
plt.title('brain image')
plt.plot()
plt.show()

