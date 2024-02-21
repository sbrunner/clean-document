# Currently only build_autoencoder gives acceptable results
# No good results with bigger images
#
# TODO:
# - Try in color


import glob
import os.path
import shutil

import cv2
import keras.saving
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import BatchNormalization, Conv2D, Input, LeakyReLU, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras_preprocessing.image import img_to_array, load_img, save_img
from scipy import ndimage

IMAGE_DIMENSION = 256


def data_augmentation_clean_():
    shutil.rmtree("augmented_data", ignore_errors=True)

    images = glob.glob(os.path.join("clean-data/*.png"))
    for image_filename in images:
        data_augmentation_image(image_filename, "train-xy")


def data_augmentation_(xy=False):
    shutil.rmtree("augmented_data", ignore_errors=True)

    for category in ("train-xy",) if xy else ("train-x", "train-y"):
        images = (
            glob.glob("clean-sata/*.png") if xy else glob.glob(os.path.join("data", category, "sbr*.png"))
        )
        for image_filename in images:
            data_augmentation_image(image_filename, category)


MODELS = ()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sub", action="store_true")
parser.add_argument("--one", action="store_true")
parser.add_argument("--apply", action="store_true")

args = parser.parse_args()

if args.sub:
    while True:
        subprocess.run(["python", "try-ai.py", "--one"], check=True)

if args.one:
    # get_original_pdf()
    # convert_original_pdf_to_png()
    # data_augmentation(xy=True)
    train_many()
    exit()

if args.apply:
    do_apply()
    exit()

# list_pdf_images('/tmp/test.pdf')
data_augmentation()
train()
do_apply()
