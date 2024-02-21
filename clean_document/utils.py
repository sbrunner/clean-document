import math
import os.path

import cv2
import numpy as np
from scipy import ndimage


def data_augmentation_image(image_filename, image_dimension, category=None, save=True):
    results = []
    image = cv2.imread(image_filename)
    # to yuv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # split the image into regular IMAGE_DIMENSIONxIMAGE_DIMENSION images
    nb_x = math.ceil(image.shape[0] / image_dimension)
    nb_y = math.ceil(image.shape[1] / image_dimension)
    slide_x = (image.shape[0] - image_dimension) / (nb_x - 1)
    slide_y = (image.shape[1] - image_dimension) / (nb_y - 1)
    for x in range(nb_x):
        for y in range(nb_y):
            x0 = round(x * slide_x)
            y0 = round(y * slide_y)
            if x0 + image_dimension > image.shape[0]:
                print(nb_x, x, slide_x)
                print(image.shape, x0 + image_dimension, y0 + image_dimension)
            if y0 + image_dimension > image.shape[1]:
                print(nb_y, y, slide_y)
                print(image.shape, x0 + image_dimension, y0 + image_dimension)
            cropped = image[x0 : x0 + image_dimension, y0 : y0 + image_dimension]

            base = f"-{x}-{y}-"
            # base = '-'
            # cropped = image
            dest_folder = None
            if save:
                dest_folder = os.path.join("augmented_data", category)
                if not os.path.exists(dest_folder):
                    os.makedirs(dest_folder)

            results += [cropped]
            if save:
                cv2.imwrite(
                    os.path.join(
                        dest_folder,
                        f"base{base}{os.path.basename(image_filename)}",
                    ),
                    results[-1],
                )
            results += [ndimage.rotate(cropped, 180)]
            if save:
                cv2.imwrite(
                    os.path.join(
                        dest_folder,
                        f"rotated{base}{os.path.basename(image_filename)}",
                    ),
                    results[-1],
                )
            results += [np.fliplr(cropped)]
            if save:
                cv2.imwrite(
                    os.path.join(
                        dest_folder,
                        f"fliplr{base}{os.path.basename(image_filename)}",
                    ),
                    results[-1],
                )
            results += [np.flip(cropped)]
            if save:
                cv2.imwrite(
                    os.path.join(
                        dest_folder,
                        f"flipud{base}{os.path.basename(image_filename)}",
                    ),
                    results[-1],
                )

    # scale = random.uniform(0.8, 0.9)
    scale = 0.8
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    nb_x = math.ceil(image.shape[0] / image_dimension)
    nb_y = math.ceil(image.shape[1] / image_dimension)
    slide_x = (image.shape[0] - image_dimension) / (nb_x - 1)
    slide_y = (image.shape[1] - image_dimension) / (nb_y - 1)
    for x in range(nb_x):
        for y in range(nb_y):
            x0 = round(x * slide_x)
            y0 = round(y * slide_y)
            if x0 + image_dimension > image.shape[0]:
                print(nb_x, x, slide_x)
                print(image.shape, x0 + image_dimension, y0 + image_dimension)
            if y0 + image_dimension > image.shape[1]:
                print(nb_y, y, slide_y)
                print(image.shape, x0 + image_dimension, y0 + image_dimension)
            cropped = image[x0 : x0 + image_dimension, y0 : y0 + image_dimension]

            results += [cropped]
            if save:
                base = f"-{x}-{y}-"
                cv2.imwrite(
                    os.path.join(
                        dest_folder,
                        f"zoom{base}{os.path.basename(image_filename)}",
                    ),
                    results[-1],
                )
    return results


def train_val_split(x_train, y_train):
    x_train = x_train
    y_train = y_train
    rnd = np.random.RandomState(seed=42)
    perm = rnd.permutation(len(x_train))
    train_idx = perm[: int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)) :]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]
