import glob
import math
import os.path
import random

import numpy as np
from scipy import ndimage
from skimage import color, exposure, io, morphology, transform
from skimage.draw import disk

dirty_images = []
dirty_shape = []


def data_augmentation_image(
    image_filename, image_dimension, category=None, save=True, segmentation=False, test=False
):
    global dirty_images, dirty_shape
    if not dirty_images:
        dirty_images = [io.imread(filename) for filename in glob.glob("data/dirty/*.png")]
        x = max([image.shape[0] for image in dirty_images])
        y = max([image.shape[1] for image in dirty_images])
        dirty_shape = (x, y)

    results = []
    image = io.imread(image_filename)

    if not segmentation:
        # to yuv
        image = color.rgb2yuv(image)
    else:
        if test:
            image_bw = color.rgb2gray(image)
            image_bw = np.expand_dims(image_bw, axis=-1)
            image_2 = np.ones(
                [max(dirty_shape[0], image.shape[0]), max(dirty_shape[1], image.shape[1]), 1], dtype=np.uint8
            )
            # padding image_bw with 255 to fit with image_2
            image_bw = np.pad(
                image_bw,
                (
                    (0, image_2.shape[0] - image_bw.shape[0]),
                    (0, image_2.shape[1] - image_bw.shape[1]),
                    (0, 0),
                ),
                mode="constant",
                constant_values=255,
            )

            image_2[image_bw < 250] = 0
            radius = 6
            kernel = np.zeros((2 * radius - 1, 2 * radius - 1), np.uint8)
            # draw a circle in the center
            radius = kernel.shape[0] // 2
            kernel = disk(radius)

            image_2 = morphology.erosion(image_2, kernel)
            image = image_2
        else:
            dirty_image = random.choice(dirty_images)
            image_dirty = np.zeros(
                [max(dirty_shape[0], image.shape[0]), max(dirty_shape[1], image.shape[1]), 3],
                dtype=np.uint8,
            )
            image_dirty[0 : image.shape[0], 0 : image.shape[1]] = image
            image_dirty[0 : dirty_image.shape[0], 0 : dirty_image.shape[1]] -= 255 - dirty_image
            image = image_dirty
            image = color.rgb2yuv(image)

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
            assert len(results[-1].shape) == 3
            if save:
                io.imwrite(
                    os.path.join(
                        dest_folder,
                        f"base{base}{os.path.basename(image_filename)}",
                    ),
                    color.yuv2rgb(results[-1]),
                )
            results += [ndimage.rotate(cropped, 180)]
            assert len(results[-1].shape) == 3
            if save:
                io.imwrite(
                    os.path.join(
                        dest_folder,
                        f"rotated{base}{os.path.basename(image_filename)}",
                    ),
                    color.yuv2rgb(results[-1]),
                )
            results += [np.fliplr(cropped)]
            assert len(results[-1].shape) == 3
            if save:
                io.imwrite(
                    os.path.join(
                        dest_folder,
                        f"fliplr{base}{os.path.basename(image_filename)}",
                    ),
                    color.yuv2rgb(results[-1]),
                )
            # flip up down
            results += [np.flipud(cropped)]
            if save:
                io.imwrite(
                    os.path.join(
                        dest_folder,
                        f"flipud{base}{os.path.basename(image_filename)}",
                    ),
                    color.yuv2rgb(results[-1]),
                )

            if not segmentation:
                results += [np.flip(cropped)]
                assert len(results[-1].shape) == 3
                if save:
                    io.imwrite(
                        os.path.join(
                            dest_folder,
                            f"flip{base}{os.path.basename(image_filename)}",
                        ),
                        color.yuv2rgb(results[-1]),
                    )

                # Change the background color (white), and the foreground color (black)
                background_huv = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                forground_huv = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
                # hsv to yuv
                background_rgb = color.hsv2rgb(np.uint8([[background_huv]]) / 255.0)
                background_yuv = color.rgb2yuv(background_rgb)
                background_yuv = background_yuv[0][0]
                forground_rgb = color.hsv2rgb(np.uint8([[forground_huv]]) / 255.0)
                forground_yuv = color.rgb2yuv(forground_rgb)
                forground_yuv = forground_yuv[0][0]
                full_background = (
                    np.zeros([cropped.shape[0], cropped.shape[1], 3], dtype=np.uint8) + background_yuv
                )
                full_forground = (
                    np.zeros([cropped.shape[0], cropped.shape[1], 3], dtype=np.uint8) + forground_yuv
                )

                # Convertir l'image de YUV à RGB
                image_rgb = color.yuv2rgb(cropped)

                # Convertir l'image en niveaux de gris
                image_gray = color.rgb2gray(image_rgb)

                # Normaliser l'image en niveaux de gris à la plage [0, 1]
                image_gray = exposure.rescale_intensity(image_gray, out_range=(0, 1))

                # Fusionner les canaux de l'image
                i = np.dstack(
                    (
                        image_gray * full_background[:, :, 0],
                        image_gray * full_background[:, :, 1],
                        image_gray * full_background[:, :, 2],
                    )
                )

                j = np.dstack(
                    (
                        (1 - image_gray) * full_forground[:, :, 0],
                        (1 - image_gray) * full_forground[:, :, 1],
                        (1 - image_gray) * full_forground[:, :, 2],
                    )
                )
                results += [(i + j).astype(np.uint8)]
                assert len(results[-1].shape) == 3
                if save:
                    io.imwrite(
                        os.path.join(
                            dest_folder,
                            f"colored{base}{os.path.basename(image_filename)}",
                        ),
                        color.yuv2rgb(results[-1]),
                    )

    if segmentation is False:
        scale = random.uniform(0.5, 0.9)
        # scale = 0.8
        image = transform.rescale(image, scale, mode="reflect", multichannel=True)

        nb_x = math.ceil(image.shape[0] / image_dimension)
        nb_y = math.ceil(image.shape[1] / image_dimension)
        slide_x = (image.shape[0] - image_dimension) / (nb_x - 1) if nb_x > 1 else 0
        slide_y = (image.shape[1] - image_dimension) / (nb_y - 1) if nb_y > 1 else 0
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
                assert len(image.shape) == 3
                cropped = image[x0 : x0 + image_dimension, y0 : y0 + image_dimension]

                results += [cropped]
                assert len(results[-1].shape) == 3
                if save:
                    base = f"-{x}-{y}-"
                    io.imwrite(
                        os.path.join(
                            dest_folder,
                            f"zoom{base}{os.path.basename(image_filename)}",
                        ),
                        color.yuv2rgb(results[-1]),
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
