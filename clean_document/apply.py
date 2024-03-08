import math
import os

import cv2
import keras.models
import numpy as np
import tensorflow as tf


def apply(
    model: keras.models.Model,
    image_src_filename: str,
    image_dst_filename: str,
    yuv: bool = True,
    margin: int = 5,
    segmentation: bool = True,
    square_size: int = -1,
):
    model_input_shape = model.input_shape[1:]
    model_output_shape = model.output_shape[1:]

    image = cv2.imread(image_src_filename)
    original_image = None
    predictions = None
    if segmentation:
        # Copy the image to avoid modifying the original
        original_image = image.copy()

    if square_size > 0:
        x = math.ceil(image.shape[0] / square_size) * square_size
        y = math.ceil(image.shape[1] / square_size) * square_size
        # 255 padding on the image to fit with the square_size
        image = np.pad(
            image,
            ((0, x - image.shape[0]), (0, y - image.shape[1]), (0, 0)),
            mode="constant",
            constant_values=255,
        )

    if model_input_shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = np.expand_dims(image, axis=-1)
    elif not yuv:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    if model_input_shape[0] is None and model_input_shape[1] is None:
        sample_test_img = np.expand_dims(image, axis=0).astype("float32") / 255.0
        # Get the prediction
        predictions = model.predict(sample_test_img)[0] * 255.0
        if segmentation:
            image = predictions
        else:
            predictions = predictions.astype("uint8")
            if model_input_shape[2] == 1:
                image[:, :, 0] = predictions
            else:
                image = predictions
    else:
        assert model_input_shape[0] is not None
        assert model_input_shape[1] is not None
        images = []

        image_to_crop = image
        if model_input_shape[2] == 1:
            image_to_crop = image[:, :, 0]

        # split the image into regular image_dimension image_dimension images
        nb_x = math.ceil(image.shape[0] / (model_input_shape[0] - 2 * margin))
        nb_y = math.ceil(image.shape[1] / (model_input_shape[1] - 2 * margin))
        slide_x = (image.shape[0] - model_input_shape[0]) / (nb_x - 1)
        slide_y = (image.shape[1] - model_input_shape[1]) / (nb_y - 1)
        for x in range(nb_x):
            for y in range(nb_y):
                x0 = round(x * slide_x)
                y0 = round(y * slide_y)
                cropped = image_to_crop[x0 : x0 + model_input_shape[0], y0 : y0 + model_input_shape[1], :]

                sample_test_img = cropped.astype("float32") / 255.0
                images.append(sample_test_img)

        samples = np.zeros((len(images), model_input_shape[0], model_input_shape[1], model_input_shape[2]))
        for nb, img in enumerate(images):
            samples[nb] = img

        predictions = model.predict(samples)
        if not segmentation:
            predictions = (predictions * 255.0).astype("uint8")
        index = 0

        image = np.zeros((image.shape[0], image.shape[1], model_output_shape[2]), dtype="uint8")

        for x in range(nb_x):
            for y in range(nb_y):
                x0 = round(x * slide_x)
                y0 = round(y * slide_y)

                # Get the prediction
                predicted = predictions[index]
                index += 1

                # Update image with the prediction
                min_x = 0 if x == 0 else x0 + margin
                min_y = 0 if y == 0 else y0 + margin
                max_x = x0 + model_input_shape[0] if x == nb_x - 1 else x0 + model_input_shape[0] - margin
                max_y = y0 + model_input_shape[1] if y == nb_y - 1 else y0 + model_input_shape[1] - margin
                predict_min_x = 0 if x == 0 else margin
                predict_min_y = 0 if y == 0 else margin
                predict_max_x = model_input_shape[0] if x == nb_x - 1 else model_input_shape[0] - margin
                predict_max_y = model_input_shape[1] if y == nb_y - 1 else model_input_shape[1] - margin

                if model_input_shape[2] == 1:
                    image[min_x:max_x, min_y:max_y, 0] = predicted[
                        predict_min_x:predict_max_x, predict_min_y:predict_max_y, 0
                    ]
                else:
                    image[min_x:max_x, min_y:max_y, :] = predicted[
                        predict_min_x:predict_max_x, predict_min_y:predict_max_y, :
                    ]

    if segmentation:
        file_name = os.path.splitext(image_dst_filename)

        # print(image.shape, np.min(image), np.max(image))
        # for i in range(model_output_shape[2]):
        #    min = np.min(image[:, :, i])
        #    max = np.max(image[:, :, i])
        #    print(np.min(image[:, :, i]), np.max(image[:, :, i]))
        #    mask = ((image[:, :, i] - min) / (max - min) * 255).astype(np.uint8)
        #    print(np.min(mask), np.max(mask))
        #    cv2.imwrite(f"{file_name[0]}-mask{i}{file_name[1]}", mask)

        # predictions = predictions[:, :, 1]
        # Apply the mask to the original image
        # predictions = tf.argmax(image, axis=-1)
        predictions = predictions[: original_image.shape[0], : original_image.shape[1], 1]
        # save the predictions as numpy array
        np.save(f"{file_name[0]}{file_name[1]}", predictions)

        max = 1280/4
        min = -max
        mask = predictions
        # Save the prediction as a mask
        cv2.imwrite(f"{file_name[0]}-mask{file_name[1]}", ((mask - min) / (max - min) * 255).astype(np.uint8))

        # mask = (predictions > 0).astype(np.uint8) * 255
        # blur the mask
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        #mask = cv2.GaussianBlur(mask, (3, 3), 0)
        cv2.imwrite(f"{file_name[0]}-mask2{file_name[1]}", ((mask - min) / (max - min) * 255).astype(np.uint8))
        # 10
        mask = (mask > 0).astype(np.uint8)

        # erode the mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        cv2.imwrite(f"{file_name[0]}-mask-tresh{file_name[1]}", mask * 255)

        image = original_image
        # save the original image
        cv2.imwrite(f"{file_name[0]}-original{file_name[1]}", image)
        image[mask == 1] = [255, 255, 255]
    else:
        if model_output_shape[2] == 1:
            # to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif yuv:
            # to BGR
            image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)

    # Save the image
    cv2.imwrite(image_dst_filename, image)
