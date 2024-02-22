import math

import cv2
import numpy as np


def apply(model, image_src_filename, image_dst_filename, rgb=False, margin=5):
    model_input_shape = model.input_shape[1:]

    image = cv2.imread(image_src_filename)
    if not rgb or model_input_shape[2] == 1:
        # to yuv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    if model_input_shape[0] is None and model_input_shape[1] is None:
        sample_test_img = np.expand_dims(image, axis=0)
        # Get the prediction
        predicted_label = np.squeeze(model.predict(sample_test_img.astype("float32") / 255.0))
        if model_input_shape[2] == 1:
            image[:, :, 0] = predicted_label * 255.0
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

        samples = np.zeros((len(images), model_input_shape[0], model_input_shape[1], 1))
        for nb, img in enumerate(images):
            samples[nb] = img

        predictions = model.predict(samples) * 255.0
        index = 0

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
    if not rgb or model_input_shape[2] == 1:
        # to BGR
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    # Save the image
    cv2.imwrite(image_dst_filename, image)
