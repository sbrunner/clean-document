import math

import cv2
import numpy as np
from keras_preprocessing.image import img_to_array, load_img, save_img


def apply_simple(model, image_src_filename, image_dst_filename):
    print(image_src_filename)
    sample_test = load_img(image_src_filename, color_mode="grayscale")
    sample_test = img_to_array(sample_test)
    print(sample_test.shape)
    print(np.min(sample_test), np.max(sample_test))
    sample_test_img = sample_test.astype("float32") / 255.0
    sample_test_img = np.expand_dims(sample_test, axis=0)
    print(sample_test_img.shape)

    # Get the prediction
    predicted_label = np.squeeze(model.predict(sample_test_img))
    print(np.min(predicted_label), np.max(predicted_label))
    print(predicted_label.shape)
    data = np.zeros((predicted_label.shape[0], predicted_label.shape[1], 1), dtype=np.uint8)
    data[:, :, 0] = predicted_label
    save_img(image_dst_filename, data)


def apply(model, image_src_filename, image_dst_filename, image_dimension, margin=5):
    print(image_src_filename)

    image = cv2.imread(image_src_filename)
    # to yuv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # img = load_img(        path    )
    # image = img_to_array(img).astype("float32")

    images = []

    # split the image into regular image_dimensionximage_dimension images
    nb_x = math.ceil(image.shape[0] / (image_dimension - 2 * margin))
    nb_y = math.ceil(image.shape[1] / (image_dimension - 2 * margin))
    slide_x = (image.shape[0] - image_dimension) / (nb_x - 1)
    slide_y = (image.shape[1] - image_dimension) / (nb_y - 1)
    for x in range(nb_x):
        for y in range(nb_y):
            x0 = round(x * slide_x)
            y0 = round(y * slide_y)
            cropped = image[x0 : x0 + image_dimension, y0 : y0 + image_dimension, :]

            sample_test_img = cropped.astype("float32") / 255.0
            images.append(sample_test_img)

    samples = np.zeros((len(images), image_dimension, image_dimension, 3))
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
            print(predicted.shape)
            index += 1

            # Update image with the prediction
            min_x = 0 if x == 0 else x0 + margin
            min_y = 0 if y == 0 else y0 + margin
            max_x = x0 + image_dimension if x == nb_x - 1 else x0 + image_dimension - margin
            max_y = y0 + image_dimension if y == nb_y - 1 else y0 + image_dimension - margin
            predict_min_x = 0 if x == 0 else margin
            predict_min_y = 0 if y == 0 else margin
            predict_max_x = image_dimension if x == nb_x - 1 else image_dimension - margin
            predict_max_y = image_dimension if y == nb_y - 1 else image_dimension - margin
            print(min_x, max_x, min_y, max_y)
            print(predict_min_x, predict_max_x, predict_min_y, predict_max_y)
            print()

            image[min_x:max_x, min_y:max_y, :] = predicted[
                predict_min_x:predict_max_x, predict_min_y:predict_max_y, :
            ]
            # save_img(f"{name}-{x}-{y}-gray-{os.path.basename(path)}", data)
    # to BGR
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    # Save the image
    cv2.imwrite(image_dst_filename, image)
