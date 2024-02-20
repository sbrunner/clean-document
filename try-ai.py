# Currently only build_autoencoder gives acceptable results
# No good results with bigger images
#
# TODO:
# - Try in color


import random
from keras.models import Model
import keras.saving
from keras_preprocessing.image import save_img
import glob
import shutil
import numpy as np

import os.path
import math

import keras.saving
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, img_to_array

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.optimizers import Adam
from keras import backend as K

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, BatchNormalization, Conv2D, LeakyReLU
from keras.layers import MaxPooling2D, UpSampling2D
import cv2
import glob
import numpy as np
from scipy import ndimage
from keras.preprocessing import image

IMAGE_DIMENSION = 256


def build_model() -> Model:
    input_layer = Input(shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3))

    # encoder
    h = Conv2D(64, (3, 3), padding="same")(input_layer)
    h = LeakyReLU(0.1)(h)
    h = BatchNormalization()(h)
    h = Conv2D(64, (3, 3), padding="same")(h)
    h = LeakyReLU(0.1)(h)
    h = MaxPooling2D((2, 2), padding="same")(h)

    h = Conv2D(64, (3, 3), padding="same")(h)
    h = LeakyReLU(0.1)(h)
    h = BatchNormalization()(h)
    # decoder
    h = Conv2D(64, (3, 3), padding="same")(h)
    h = LeakyReLU(0.1)(h)
    h = UpSampling2D((2, 2))(h)
    output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(h)

    m = Model(input_layer, output_layer)
    optimizer = Adam(learning_rate=0.001)
    m.compile(loss="mse", optimizer=optimizer)
    return m


def build_autoencoder() -> Model:
    input_img = Input(shape=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3), name="image_input")

    # Encoder
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="ConvEnc1")(
        input_img
    )
    x = MaxPooling2D((2, 2), padding="same", name="pool1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="ConvEnc2")(x)
    x = MaxPooling2D((2, 2), padding="same", name="pool2")(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="ConvDec1")(x)
    x = UpSampling2D((2, 2), name="upsample1")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="ConvDec2")(x)
    x = UpSampling2D((2, 2), name="upsample2")(x)
    x = Conv2D(3, (3, 3), activation="sigmoid", padding="same", name="ConvDec3")(x)

    # Model
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return autoencoder


def data_augmentation_clean():
    shutil.rmtree("augmented_data", ignore_errors=True)

    images = glob.glob(os.path.join("clean-data/*.png"))
    for image_filename in images:
        data_augmentation_image(image_filename, "train-xy")


def data_augmentation(xy=False):
    shutil.rmtree("augmented_data", ignore_errors=True)

    for category in ("train-xy",) if xy else ("train-x", "train-y"):
        images = (
            glob.glob("clean-sata/*.png")
            if xy
            else glob.glob(os.path.join("data", category, "sbr*.png"))
        )
        for image_filename in images:
            data_augmentation_image(image_filename, category)


def data_augmentation_image(image_filename, category=None, save=True):
    results = []
    image = cv2.imread(image_filename)
    # to yuv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # split the image into regular IMAGE_DIMENSIONxIMAGE_DIMENSION images
    nb_x = math.ceil(image.shape[0] / IMAGE_DIMENSION)
    nb_y = math.ceil(image.shape[1] / IMAGE_DIMENSION)
    slide_x = (image.shape[0] - IMAGE_DIMENSION) / (nb_x - 1)
    slide_y = (image.shape[1] - IMAGE_DIMENSION) / (nb_y - 1)
    for x in range(nb_x):
        for y in range(nb_y):
            x0 = round(x * slide_x)
            y0 = round(y * slide_y)
            if x0 + IMAGE_DIMENSION > image.shape[0]:
                print(nb_x, x, slide_x)
                print(image.shape, x0 + IMAGE_DIMENSION, y0 + IMAGE_DIMENSION)
            if y0 + IMAGE_DIMENSION > image.shape[1]:
                print(nb_y, y, slide_y)
                print(image.shape, x0 + IMAGE_DIMENSION, y0 + IMAGE_DIMENSION)
            cropped = image[x0 : x0 + IMAGE_DIMENSION, y0 : y0 + IMAGE_DIMENSION]

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

    nb_x = math.ceil(image.shape[0] / IMAGE_DIMENSION)
    nb_y = math.ceil(image.shape[1] / IMAGE_DIMENSION)
    slide_x = (image.shape[0] - IMAGE_DIMENSION) / (nb_x - 1)
    slide_y = (image.shape[1] - IMAGE_DIMENSION) / (nb_y - 1)
    for x in range(nb_x):
        for y in range(nb_y):
            x0 = round(x * slide_x)
            y0 = round(y * slide_y)
            if x0 + IMAGE_DIMENSION > image.shape[0]:
                print(nb_x, x, slide_x)
                print(image.shape, x0 + IMAGE_DIMENSION, y0 + IMAGE_DIMENSION)
            if y0 + IMAGE_DIMENSION > image.shape[1]:
                print(nb_y, y, slide_y)
                print(image.shape, x0 + IMAGE_DIMENSION, y0 + IMAGE_DIMENSION)
            cropped = image[x0 : x0 + IMAGE_DIMENSION, y0 : y0 + IMAGE_DIMENSION]

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


def load_image(path):
    image_list = np.zeros((len(path), IMAGE_DIMENSION, IMAGE_DIMENSION, 3))
    for i, fig in enumerate(path):
        img = image.load_img(
            fig,
            target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION, 3),
        )
        x = image.img_to_array(img).astype("float32")
        image_list[i] = x

    return image_list


def train_val_split(x_train, y_train):
    x_train = x_train / 255.0
    y_train = y_train / 255.0
    rnd = np.random.RandomState(seed=42)
    perm = rnd.permutation(len(x_train))
    train_idx = perm[: int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)) :]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]


def train_model(name, model, x_train, y_train, x_val, y_val, epochs, batch_size=20):
    # early_stopping = EarlyStopping(monitor='val_loss',
    #                                min_delta=0,
    #                                patience=5,
    #                                verbose=1,
    #                                mode='auto')
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
    )
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    # save plot to file
    plt.savefig(f"{name}.png")


def apply_simple(model, name, path):
    print(path)
    sample_test = load_img(
        path, color_mode="grayscale", target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION)
    )
    sample_test = load_img(path, color_mode="grayscale")
    # sample_test = load_img(path, target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))
    sample_test = img_to_array(sample_test)
    print(sample_test.shape)
    print(np.min(sample_test), np.max(sample_test))
    sample_test_img = sample_test.astype("float32") / 255.0
    sample_test_img = np.expand_dims(sample_test, axis=0)
    print(sample_test_img.shape)

    # Get the prediction
    predicted_label = np.squeeze(model.predict(sample_test_img))
    print("gggg")
    print(name)
    print(np.min(predicted_label), np.max(predicted_label))
    print(predicted_label.shape)
    data = np.zeros(
        (predicted_label.shape[0], predicted_label.shape[1], 1), dtype=np.uint8
    )
    data[:, :, 0] = predicted_label
    save_img(f"{name}-gray-{os.path.basename(path)}", data)


def apply(model, name, path):
    print(path)
    # sample_test = load_img(path, color_mode="grayscale", target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))
    # sample_test = load_img(path, color_mode="grayscale")
    # sample_test = load_img(path, target_size=(IMAGE_DIMENSION, IMAGE_DIMENSION))
    # sample_test = img_to_array(sample_test)
    margin = 5
    margin = 5

    image = cv2.imread(path)
    # to yuv
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # img = load_img(        path    )
    # image = img_to_array(img).astype("float32")

    images = []

    # split the image into regular IMAGE_DIMENSIONxIMAGE_DIMENSION images
    nb_x = math.ceil(image.shape[0] / (IMAGE_DIMENSION - 2 * margin))
    nb_y = math.ceil(image.shape[1] / (IMAGE_DIMENSION - 2 * margin))
    slide_x = (image.shape[0] - IMAGE_DIMENSION) / (nb_x - 1)
    slide_y = (image.shape[1] - IMAGE_DIMENSION) / (nb_y - 1)
    for x in range(nb_x):
        for y in range(nb_y):
            x0 = round(x * slide_x)
            y0 = round(y * slide_y)
            cropped = image[x0 : x0 + IMAGE_DIMENSION, y0 : y0 + IMAGE_DIMENSION, :]

            sample_test_img = cropped.astype("float32") / 255.0
            images.append(sample_test_img)

    samples = np.zeros((len(images), IMAGE_DIMENSION, IMAGE_DIMENSION, 3))
    for nb, img in enumerate(images):
        samples[nb] = img

    predictions = model.predict(samples)                * 255.0
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
            max_x = (
                x0 + IMAGE_DIMENSION if x == nb_x - 1 else x0 + IMAGE_DIMENSION - margin
            )
            max_y = (
                y0 + IMAGE_DIMENSION if y == nb_y - 1 else y0 + IMAGE_DIMENSION - margin
            )
            predict_min_x = 0 if x == 0 else margin
            predict_min_y = 0 if y == 0 else margin
            predict_max_x = (
                IMAGE_DIMENSION if x == nb_x - 1 else IMAGE_DIMENSION - margin
            )
            predict_max_y = (
                IMAGE_DIMENSION if y == nb_y - 1 else IMAGE_DIMENSION - margin
            )
            print(min_x,max_x, min_y,max_y)
            print(predict_min_x,predict_max_x, predict_min_y,predict_max_y)
            print()

            image[min_x:max_x, min_y:max_y, :] = (
                predicted[
                    predict_min_x:predict_max_x, predict_min_y:predict_max_y, :
                ]
            )
            data = predicted
            # save_img(f"{name}-{x}-{y}-gray-{os.path.basename(path)}", data)
    # to BGR
    image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
    # Save the image
    cv2.imwrite(f"{name}-{os.path.basename(path)}", image)
    # save_img(f"{name}-{os.path.basename(path)}", image)


MODELS = ()


def train():
    TRAIN_IMAGES = glob.glob("augmented_data/train-x/*.png")
    CLEAN_IMAGES = glob.glob("augmented_data/train-y/*.png")

    x_train = load_image(TRAIN_IMAGES)
    y_train = load_image(CLEAN_IMAGES)
    print(x_train.shape, y_train.shape)

    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    for name, model in MODELS:
        model.summary()
        try:
            train_model(
                name, model, x_train, y_train, x_val, y_val, epochs=20, batch_size=20
            )
            # Save the model
            model.save(f"{name}.keras", overwrite=True)
        except Exception as e:
            print(f"Failed to train model {name}: {e}")
            raise e


import gc


def train_many():
    for name, model in MODELS:
        model.summary()
        if os.path.exists(f"{name}.keras"):
            model = keras.saving.load_model(f"{name}.keras")
        for epoch in range(20):
            filenames = [random.choice(list(glob.glob("clean-data/*.png")))]
            for nb, filename in enumerate(filenames):
                print(f"Epoch {epoch} {filename} {nb}/{len(filenames)}")
                images = data_augmentation_image(filename, save=False)

                image_list = np.zeros(
                    (len(images), IMAGE_DIMENSION, IMAGE_DIMENSION, 3)
                )

                for i, image in enumerate(images):
                    image_list[i] = image.astype("float32") / 255.0

                try:
                    model.fit(image_list, image_list, batch_size=20)
                    # Save the model
                    model.save(f"{name}.keras", overwrite=True)
                    return
                except Exception as e:
                    print(f"Failed to train model {name}: {e}")
                    raise e


def do_apply():
    for name, model in MODELS:
        try:
            # Load model
            model = keras.saving.load_model(f"{name}.keras")
            model.summary()
            images = glob.glob("data/train-x/*.png")
            for image_path in images:
                apply(model, name, image_path)
        except Exception as e:
            print(f"Failed to apply model {name}: {e}")
            import traceback

            print(traceback.format_exc())
            raise e


import pikepdf


def get_original_pdf():
    for path in glob.glob(
        "/home/sbrunner/Documents Stéphane/originals/**/*.pdf", recursive=True
    ):
        max_image_size = 0
        with pikepdf.open(path) as pdf:
            for page in pdf.pages:
                for image in page.images.values():
                    pdfimage = pikepdf.PdfImage(image)
                    max_image_size = max(
                        max_image_size, pdfimage.width * pdfimage.height
                    )

        if max_image_size < 100000:
            print(path)
            # move file to clean-dat if he didn't already exists
            dest = os.path.join("clean-data", os.path.basename(path))
            if not os.path.exists(dest):
                shutil.move(path, dest)
            else:
                print(f"{dest} already exists")


import subprocess


def convert_original_pdf_to_png():
    for path in glob.glob("clean-data/*.pdf"):
        print(path)
        # convert pdf to png in 300 dpi
        subprocess.run(
            ["convert", "-density", "300", path, f"{path[:-4]}.png"], check=True
        )


MODELS = (
    # ("model", build_model()),
    ("autoencoder", build_autoencoder()),
    # ("autoencoder_my", build_autoencoder_my()),
)

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
