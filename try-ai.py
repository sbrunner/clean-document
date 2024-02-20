# Currently only build_autoencoder gives acceptable results
#
# TODO:
# - Fix image end in data augmentation?
# - Add more data augmentation (fliplr, flip)
# - Use bigger images (224 => 2000)
# - Try in color


from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import keras.saving
from keras_preprocessing.image import save_img
import glob

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os.path
import math
import sys

import keras.saving
import matplotlib.pyplot as plt
from keras_preprocessing.image import load_img, array_to_img, img_to_array

# from keras.popencv-pythonopencv-pythonopencv-pythonreprocessing.image import load_img, array_to_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras import backend as K
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Activation, BatchNormalization, Conv2D, LeakyReLU
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
import cv2
import glob
import numpy as np
from scipy import ndimage
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16


def build_model() -> Model:
    input_layer = Input(shape=(224, 224, 1))

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
    input_img = Input(shape=(224, 224, 1), name="image_input")

    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="Conv1")(input_img)
    x = MaxPooling2D((2, 2), padding="same", name="pool1")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv2")(x)
    x = MaxPooling2D((2, 2), padding="same", name="pool2")(x)

    # Decoder
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="Conv3")(x)
    x = UpSampling2D((2, 2), name="upsample1")(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="Conv4")(x)
    x = UpSampling2D((2, 2), name="upsample2")(x)
    x = Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="Conv5")(x)

    # Model
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return autoencoder


def build_autoencoder_my() -> Model:
    input_img = Input(shape=(224, 224, 1), name="image_input")

    # Encoder
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="Enc1")(input_img)
    x = MaxPooling2D((2, 2), padding="same", name="pool1")(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="Enc2")(x)
    x = MaxPooling2D((2, 2), padding="same", name="pool2")(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="Enc3")(x)
    x = MaxPooling2D((2, 2), padding="same", name="pool3")(x)

    # Decoder
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="Dec1")(x)
    x = UpSampling2D((2, 2), name="upsample1")(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="Dec2")(x)
    x = UpSampling2D((2, 2), name="upsample2")(x)
    x = Conv2D(16, (3, 3), activation="relu", padding="same", name="Dec3")(x)
    x = UpSampling2D((2, 2), name="upsample3")(x)
    x = Conv2D(1, (3, 3), activation="sigmoid", padding="same", name="D3c4")(x)

    # Model
    autoencoder = Model(inputs=input_img, outputs=x)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return autoencoder


def data_augmentation():
    for category in ("train-x", "train-y"):
        images = glob.glob(os.path.join("data", category, "*.png"))
        for image_filename in images:
            image = cv2.imread(image_filename)

            # split the image into regular 224x224 images
            nb_x = math.ceil(image.shape[0] / 224)
            nb_y = math.ceil(image.shape[1] / 224)
            slide_x = 224 - (224 * nb_x - image.shape[0]) / nb_x
            slide_y = 224 - (224 * nb_y - image.shape[1]) / nb_y
            for x in range(nb_x):
                for y in range(nb_y):
                    x0 = round(x * slide_x)
                    y0 = round(y * slide_y)
                    cropped = image[x0 : x0 + 224, y0 : y0 + 224]

                    dest_folder = os.path.join("augmented_data", category)
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)

                    cv2.imwrite(
                        os.path.join(
                            dest_folder,
                            f"cropped-{x}-{y}-{os.path.basename(image_filename)}",
                        ),
                        cropped,
                    )
                    cv2.imwrite(
                        os.path.join(
                            dest_folder,
                            f"rotated-{x}-{y}-{os.path.basename(image_filename)}",
                        ),
                        ndimage.rotate(cropped, 180),
                    )


def load_image(path):
    image_list = np.zeros((len(path), 224, 224, 1))
    for i, fig in enumerate(path):
        img = image.load_img(fig, color_mode="grayscale", target_size=(224, 224, 1))
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


def apply(model, name, path):
    print(path)
    sample_test = img_to_array(load_img(path))
    print(sample_test.shape)

    # Convert the image to yuv
    sample_test = cv2.cvtColor(sample_test, cv2.COLOR_BGR2YUV)
    print(sample_test.shape)
    # Get Y channel
    sample_test2 = sample_test[:, :, 0]
    print(sample_test2.shape)
    # sample_test2 = img_to_array(sample_test2)
    print(sample_test2.shape)

    sample_test_img = sample_test.astype("float32") / 255.0
    sample_test_img = np.expand_dims(sample_test, axis=0)
    print(sample_test_img.shape)

    # Get the prediction
    predicted_label = np.squeeze(model.predict(sample_test_img))
    sample_test[:, :, 0] = predicted_label
    data = cv2.cvtColor(sample_test, cv2.COLOR_YUV2BGR)
    save_img(f"{name}-{os.path.basename(path)}", data)

    data = np.zeros(
        (predicted_label.shape[0], predicted_label.shape[1], 1), dtype=np.uint8
    )
    data[:, :, 0] = predicted_label
    save_img(f"{name}-gray-{os.path.basename(path)}", data)


def apply2(model, name, path):
    print(path)
    sample_test = load_img(path, color_mode="grayscale", target_size=(224, 224))
    sample_test = load_img(path, color_mode="grayscale")
    # sample_test = load_img(path, target_size=(224, 224))
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


def train():
    TRAIN_IMAGES = glob.glob("augmented_data/train-x/*.png")
    CLEAN_IMAGES = glob.glob("augmented_data/train-y/*.png")

    x_train = load_image(TRAIN_IMAGES)
    y_train = load_image(CLEAN_IMAGES)
    print(x_train.shape, y_train.shape)

    x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    for name, model in (
        ("model", build_model()),
        # ("autoencoder", build_autoencoder()),
        #("autoencoder_my", build_autoencoder_my()),
    ):
        model.summary()
        try:
            # train_model(
            #    name, model, x_train, y_train, x_val, y_val, epochs=20, batch_size=20
            # )
            # Save the model
            # model.save(f"{name}.keras", overwrite=True)
            # Load model
            model = keras.saving.load_model(f"{name}.keras")
            model.summary()
            images = glob.glob("data/train-x/*.png")
            for image_path in images:
                apply2(model, name, image_path)
        except Exception as e:
            print(f"Failed to train model {name}: {e}")
            import traceback

            print(traceback.format_exc())
            raise e


train()
