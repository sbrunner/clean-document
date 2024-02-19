from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Activation, BatchNormalization, Conv2D, LeakyReLU
from keras.layers import MaxPooling2D, Dropout, UpSampling2D

from keras.applications.vgg16 import VGG16

def build_model2() -> Model:
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(420, 540, 3))

    # Freeze the layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add your own layers
    x = base_model.output
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    x = Conv2D(64, (3, 3), padding="same")(x)
    x = LeakyReLU(0.1)(x)
    x = BatchNormalization()(x)
    # decoder
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2, 2))(x)
    output_layer = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    m = Model(base_model.input, output_layer)
    optimizer = Adam(lr=0.001)
    m.compile(loss="mse", optimizer=optimizer)
    return m

def build_model() -> Model:
    input_layer = Input(shape=(420, 540, 3))

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
    optimizer = Adam(lr=0.001)
    m.compile(loss="mse", optimizer=optimizer)
    return m


def build_autoencoder2() -> Model:
    input_img = Input(shape=(420, 540, 3), name="image_input")

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(420, 540, 3))

    # Freeze the layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add your own layers
    x = base_model.output

    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same", name="Conv1")(x)
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


def build_autoencoder() -> Model:
    input_img = Input(shape=(420, 540, 3), name="image_input")

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

def build_model_imagenet() -> Model:

    # Load the VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # Add a logistic layer with the number of classes you have (let's say 10)
    predictions = Dense(10, activation='softmax')(x)

    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # First: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    return model

# Total params: 113665 (444.00 KB)
# Trainable params: 113409 (443.00 KB)
# Non-trainable params: 256 (1.00 KB)
m = build_model()
m.summary()

# Total params: 15121537 (57.68 MB)
# Trainable params: 406593 (1.55 MB)
# Non-trainable params: 14714944 (56.13 MB)
m = build_model2()
m.summary()

# Total params: 15250250 (58.18 MB)
# Trainable params: 535562 (2.04 MB)
# Non-trainable params: 14714688 (56.13 MB)
m = build_model_imagenet()
m.summary()

# Total params: 75073 (293.25 KB)
# Trainable params: 75073 (293.25 KB)
# Non-trainable params: 0 (0.00 Byte)
m = build_autoencoder()
m.summary()

#m = build_autoencoder2()
#m.summary()
