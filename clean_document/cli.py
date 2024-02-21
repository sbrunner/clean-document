import argparse

import keras.saving

from clean_document import apply


def do_apply(image_src_filename, image_dst_filename, image_dimension, model_filename):
    # Load model
    model = keras.saving.load_model(model_filename)
    model.summary()
    apply.apply(model, image_src_filename, image_dst_filename, image_dimension)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model_64_128-finalized.keras")
    parser.add_argument("--dimension", default=255, type=int)
    parser.add_argument("image_src", type=str)
    parser.add_argument("image_dst", type=str)

    args = parser.parse_args()

    do_apply(args.image_src, args.image_dst, args.dimension, args.model)
