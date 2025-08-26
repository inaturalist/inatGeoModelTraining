import math
import os

import click
import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd

from lib.models.geo_model_net import ResLayer

@click.command()
@click.option("--keras_model", type=str, required=True)
@click.option("--output", type=str, required=True)
def main(keras_model, output):
    keras_model = tf.keras.models.load_model(
        keras_model,
        custom_objects={"ResLayer": ResLayer},
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()
    with open(output, "wb") as f:
        f.write(quantized_model)


if __name__ == "__main__":
    main()
