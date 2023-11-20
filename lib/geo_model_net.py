import tensorflow as tf


class ResLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ResLayer, self).__init__()
        self.w1 = tf.keras.layers.Dense(
            256, activation="relu", kernel_initializer="he_normal"
        )
        self.w2 = tf.keras.layers.Dense(
            256, activation="relu", kernel_initializer="he_normal"
        )
        self.dropout = tf.keras.layers.Dropout(rate=0.5)
        self.add = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.w1(inputs)
        x = self.dropout(x)
        x = self.w2(x)
        x = self.add([x, inputs])
        return x

    def get_config(self):
        return {}


def make_geo_model_net(num_classes):
    fcnet = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(
                5,
            ),
            # encode_location_layer,
            tf.keras.layers.Dense(
                256, activation="relu", kernel_initializer="he_normal"
            ),
            ResLayer(),
            ResLayer(),
            ResLayer(),
            ResLayer(),
            tf.keras.layers.Dense(num_classes, use_bias=False),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Activation("sigmoid", dtype="float32", name="predictions"),
        ]
    )
    return fcnet
