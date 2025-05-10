import json
import math
import os
import time
from pathlib import Path

import wandb
from wandb.integration.keras import WandbMetricsLogger
import pandas as pd
import tensorflow as tf
import numpy as np

from .models.geo_model_net import make_geo_model_net

class LRLogger(tf.keras.callbacks.Callback):
    def get_current_learning_rate(self):
        if callable(self.model.optimizer.learning_rate):
            iters = self.model.optimizer.iterations
            return self.model.optimizer.learning_rate(iters)
        else:
            return float(self.model.optimizer.learning_rate)

    def on_batch_end(self, batch, logs):
        lr = self.get_current_learning_rate()

    def on_epoch_end(self, epoch, logs):
        lr = self.get_current_learning_rate()
        print("epoch {}, lr {}".format(epoch, lr))
        if lr == None:
            lr = 0.0
        wandb.log({"lr": lr}, commit=False)


def _lr_scheduler(epoch, lr):
    if epoch < 30:
        return lr
    else:
        if epoch % 5 == 0:
            return lr * tf.math.exp(-0.1)
        else:
            return lr


class DiscretizedInatGeoModelTrainer:
    def __init__(self, config):
        self.config = config

    def make_tfdata_dataset(
        self,
        tfrecord_file,
        num_classes,
        shuffle_buffer_size,
        batch_size,
        cache_file=None,
    ):
        # Create a dictionary describing the features.
        gp_grid_feature_description = {
            "l0": tf.io.FixedLenFeature([], tf.float32),
            "l1": tf.io.FixedLenFeature([], tf.float32),
            "l2": tf.io.FixedLenFeature([], tf.float32),
            "l3": tf.io.FixedLenFeature([], tf.float32),
            "elevation": tf.io.FixedLenFeature([], tf.float32),
            "leaf_class_ids": tf.io.VarLenFeature(tf.int64),
        }

        def grid_parse_function(example_proto):
            # Parse the input tf.train.Example proto using the dictionary above.
            return tf.io.parse_single_example(
                example_proto, gp_grid_feature_description
            )

        def preprocess_line(line):
            l0 = tf.expand_dims(line["l0"], axis=0)
            l1 = tf.expand_dims(line["l1"], axis=0)
            l2 = tf.expand_dims(line["l2"], axis=0)
            l3 = tf.expand_dims(line["l3"], axis=0)
            elevation = tf.expand_dims(line["elevation"], axis=0)
            encoded_loc = tf.concat([l0, l1, l2, l3, elevation], axis=0)
            leaf_class_ids = multi_hot(line["leaf_class_ids"])
            return encoded_loc, leaf_class_ids

        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
        # TODO: this is super slow
        num_examples = len(list(raw_dataset))

        ds = raw_dataset.map(grid_parse_function)
        multi_hot = tf.keras.layers.CategoryEncoding(
            num_tokens=num_classes, output_mode="multi_hot"
        )
        ds = ds.map(preprocess_line)
        if cache_file is not None:
            ds = ds.cache(filename=cache_file)
        ds = ds.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds, num_examples

    def _make_and_compile_model(self, learning_rate, num_classes, num_input_feats):

        fcnet = make_geo_model_net(
            num_classes=num_classes, num_input_feats=num_input_feats
        )
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        fcnet.compile(
            optimizer=optimizer,
            loss=bce,
            metrics=[
                "Precision",
                "Recall",
                tf.keras.metrics.AUC(curve="PR", name="prauc"),
                tf.keras.metrics.TruePositives(),
                tf.keras.metrics.FalseNegatives(),
            ],
        )

        return fcnet

    def train_geomodel(self):
        wandb.init(project=self.config["wandb_project"], config=self.config)

        if self.config["dataset_type"] == "sinr":
            sinr_train_data_dir = Path(self.config["dataset_dir"])
            tax = pd.read_json(sinr_train_data_dir / "geo_prior_train_meta.json")
            leaf_tax = tax
        elif self.config["dataset_type"] == "inat":
            train_data_dir = Path(self.config["dataset_dir"])
            tax = pd.read_csv(train_data_dir / "taxonomy.csv")
            leaf_tax = tax[~tax.leaf_class_id.isna()]

        num_leaf_taxa = len(leaf_tax)
        num_taxa = len(tax)
        if self.config["inner_nodes"]:
            num_classes = num_taxa
        else:
            num_classes = num_leaf_taxa

        EXPERIMENT_DIRNAME = "{}_{}_{}e_{}lr_elev_{}".format(
            self.config["export_short_version"],
            self.config["batch_size"],
            self.config["num_epochs"],
            str(self.config["initial_lr"]).replace(".", "_"),
            int(time.time()),
        )
        EXPERIMENT_DIR = os.path.join(self.config["experiment_dir"], EXPERIMENT_DIRNAME)
        os.makedirs(EXPERIMENT_DIR, exist_ok=True)

        TENSORBOARD_LOGDIR = os.path.join(EXPERIMENT_DIR, "logdir")
        MODEL_SAVE_FILE = os.path.join(EXPERIMENT_DIR, "saved_model.h5")
        CONFIG_FILE = os.path.join(EXPERIMENT_DIR, "config.json")

        with open(CONFIG_FILE, "w") as fp:
            json.dump(self.config, fp)

        print(f"  loading dataset from {self.config['tfrecord_file']}")
        dataset, num_examples = self.make_tfdata_dataset(
            tfrecord_file=self.config["tfrecord_file"],
            num_classes=num_classes,
            shuffle_buffer_size=self.config["shuffle_buffer_size"],
            batch_size=self.config["batch_size"],
        )

        print("{} examples".format(num_examples))
        print("{} batch size".format(self.config["batch_size"]))
        steps_per_epoch = int(np.ceil(num_examples / self.config["batch_size"]))
        total_steps = steps_per_epoch * self.config["num_epochs"]
        warmup_steps = total_steps * 0.1
        decay_steps = total_steps - warmup_steps
        print("{} total epochs".format(self.config["num_epochs"]))
        print("{} steps per epoch".format(steps_per_epoch))
        print("{} total steps".format(total_steps))
        print("{} warmup steps".format(warmup_steps))
        print("{} decay steps".format(decay_steps))

        if self.config.get("lr_warmup_cosine_decay"):
            learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-7,
                decay_steps=decay_steps,
                warmup_target=self.config["initial_lr"],
                warmup_steps=warmup_steps,
            )
        else:
            learning_rate = self.config["initial_lr"]

        fcnet = self._make_and_compile_model(
            learning_rate=learning_rate,
            num_classes=num_classes,
            num_input_feats=5
        )

        callbacks = [
            tf.keras.callbacks.TensorBoard(TENSORBOARD_LOGDIR),
            LRLogger(),
            WandbMetricsLogger(log_freq=100),
        ]

        if not self.config.get("lr_warmup_cosine_decay"):
            callbacks.append(
                tf.keras.callbacks.LearningRateScheduler(_lr_scheduler, verbose=1)
            )

        history = fcnet.fit(
            dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config["num_epochs"],
            callbacks=callbacks,
        )
        # TODO: archive this?
        print(history.history)

        fcnet.save(MODEL_SAVE_FILE)

        wandb.finish()
