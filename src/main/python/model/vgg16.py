from random import sample

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.python.keras.layers import (
    Conv2D,
    Activation,
    MaxPooling2D,
    Flatten,
    Dense,
    LeakyReLU,
    Dropout,
    Reshape,
    Lambda,
)
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.regularizers import l2

from common.constants import NORM_H, NORM_W, BIN, SPLIT, BATCH_SIZE
from common.loss import orientation_loss
from processing.data_generation import data_gen
from processing.data_loader import DataLoader
from processing.preprocessing import orientation_confidence_flip


class VGG16(object):
    def __init__(self):
        self.model = VGG16.build_model()
        self.model.summary()

    @staticmethod
    def build_model():
        inputs = Input(shape=(NORM_H, NORM_W, 3))
        # Block 1__
        x = Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block1_conv1",
        )(inputs)
        x = Activation("relu")(x)
        x = Conv2D(
            64,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block1_conv2",
        )(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(strides=(2, 2), name="block1_pool")(x)
        # Block 2
        x = Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block2_conv1",
        )(x)
        x = Activation("relu")(x)
        x = Conv2D(
            128,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block2_conv2",
        )(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(strides=(2, 2), name="block2_pool")(x)
        # Block 3
        x = Conv2D(
            256,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block3_conv1",
        )(x)
        x = Activation("relu")(x)
        x = Conv2D(
            256,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block3_conv2",
        )(x)
        x = Activation("relu")(x)
        x = Conv2D(
            256,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block3_conv3",
        )(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(strides=(2, 2), name="block3_pool")(x)
        # Block 4
        x = Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block4_conv1",
        )(x)
        x = Activation("relu")(x)
        x = Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block4_conv2",
        )(x)
        x = Activation("relu")(x)
        x = Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block4_conv3",
        )(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(strides=(2, 2), name="block4_pool")(x)
        # Block 5
        x = Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block5_conv1",
        )(x)
        x = Activation("relu")(x)
        x = Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block5_conv2",
        )(x)
        x = Activation("relu")(x)
        x = Conv2D(
            512,
            (3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1e-4),
            name="block5_conv3",
        )(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(strides=(2, 2), name="block5_pool")(x)
        # Flatten
        x = Flatten(name="Flatten")(x)
        # Dimensions branch
        dimensions = Dense(512, name="d_fc_1")(x)
        dimensions = LeakyReLU(alpha=0.1)(dimensions)
        dimensions = Dropout(0.5)(dimensions)
        dimensions = Dense(3, name="d_fc_2")(dimensions)
        dimensions = LeakyReLU(alpha=0.1, name="dimensions")(dimensions)
        # Orientation branch
        orientation = Dense(256, name="o_fc_1")(x)
        orientation = LeakyReLU(alpha=0.1)(orientation)
        orientation = Dropout(0.5)(orientation)
        orientation = Dense(BIN * 2, name="o_fc_2")(orientation)
        orientation = LeakyReLU(alpha=0.1)(orientation)
        orientation = Reshape((BIN, -1))(orientation)
        orientation = Lambda(VGG16.l2_normalize, name="orientation")(orientation)
        # Confidence branch
        confidence = Dense(256, name="c_fc_1")(x)
        confidence = LeakyReLU(alpha=0.1)(confidence)
        confidence = Dropout(0.5)(confidence)
        confidence = Dense(BIN, activation="softmax", name="confidence")(confidence)
        # Build model
        return tf.keras.Model(inputs, [dimensions, orientation, confidence])

    def train(self, filename_weights: str, debug: bool = False):
        data_loader = DataLoader(
            input_path="/media/wittmaan/Elements/data/a2d2/camera_lidar_semantic_bboxes",
            cam_config="/media/wittmaan/Elements/data/a2d2/cams_lidars.json",
            data_type="train",
        )
        dim_avg, dim_cnt = data_loader.get_average_dimension()

        new_data = orientation_confidence_flip(data_loader.train_labels, dim_avg)
        if debug:
            new_data = sample(new_data, 5000)

        early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, mode="min", verbose=1)
        checkpoint = ModelCheckpoint(
            filename_weights, monitor="val_loss", verbose=1, save_best_only=True, mode="min", save_freq=50
        )
        tensorboard = TensorBoard(log_dir="logs/", histogram_freq=0, write_graph=True, write_images=False)

        all_examples = len(new_data)
        trv_split = int(SPLIT * all_examples)  # train val split

        train_gen = data_gen(new_data[:trv_split])
        valid_gen = data_gen(new_data[trv_split:all_examples])

        train_num = int(np.ceil(trv_split / BATCH_SIZE))
        valid_num = int(np.ceil((all_examples - trv_split) / BATCH_SIZE))

        # multi task learning
        self.model.compile(
            optimizer=Adam(),
            loss={
                "dimensions": "mean_squared_error",
                "orientation": orientation_loss,
                "confidence": "binary_crossentropy",
            },
            loss_weights={"dimensions": 1.0, "orientation": 10.0, "confidence": 5.0},
        )

        self.model.fit(
            train_gen,
            steps_per_epoch=train_num,
            epochs=500,
            verbose=1,
            validation_data=valid_gen,
            validation_steps=valid_num,
            shuffle=True,
            callbacks=[early_stop, checkpoint, tensorboard],
            max_queue_size=3,
        )

    @staticmethod
    def l2_normalize(x):
        return tf.nn.l2_normalize(x, axis=2)

    def load_weights(self, weights: str):
        self.model.load_weights(weights)
