import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Import the MoViNet model from TensorFlow Models (tf-models-official) for the MoViNet model
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model

from main import get_data
# import tensorflow as tf

validation_steps = 20

with tf.device('/CPU:0'):
    train_data = get_data(
        mode='train',
        extract_root='/content/drive/MyDrive/deep-learning-final-project/data-2024-03-25/extract/',
        data_root='/content/deep-learning-final-project/data',
        batch_size=8,
        frame_count=15,
    )
    test_data = get_data(
        mode='test',
        extract_root='/content/drive/MyDrive/deep-learning-final-project/data-2024-03-25/extract/',
        data_root='/content/deep-learning-final-project/data',
        batch_size=8,
        frame_count=15,
        validation_steps=validation_steps,
    )

model_id = 'a0'
resolution = 224

tf.keras.backend.clear_session()

backbone = movinet.Movinet(model_id='a0')
backbone.trainable = False

# Set num_classes=600 to load the pre-trained weights from the original model
model = movinet_model.MovinetClassifier(backbone=backbone, num_classes=600)
model.build([None, None, None, None, 3])

checkpoint_dir = f'movinet_{model_id}_base'
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
status = checkpoint.restore(checkpoint_path)
status.assert_existing_objects_matched()

count = 0
for layer in model.layers:
  layer.trainable = False
model.layers[-1].trainable = True


model = movinet_model.MovinetClassifier(
    backbone,
    num_classes=3)

inputs = tf.ones([8, 15, 224, 224, 3])

# [Optional] Build the model and load a pretrained checkpoint.
model.build(inputs.shape)

optimizer=Adam(learning_rate=1e-3)
# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer, #need to set learning rate and schedule but getting keras error
    metrics=['accuracy']
)

history = model.fit(
  train_data,
  validation_data=test_data,
  steps_per_epoch=300,#len(train_data) // batch_size,  analogous to window size? - tiff
  validation_steps=validation_steps,
  epochs=10, # change this
)