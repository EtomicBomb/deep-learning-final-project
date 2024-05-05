#!/usr/bin/env python

import time
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
from argparse import ArgumentParser
import time
from tensorflow.data import Dataset
from typing import Literal

from dataset import get_dataset, get_index
from augment import VideoRandomPerspective, VideoRandomFlip, VideoRandomContrast, VideoRandomMultiply, VideoRandomAdd, VideoRandomNoise, VideoCropAndResize, ClipZeroOne, Scale, Gray2RGB
from dimensions import Dimensions

def get_data(
    mode: Literal['train', 'test'],
    extract_root: str, 
    data_root: str, 
    batch_size: int,
    frame_count: int,
):
    src_shape = Dimensions(
        batch_size=batch_size,
        frame_count=frame_count,
        height=112,
        width=224,
        channels=1,
    )
    data = Dataset.load(str(Path(data_root, f'{mode}{frame_count}.dataset')))
    data = data.shuffle(data.cardinality(), reshuffle_each_iteration=True)
    if mode == 'train':
        data = data.repeat()
    data = data.map(lambda pts, path, label: (pts, tf.strings.join([extract_root, path]), label))
    data = get_dataset(data, s=src_shape)

    model = keras.Sequential([
        keras.Input(shape=src_shape.example_shape, batch_size=src_shape.batch_size),
        VideoCropAndResize(),
        Scale(),
        Gray2RGB(),
    ])

    if mode == 'train':
        rng = tf.random.Generator.from_non_deterministic_state()
        model = keras.Sequential([
            keras.Input(shape=src_shape.example_shape, batch_size=src_shape.batch_size),
            VideoRandomNoise(rng=rng), 
            VideoRandomPerspective(rng=rng),
            VideoRandomFlip(rng=rng),
            VideoRandomContrast(rng=rng),
            VideoRandomMultiply(rng=rng),
            VideoRandomAdd(rng=rng),
            ClipZeroOne(),
            model,
        ])
    data = data.map(
        lambda x, y: (model(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    data = data.prefetch(tf.data.AUTOTUNE)
    return data

def train_test(extract_root: str, data_root: str, batch_size=2, frame_count=32):
    train = get_data(
        mode='train',
        extract_root=extract_root,
        data_root=data_root,
        batch_size=batch_size,
        frame_count=frame_count,
    )
    test = get_data(
        mode='test',
        extract_root=extract_root,
        data_root=data_root,
        batch_size=batch_size,
        frame_count=frame_count,
    )

    return train, test

def index():
    frame_count = 32
    data_split = json.loads(Path('data/split.json').read_text())
    data = get_index('data/extract', paths=data_split['train'], frame_count=32)
    Dataset.save(data, f'data/train{frame_count}.dataset')
    data = get_index('data/extract', paths=data_split['test'], frame_count=32)
    Dataset.save(data, f'data/test{frame_count}.dataset')

def demo():
    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt
    data = get_data(
        mode='train',
        extract_root='data/extract/',
        data_root='data',
        batch_size=3,
        frame_count=32,
    )

    fig, ax = plt.subplots()
    print(f"data: {data}")
    data = data.as_numpy_iterator()
    
    data = iter((frame, label) for batch, labels in data for video, label in zip(batch, labels) for frame in video)
    image = ax.imshow(next(data)[0], cmap='gray')
    def animate(data):
        x, y = data
        image.set_data(x)
        print(y)
        return [image]
    ani = FuncAnimation(fig, animate, data, cache_frame_data=False, blit=True, interval=1)
    plt.show()

class VideoMobileNet(keras.layers.Layer):
    def __init__(self, *args, start=None, end=None, trainable=True, **kwargs):
        super().__init__(*args, **kwargs)
        model = keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))
        if start is not None:
            start = model.get_layer(start)
        else:
            start = model.layers[0]
        end = model.get_layer(end)
        model = keras.Model(inputs=start.output,outputs=end.output)
        for layer in model.layers:
           layer.trainable = trainable
        self.model = model
    def call(self, x):
        batch_size, frame_count, height, width, channels = x.shape
        x = tf.reshape(x, (batch_size*frame_count, height, width, channels))
        x = self.model(x)
        _, height, width, channels = x.shape
        x = tf.reshape(x, (batch_size, frame_count, height, width, channels))
        return x

class Video1DConvolution(keras.layers.Layer):
    def __init__(self, filters, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = keras.layers.Conv1D(filters, kernel_size, padding='same', data_format='channels_first')
    def call(self, x):
        batch_size, frame_count, height, width, channels = x.shape
        x = tf.reshape(x, (batch_size, frame_count, height * width * channels), name='bar2')
        x = self.m(x)
        x = tf.reshape(x, (batch_size, x.shape[1], height, width, channels), name='foo2')
        return x

def experiment():
    # https://keras.io/api/applications/
    s = Dimensions(
        batch_size=4,
        frame_count=32,
        height=224,
        width=224,
        channels=3,
    )

    model = keras.Sequential([
        keras.Input(shape=s.example_shape, batch_size=s.batch_size),
        tf.keras.layers.Rescaling(2.0, -1.0), # [0,1] -> [-1, 1]
        VideoMobileNet(start=None,end='block_3_depthwise', trainable=False),
        Video1DConvolution(32, 20),
        VideoMobileNet(start='block_3_depthwise',end='block_6_depthwise', trainable=False),
        Video1DConvolution(32, 20),
        VideoMobileNet(start='block_6_depthwise',end='block_13_depthwise', trainable=False),
        Video1DConvolution(32, 20),
        VideoMobileNet(start='block_13_depthwise',end='block_16_expand', trainable=True),
        Video1DConvolution(32, 20),
        VideoMobileNet(start='block_16_expand',end='out_relu', trainable=True),
        Video1DConvolution(32, 20),
        keras.layers.Conv3D(1, (10, 3, 3), strides=(3, 3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(3, activation='sigmoid'),
    ])

    train_data = get_data(
        mode='train',
        extract_root='data/extract/',
        data_root='data',
        batch_size=3,
        frame_count=32,
    )

    test_data = get_data(
        mode='test',
        extract_root='data/extract/',
        data_root='data',
        batch_size=3,
        frame_count=32,
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=f'data/checkpoints/{timestamp}{{epoch}}.keras',
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
          learning_rate=1e-7
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    history = model.fit(
        train_data,
        steps_per_epoch=300,
        epochs=10, 
        callbacks=[model_checkpoint_callback],
    )
    print(history)

    model.summary()

if __name__ == '__main__':
    modes = dict(demo=demo, index=index, experiment=experiment)
    parser = ArgumentParser(
        prog='drowsiness classifier',
        description='sees if someone is drowsy',
        epilog='text at the bottom of help')
    parser.add_argument('-m', '--mode', choices=list(modes), default=next(iter(modes)))
    args = parser.parse_args()
    modes[args.mode]()
