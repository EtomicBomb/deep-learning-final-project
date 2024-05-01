#!/usr/bin/env python

from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dataclasses import dataclass

from dataset import get_dataset
from augment import VideoRandomPerspective, VideoRandomFlip, VideoRandomContrast, VideoRandomMultiply, VideoRandomAdd, VideoRandomNoise, VideoCropAndResize, ClipZeroOne

batch_size = 2
frames_per_example = 30 * 3
video_height = 112
video_width = 224
channels = 1
num_classes = 3
rng = tf.random.Generator.from_non_deterministic_state()

augment_model = keras.Sequential([
    keras.Input(shape=(frames_per_example, video_height, video_width, channels), batch_size=batch_size),
    VideoCropAndResize(),
    VideoRandomNoise(rng=rng),
    VideoRandomPerspective(rng=rng),
    VideoRandomFlip(rng=rng),
    VideoRandomContrast(rng=rng),
    VideoRandomMultiply(rng=rng),
    VideoRandomAdd(rng=rng),
    ClipZeroOne(),
])

data = json.loads(Path('data/split.json').read_text())
data = data['train'] # select only one participant so we can see changes easily
data = get_dataset(
    data_root='data',
    paths=data,
    shuffle_batch=1000, 
    frames_per_example=frames_per_example, 
    video_height=video_height, 
    video_width=video_width,
    channels=1,
)
data = data.batch(batch_size)
data = data.map(lambda x, y: (augment_model(x), y))
data = data.prefetch(4)

def demo():
    fig, ax = plt.subplots()
    global data
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

def train():
    from model import Model 
    model = Model(
        batch_size=batch_size,
        frames_per_example=frames_per_example,
        video_height=video_height,
        video_width=video_width,
        channels=channels,
        num_classes=num_classes,
    ) # classifying 3 levels of drowsiness: low, med, high
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_cross_entropy',
        metrics=['accuracy'],
    )
    model.fit(data, steps_per_epoch=10, epochs=10)

def evaluate():
    assert False, 'TODO'
    test_loss, test_acc = model.evaluate(dataset)

if __name__ == '__main__':
    modes = dict(demo=demo, train=train, evaluate=evaluate)
    parser = ArgumentParser(
        prog='drowsiness classifier',
        description='sees if someone is drowsy',
        epilog='text at the bottom of help')
    parser.add_argument('-m', '--mode', choices=list(modes), default=next(iter(modes)))
    args = parser.parse_args()
    modes[args.mode]()
