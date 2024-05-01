#!/usr/bin/env python

from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from dataclasses import dataclass
import time

from dataset import get_dataset
from augment import VideoRandomPerspective, VideoRandomFlip, VideoRandomContrast, VideoRandomMultiply, VideoRandomAdd, VideoRandomNoise, VideoCropAndResize, ClipZeroOne, Scale, Gray2RGB
from dimensions import Dimensions

s = Dimensions(
    batch_size=2,
    frame_count=32,
    height=112,
    width=224,
    channels=1,
)
rng = tf.random.Generator.from_non_deterministic_state()

augment_model = keras.Sequential([
    keras.Input(shape=(s.example_shape), batch_size=s.batch_size),
    VideoCropAndResize(),
    VideoRandomNoise(rng=rng),
    VideoRandomPerspective(rng=rng),
    VideoRandomFlip(rng=rng),
    VideoRandomContrast(rng=rng),
    VideoRandomMultiply(rng=rng),
    VideoRandomAdd(rng=rng),
    Scale(),
    Gray2RGB(),
    ClipZeroOne(),
])

data = json.loads(Path('data/split.json').read_text())
data = data['train'] # select only one participant so we can see changes easily
data = get_dataset(
    data_root='data/extract',
    paths=data,
    s=s,
)
data = data.map(lambda x, y: (augment_model(x), y))
data = data.prefetch(4)

def demo():
    fig, ax = plt.subplots()
    global data
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

if __name__ == '__main__':
    modes = dict(demo=demo)
    parser = ArgumentParser(
        prog='drowsiness classifier',
        description='sees if someone is drowsy',
        epilog='text at the bottom of help')
    parser.add_argument('-m', '--mode', choices=list(modes), default=next(iter(modes)))
    args = parser.parse_args()
    modes[args.mode]()
