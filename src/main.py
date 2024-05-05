#!/usr/bin/env python

from matplotlib.animation import FuncAnimation
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import matplotlib.pyplot as plt
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
    print(data.cardinality())
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

if __name__ == '__main__':
    modes = dict(demo=demo, index=index)
    parser = ArgumentParser(
        prog='drowsiness classifier',
        description='sees if someone is drowsy',
        epilog='text at the bottom of help')
    parser.add_argument('-m', '--mode', choices=list(modes), default=next(iter(modes)))
    args = parser.parse_args()
    modes[args.mode]()
