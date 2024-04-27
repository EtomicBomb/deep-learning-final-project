#!/usr/bin/env python

from matplotlib import animation, widgets
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import json
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from dataset import get_dataset
from augment import VideoRandomPerspective, VideoRandomFlip, VideoRandomContrast, VideoRandomBrightness

def demo():
    frames_per_example = 30 * 3
    video_height = 112
    video_width = 224
    data = json.loads(Path('data/split.json').read_text())
    data = data['train'][:1] # select only one participant so we can see changes easily
    data = get_dataset(
        data,
        frames_per_example, 
        shuffle_batch=1000, 
        video_height=video_height, 
        video_width=video_width,
    ).prefetch(4).batch(1).as_numpy_iterator()

    augment_model = tf.keras.Sequential([
        VideoRandomPerspective(),
        VideoRandomFlip(),
        VideoRandomContrast(),
        VideoRandomBrightness(),
    ])

    fig, ax = plt.subplots()
    data = iter((x, y) for xs, y in data for x in tf.squeeze(augment_model(xs), 0))
    image = ax.imshow(next(data)[0], cmap='gray')
    def animate(data):
        x, y = data
        image.set_data(x)
        print(y)
        return [image]
    ani = animation.FuncAnimation(fig, animate, data, cache_frame_data=False, blit=True, interval=1)
    plt.show()

def train():
    from model import Model 
    batch_size = 4
    frames_per_example = 30 * 3
    video_height = 112
    video_width = 224
    model = Model(
        batch_size=batch_size,
        frames_per_example=frames_per_example,
        video_height=video_height,
        video_width=video_width,
        channels=1,
        num_classes=3,
    ) # classifying 3 levels of drowsiness: low, med, high
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_cross_entropy',
        metrics=['accuracy'],
    )

    splits = json.loads(Path('data/split.json').read_text())
    dataset = splits['train'][:1]
    dataset = get_dataset(
        dataset, 
        frames_per_example, 
        shuffle_batch=1000, 
        video_height=video_height, 
        video_width=video_width,
    ).prefetch(4).batch(batch_size)
    model.fit(dataset, steps_per_epoch=10, epochs=10)

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
