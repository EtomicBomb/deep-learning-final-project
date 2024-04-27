#!/usr/bin/env python

from matplotlib import animation, widgets
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import Dataset
import av
from typing import Iterable
from itertools import islice
from pathlib import Path
import json
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_graphics.image.transformer as tfg
from argparse import ArgumentParser
from skimage import transform

def file_ptss(path: Path, frames_per_example: int) -> Dataset:
    with av.open(str(path)) as container:
        stream, = container.streams
        stream = (frame.pts for frame in container.demux(stream) if frame.pts is not None)
        stream = islice(stream, 0, None, frames_per_example)
        stream = np.fromiter(stream, np.int64)
        stream = stream[:-1]
        return Dataset.from_tensor_slices(stream)

def more_data(
    paths: Iterable[str], 
    label: int, 
    frames_per_example: int,
) -> Dataset:
    """
    returns: Dataset[pts: int32, path: str, label: int]
    """
    ret = []
    for path in paths:
        path, = Path('data/extract', path).glob(f'{label}*.mp4')
        ptss = file_ptss(path, frames_per_example)
        ptss = ptss.map(lambda pts: (pts, str(path)))
        ret += [ptss]
    ret = reduce(Dataset.concatenate, ret)
    ret = ret.map(lambda pts, path: (pts, path, label))
    return ret

def get_dataset(
    paths: Iterable[str], 
    frames_per_example: int,
    shuffle_batch: int,
    video_height: int,
    video_width: int,
) -> Dataset:
    data = [
        more_data(paths, 0, frames_per_example),
        more_data(paths, 5, frames_per_example),
        more_data(paths, 10, frames_per_example),
    ]
    data = Dataset.sample_from_datasets(data, rerandomize_each_iteration=True) 
    data = data.shuffle(data.cardinality(), reshuffle_each_iteration=True)
    data = data.repeat()
    data = data.shuffle(shuffle_batch, reshuffle_each_iteration=True)

    @tf.py_function(Tout=tf.TensorSpec(shape=(frames_per_example, video_height, video_width, 1), dtype=tf.float32))
    def fetch_segment(pts: tf.int64, path: tf.string, frames_per_example: tf.int64): 
        path = path.numpy().decode('utf-8')
        pts = int(pts.numpy())
        with av.open(path) as container:
            stream, = container.streams
            container.seek(pts, backward=True, any_frame=False, stream=stream)
            stream = iter(container.decode(stream))
            frames = []
            while len(frames) < frames_per_example:
                frame = next(stream)
                if frame.pts >= pts:
                    frame = frame.to_ndarray()
                    frame = frame[:video_height, :video_width] # XXX: why?
                    frame = np.expand_dims(frame, -1)
                    frame = np.float32(frame) / 255.0
                    frames.append(tf.convert_to_tensor(frame))
            return tf.stack(frames)
    data = data.map(lambda pts, path, label: (fetch_segment(pts, path, frames_per_example), label))
    return data

class VideoRandomOperation(keras.layers.Layer):
    def __init__(self, operation, rng=None):
        super().__init__(trainable=False)
        if rng is None:
            rng = tf.random.Generator.from_non_deterministic_state()
        self.rng = rng
        self.operation = operation
    def call(self, x, training=True):
        return self.operation(x, self.rng) if training else x

def video_random_flip(x, rng):
    batch_size, frame_count, height, width, channels = x.shape
    mask = rng.binomial(shape=(batch_size,), counts=1., probs=0.5)
    mask = tf.reshape(mask, (batch_size, 1, 1, 1, 1))
    return tf.where(mask == 1, x, tf.reverse(x, axis=(3,)))

def smooth_exponential(x, base, mean, stddev, rng):
    """
    x: length 
    w: size of the window (convolution window has length 2 * w - 1)
    base: larger means sharper exponential
    """
    w = x
    w = base ** tf.range(w, dtype=tf.float32)
    w = tf.concat((w[:-1], tf.reverse(w, axis=(0,))), axis=0)
    w = w / tf.reduce_sum(w)
    w = tf.reshape(w, (-1, 1, 1))
    x = x + w.shape[0] - 1
    x = rng.truncated_normal(shape=(1, x, 1), mean=mean, stddev=stddev)
    x = tf.nn.convolution(x, w)
    x = tf.reshape(x, (-1,))
    return x

def video_random_contrast(x, rng):
    batch_size, frame_count, height, width, channels = x.shape
    mask = smooth_exponential(x=batch_size * frame_count, base=1.1, mean=1., stddev=1.5, rng=rng)
    mask = tf.reshape(mask, (batch_size, frame_count, 1, 1, 1))
    x = (x - 0.5) * mask + 0.5
    return tf.clip_by_value(x, 0.0, 1.0)

def video_random_brightness(x, rng):
    batch_size, frame_count, height, width, channels = x.shape
    mask = smooth_exponential(x=batch_size * frame_count, base=1.1, mean=1., stddev=1.5, rng=rng)
    mask = tf.reshape(mask, (batch_size, frame_count, 1, 1, 1))
    x = x * mask
    return tf.clip_by_value(x, 0.0, 1.0)

def scale(sr: tf.float32, sc: tf.float32):
    return tf.convert_to_tensor([[[sc, 0, 0], [0, sr, 0], [0, 0, 1]]], dtype=tf.float32)

def translation(tr: tf.float32, tc: tf.float32):
    return tf.convert_to_tensor([[[1, 0, tc], [0, 1, tr], [0, 0, 1]]], dtype=tf.float32)

def video_perspective_transform():
    corners = {(0, 0): (-1, 0), (0, 1): (0, 2), (1, 0): (1, -1), (1, 1): (2, 1)}
    src = list(corners)
    warp_basis = []
    for corner, tweak in corners.items():
        dst = [tweak if s == corner else s for s in src]
        tform = transform.ProjectiveTransform()
        success = tform.estimate(np.array(dst) - 0.5, np.array(src) - 0.5, )
        assert success
        warp_basis += [tform.params]
    warp_basis = tf.convert_to_tensor(warp_basis, dtype=np.float32)
    warp_basis = tf.reshape(warp_basis, (4, 1, 3, 3))
    def ret(x, rng):
        batch_size, frame_count, height, width, channels = x.shape
        length = batch_size * frame_count
        a = smooth_exponential(x=length, base=1.1, mean=0., stddev=5.0, rng=rng)
        b = smooth_exponential(x=length, base=1.1, mean=0., stddev=5.0, rng=rng)
        c = smooth_exponential(x=length, base=1.1, mean=0., stddev=5.0, rng=rng)
        d = smooth_exponential(x=length, base=1.1, mean=0., stddev=5.0, rng=rng)
        warp = tf.reshape([a, b, c, d], (4, length, 1, 1))
        warp = tf.math.maximum(warp, 0.0)
        warp = warp * warp_basis + (1 - warp) * tf.eye(3)
        warp = tf.reduce_sum(warp, axis=0)
        warp = (
            scale(height-1, width-1) 
            @ translation(0.5, 0.5)
            @ warp
            @ translation(-0.5, -0.5)
            @ scale(1/(height-1), 1/(width-1))
        )
        x = tf.reshape(x, (length, height, width, channels))
        x = tfg.perspective_transform(x, warp)
        x = tf.reshape(x, (batch_size, frame_count, height, width, channels))
        return x
    return ret

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
    ).prefetch(4)

    augment_model = tf.keras.Sequential([
        VideoRandomOperation(video_perspective_transform()),
        VideoRandomOperation(video_random_flip),
        VideoRandomOperation(video_random_contrast),
        VideoRandomOperation(video_random_brightness),
    ])

    fig, ax = plt.subplots()
    data = iter((x, y) for xs, y in data for x in tf.squeeze(augment_model(tf.expand_dims(xs, 0)), 0))
    image = ax.imshow(next(data)[0], cmap='gray')
    def animate(data):
        x, y = data
        image.set_data(x.numpy())
#         print(y.numpy())
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
