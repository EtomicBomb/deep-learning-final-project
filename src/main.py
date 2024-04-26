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

def smooth_exponential(x, w, base, mean, stddev, rng):
    """
    x: length 
    w: size of the window (convolution window has length 2 * w - 1)
    base: larger means sharper exponential
    """
    w = base ** tf.range(w, dtype=tf.float32)
    w = tf.concat((w[:-1], tf.reverse(w, axis=(0,))), axis=0)
    w = w / tf.reduce_sum(w)
    w = tf.reshape(w, (-1, 1, 1))
    x = x + len(w) - 1
    x = rng.truncated_normal(shape=(1, x, 1), mean=mean, stddev=stddev)
    x = tf.nn.convolution(x, w)
    x = tf.reshape(x, (-1,))
    return x

def video_random_contrast(x, rng):
    batch_size, frame_count, height, width, channels = x.shape
    mask = smooth_exponential(x=batch_size * frame_count, w=30, base=1.1, mean=1., stddev=1.5, rng=rng)
    mask = tf.reshape(mask, (batch_size, frame_count, 1, 1, 1))
    x = (x - 0.5) * mask + 0.5
    return tf.clip_by_value(x, 0.0, 1.0)

def video_random_brightness(x, rng):
    batch_size, frame_count, height, width, channels = x.shape
    mask = smooth_exponential(x=batch_size * frame_count, w=30, base=1.1, mean=1., stddev=1.5, rng=rng)
    mask = tf.reshape(mask, (batch_size, frame_count, 1, 1, 1))
    x = x * mask
    return tf.clip_by_value(x, 0.0, 1.0)

def video_crop_and_resize(x, rng):
    batch_size, frame_count, height, width, channels = x.shape
    length = batch_size * frame_count
    x1 = smooth_exponential(x=length, w=30, base=1.1, mean=0., stddev=0.2, rng=rng)
    x2 = smooth_exponential(x=length, w=30, base=1.1, mean=1., stddev=0.2, rng=rng)
    y1 = smooth_exponential(x=length, w=30, base=1.1, mean=0., stddev=0.2, rng=rng)
    y2 = smooth_exponential(x=length, w=30, base=1.1, mean=1., stddev=0.2, rng=rng)
    boxes = tf.transpose([y1, x1, y2, x2])
    boxes = tf.clip_by_value(boxes, 0.0, 1.0)
    x = tf.reshape(x, (length, height, width, channels))
    x = tf.image.crop_and_resize(
        x, 
        boxes=boxes,
        box_indices=tf.range(batch_size * frame_count),
        crop_size=(height, width))
    return tf.reshape(x, (batch_size, frame_count, height, width, channels))

def translation(v: tf.float32, h: tf.float32):
    return tf.convert_to_tensor([[[1, 0, h], [0, 1, v], [0, 0, 1],]], dtype=tf.float32)

# 0.1 -> -0.1
# 0.9 -> 1.1
# 0.9 + 2 - 2 * 0.9)

inputs = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]

inputs = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
]
out1 = [

# def warp_crop_boxes(warp, height, width):
#     height = tf.cast(height, tf.float32)
#     width = tf.cast(width, tf.float32)
#     corners = [[0, 0, 1], [0, height-1, 1], [width-1, 0, 1], [width-1, height-1, 1]] # 4, 3 -> 
#     corners = tf.reshape(corners, (4, 1, 3, 1))
#     corners = tf.cast(corners, tf.float32)
#     corners = warp @ corners # -> 4, 3, 1
#     corners = corners[:, :, :2, 0] / corners[:, :, 2:, 0] # -> 4, N, 2
#     corners = corners / tf.reshape([width - 1, height], (1, 1, 2))
# #     corners = tf.clip_by_value(corners, 0.0, 1.0)
#     cs, rs = tf.unstack(corners, axis=2) # -> 4, N
#     print(cs.shape)
#     y1 = tf.math.reduce_min(rs, axis=(0,))
#     x1 = tf.math.reduce_min(cs, axis=(0,))
#     y2 = tf.math.reduce_max(rs, axis=(0,))
#     x2 = tf.math.reduce_max(cs, axis=(0,))
#     boxes = tf.transpose([y1, x1, y2, x2])
#     print(boxes)
#     return boxes

def video_perspective_transform(x, rng):
    batch_size, frame_count, height, width, channels = x.shape
    length = batch_size * frame_count
    t = smooth_exponential(x=length, w=50, base=1.1, mean=10.0, stddev=0., rng=rng)
    a = smooth_exponential(x=length, w=50, base=1.1, mean=0., stddev=0.0, rng=rng)
    b = smooth_exponential(x=length, w=50, base=1.1, mean=0., stddev=0.0, rng=rng)
    c = smooth_exponential(x=length, w=50, base=1.1, mean=1., stddev=0.0, rng=rng)
    o = tf.ones_like(a)
    z = tf.zeros_like(a)
    warp = tf.transpose([[o, z, t], [z, o, z], [a, b, c]], perm=(2, 0, 1))
    warp = translation(height/2, width/2) @ warp @ translation(-height/2, -width/2)
    x = tf.reshape(x, (length, height, width, channels))
    x = tfg.perspective_transform(x, warp)
    x = tf.image.crop_and_resize(
        x, 
        boxes=warp_crop_boxes(warp, height, width),
        box_indices=tf.range(length),
        crop_size=(height, width))
    return tf.reshape(x, (batch_size, frame_count, height, width, channels))

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
        VideoRandomOperation(video_perspective_transform),
#         VideoRandomOperation(video_random_flip),
#         VideoRandomOperation(video_random_contrast),
#         VideoRandomOperation(video_random_brightness),
#         VideoRandomOperation(video_crop_and_resize),
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

demo()
# train = get_dataset(splits['train']).batch(8).prefetch(2)
# test = get_dataset(splits['test']).batch(8).prefetch(2)
# validation = get_dataset(splits['validation']).batch(8).prefetch(2)
