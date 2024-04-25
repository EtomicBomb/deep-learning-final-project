#!/usr/bin/env python

from matplotlib import animation, widgets
import tensorflow as tf
from tensorflow.data import Dataset
import av
from typing import Iterable
from numpy.lib.stride_tricks import sliding_window_view
from itertools import islice
from pathlib import Path
import json
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt

video_height = 112
video_width = 224

def file_ptss(path: Path, frames_per_example: int) -> Dataset:
    with av.open(str(path)) as container:
        stream, = container.streams
        stream = (frame.pts for frame in container.demux(stream) if frame.pts is not None)
        stream = islice(stream, 0, None, frames_per_example)
        stream = np.fromiter(stream, np.int64)
        stream = stream[:-1]
        return Dataset.from_tensor_slices(stream)

def more_data(
    paths: Iterable[Path], 
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
    paths: Iterable[Path], 
    frames_per_example: int,
    shuffle_batch: int,
):
    data = [
        more_data(paths, 0, frames_per_example),
        more_data(paths, 5, frames_per_example),
        more_data(paths, 10, frames_per_example),
    ]
    data = Dataset.sample_from_datasets(data, rerandomize_each_iteration=True) 
    data = data.shuffle(data.cardinality(), reshuffle_each_iteration=True)
    data = data.repeat()
    data = data.shuffle(shuffle_batch, reshuffle_each_iteration=True)

    @tf.py_function(Tout=tf.TensorSpec(shape=(frames_per_example, video_height, video_width), dtype=tf.uint8))
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
                    frames.append(tf.convert_to_tensor(frame.to_ndarray()))
            return tf.stack(frames)
    data = data.map(lambda pts, path, label: (fetch_segment(pts, path, frames_per_example), label))
    return data

splits = json.loads(Path('data/split.json').read_text())
data = splits['train'][:2]
data = list(map(Path, data))
data = get_dataset(data, 30 * 3, 1000).prefetch(4)

fig, ax = plt.subplots()
data = iter((np.float32(x.numpy()) / 255.0, y.numpy()) for xs, y in data for x in xs)
image = ax.imshow(next(data)[0], cmap='gray')
def animate(data):
    data, y = data
    image.set_data(data)
    print(y)
    return [image]
ani = animation.FuncAnimation(fig, animate, data, cache_frame_data=False, blit=True, interval=1)
plt.show()

# train = get_dataset(splits['train']).batch(8).prefetch(2)
# test = get_dataset(splits['test']).batch(8).prefetch(2)
# validation = get_dataset(splits['validation']).batch(8).prefetch(2)
