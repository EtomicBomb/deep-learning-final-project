#!/usr/bin/env python

import tensorflow as tf
from tensorflow.data import Dataset
import av
from numpy.lib.stride_tricks import sliding_window_view
from itertools import takewhile
from pathlib import Path
import json

@tf.py_function(Tout=(tf.TensorSpec(shape=(None, 224, 224), dtype=tf.uint8), tf.int32))
def fetch_segment(path, lo, hi, label): 
    path, lo, hi, label = path.numpy().decode('utf-8'), int(lo.numpy()), int(hi.numpy()), label
    with av.open(path) as container:
        stream, = container.streams
        container.seek(lo, backward=True, any_frame=False, stream=stream)
        undecoded = takewhile(lambda frame: frame.pts < hi, container.decode(stream)) 
        return tf.stack([tf.constant(frame.to_ndarray()) for frame in undecoded]), label

def process_file(path):
    for path in Path('data/extract', path).glob('*'):
        path = path.with_suffix('.mp4')
        label, *_ = path.with_suffix('').name.split('_')
        label = int(label)
        with av.open(str(path)) as container:
            stream, = container.streams
            keyframes = [frame.pts for frame in container.demux(stream) if frame.is_keyframe]
            keyframes_per_example = 3 
            lo = keyframes[::keyframes_per_example][:-1]
            hi = keyframes[::keyframes_per_example][1:]
            for lo, hi in zip(lo, hi):
                yield str(path), lo, hi, label

def get_dataset(paths):
    datasets = {0: [], 5: [], 10: []}

    for path in paths:
        for path, lo, hi, label in process_file(path):
            datasets[label].append((path, lo, hi, label))

    sample_from = []
    for label, d in datasets.items():
        e = Dataset.from_generator(
            lambda: d,
            output_signature=(
                tf.TensorSpec((), tf.string),
                tf.TensorSpec((), tf.int32),
                tf.TensorSpec((), tf.int32),
                tf.TensorSpec((), tf.int32),
            ),
        )

        e = e.shuffle(e.cardinality(), reshuffle_each_iteration=True)
        e = e.map(fetch_segment)
        sample_from.append(e)

    return Dataset.sample_from_datasets(
        sample_from,
        rerandomize_each_iteration=True,
    )

splits = json.loads(Path('data/split.json').read_text())

# train = get_dataset(splits['train']).batch(8).prefetch(2)
# test = get_dataset(splits['test']).batch(8).prefetch(2)
# validation = get_dataset(splits['validation']).batch(8).prefetch(2)

for x, y in get_dataset(splits['train'][:4]).as_numpy_iterator():
    print(x, y)
