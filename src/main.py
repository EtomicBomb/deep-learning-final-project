#!/usr/bin/env python

import tensorflow as tf
from tensorflow.data import Dataset
import av
from numpy.lib.stride_tricks import sliding_window_view
from itertools import takewhile
from pathlib import Path
import json

# no, this doesn't work
# E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:274] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
# F external/local_tsl/tsl/platform/default/env.cc:74] Check failed: ret == 0 (11 vs. 0)Thread tf_data_private_threadpool creation via pthread_create() failed.


@tf.py_function(Tout=(tf.TensorSpec(shape=(None, 224, 224), dtype=tf.uint8), tf.int32))
def fetch_segment(path, lo, hi, label): 
    print(path, lo, hi, label)
    with av.open(path) as container:
        stream, = container.streams
        container.seek(lo, backwards=True, any_frame=False, stream=stream)
        undecoded = takewhile(frame.pts < hi, container.decode(stream)) 
        return tf.stack([frame.to_ndarray() for frame in undecoded]), label

def process_file(path):
    for path in Path('data/extract', path).glob('*'):
        print(path)
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
                yield path, lo, hi, label

def get_dataset(paths):
    datasets = {0: [], 5: [], 10: []}

    for path in paths:
        for path, lo, hi, label in process_file(path):
            datasets[label].append((path, lo, hi, label))

    sample_from = []
    for label, d in datasets.items():
        d = Dataset.from_generator(
            lambda: d,
            output_signature=(
                tf.TensorSpec((), tf.string),
                tf.TensorSpec((), tf.int32),
                tf.TensorSpec((), tf.int32),
                tf.TensorSpec((), tf.int32),
            ),
        )

        d = d.shuffle(d.cardinality(), reshuffle_each_iteration=True)
        d.map(fetch_segment)
        sample_from.append(d)

    d = Dataset.sample_from_datasets(
        sample_from,
        rerandomize_each_iteration=True,
    )
    return d

splits = json.loads(Path('data/split.json').read_text())

# train = get_dataset(splits['train']).batch(8).prefetch(2)
# test = get_dataset(splits['test']).batch(8).prefetch(2)
# validation = get_dataset(splits['validation']).batch(8).prefetch(2)

for x in get_dataset(splits['train']).as_numpy_iterator():
    print(x)
