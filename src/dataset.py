import av
from typing import Iterable
from functools import reduce
import numpy as np
from tensorflow.data import Dataset
from itertools import islice
import tensorflow as tf
from pathlib import Path

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
    shuffle_batch: int,
    frames_per_example: int,
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

