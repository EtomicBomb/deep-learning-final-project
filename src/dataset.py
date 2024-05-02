import av
from typing import Iterable
from functools import reduce
import numpy as np
from tensorflow.data import Dataset
from itertools import islice
import tensorflow as tf
from pathlib import Path

from dimensions import Dimensions

def index_video(path: Path, frame_count: int) -> Dataset:
    """
    Returns a dataset over the presentation timestamps in the source video (tf.int64)
    """
    with av.open(str(path)) as container:
        print('b', len(container.streams))
        stream, = container.streams
        stream = container.demux(stream)
        stream = (frame.pts for frame in stream if frame.pts is not None)
        stream = islice(stream, 0, None, frame_count)
        stream = np.fromiter(stream, np.int64)
        stream = stream[:-1]
        return Dataset.from_tensor_slices(stream)

def index_videos(
    paths: Iterable[str], 
    label: str, 
    frame_count: int,
) -> tuple[Dataset]:
    """
    returns a dataset over the presentation timestamps in all the source videos
    returns: Dataset[pts: int64, path: str]
    """
    ret = []
    for path in paths:
        for path in path.glob(f'{label}*.mp4'):
            ptss = index_video(path, frame_count)
            ptss = ptss.map(lambda pts: (pts, str(path)))
            ret += [ptss]
    return reduce(Dataset.concatenate, ret)

def get_dataset(
    data_root: str,
    paths: Iterable[str], 
    s: Dimensions,
) -> Dataset:
    """
    @param data_root the path ending in data/extract
    @param paths paths relative to data_root
    """
    paths = [Path(data_root, path) for path in paths]
    data = [
        index_videos(paths, '0', s.frame_count).map(lambda pts, path: (pts, path, 0)),
        index_videos(paths, '5', s.frame_count).map(lambda pts, path: (pts, path, 1)),
        index_videos(paths, '10', s.frame_count).map(lambda pts, path: (pts, path, 2)),
    ]
    data = Dataset.sample_from_datasets(data, stop_on_empty_dataset=True, rerandomize_each_iteration=True) 
    data = data.shuffle(data.cardinality(), reshuffle_each_iteration=True)

    @tf.py_function(Tout=tf.TensorSpec(shape=(s.example_shape), dtype=tf.float32))
    def fetch_segment(pts: tf.int64, path: tf.string, frame_count: tf.int64): 
        path = path.numpy().decode('utf-8')
        pts = int(pts.numpy())
        with av.open(path) as container:
            print('a', len(container.streams))
            stream, = container.streams
            container.seek(pts, backward=True, any_frame=False, stream=stream)
            stream = iter(container.decode(stream))
            frames = []
            while len(frames) < frame_count:
                frame = next(stream)
                if frame.pts >= pts:
                    frame = frame.to_ndarray()
                    frame = frame[:s.height] # XXX: why?
                    frame = np.atleast_3d(frame)
                    frame = np.float32(frame) / 255.0
                    frames.append(tf.convert_to_tensor(frame))
            return tf.stack(frames)
    data = data.repeat()
    data = data.map(lambda pts, path, label: (fetch_segment(pts, path, s.frame_count), label))
    data = data.map(lambda data, label: (
        tf.ensure_shape(data, (s.example_shape)), 
        tf.ensure_shape(label, ()),
    ))
    data = data.batch(s.batch_size)
    return data

