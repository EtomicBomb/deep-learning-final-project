#!/usr/bin/env python

import sys
import numpy as np
from numpy import ma
import scipy
from skimage import draw, transform, util
from numpy.lib.stride_tricks import sliding_window_view
import io
import time
from pathlib import Path
from scipy.ndimage import convolve1d
import itertools
import av
import json

destination_shape = (112, 224)
keypoint_destinations = np.array([
    [56, 180], # left eye, left
    [56, 134], # right eye, right
    [56, 90], # right, eye, right
    [56, 45], # right eye, left
    [112, 112], # bottom of nose
])

_, source, positions, *target = sys.argv

rotation = json.loads(Path('data/rotation.json').read_text())
rotation = rotation[str(Path(source).relative_to('data/init'))]
rotate = lambda x: np.rot90(x, -rotation // 90)
decode = lambda frame: rotate(frame.to_ndarray(format='gray'))

container = av.open(source)
stream, = container.streams.video
decoded = iter(container.decode(stream))
source_rows, source_cols = stream.height, stream.width
target_rows = 480
target_cols = target_rows * source_cols // source_rows

dummy = rotate(np.zeros((source_rows, source_cols), dtype=np.uint8))
rows, cols = np.shape(dummy)

loaded = np.load(positions)
valid_indexes, keypoints = loaded['frames'], loaded['keypoints']

frame_count = 1 + np.max(valid_indexes)

mask = np.full((frame_count, 1, 1), False)
mask[valid_indexes] = True

fixup = np.array([source_rows / target_rows, source_cols / target_cols])
keypoints = np.stack(keypoints) / fixup # fixup for mistake made in positions.py
keypoints = np.flip(keypoints, axis=2)
keypoints = np.stack(keypoints) * fixup
params = np.zeros((frame_count, 3, 3))
for keypoint, i in zip(keypoints, valid_indexes):
    tform = transform.SimilarityTransform()
    ok = tform.estimate(np.fliplr(keypoint), np.fliplr(keypoint_destinations))
    mask[i] &= ok
    if ok:
        params[i] = np.linalg.inv(tform.params)

half_weights = 3
weights = np.geomspace(1, 2 ** 3, half_weights + 1)
weights = np.concatenate((weights, np.flip(weights[:-1])))
weights = np.reshape(weights, (1, 1, 1, -1))

mask = np.pad(mask, ((half_weights, half_weights), (0, 0), (0, 0)))
params = np.pad(params, ((half_weights, half_weights), (0, 0), (0, 0)))
mask = sliding_window_view(mask, weights.size, axis=0)
params = sliding_window_view(params, weights.size, axis=0)
assert weights.shape == (1, 1, 1, weights.size)
assert params.shape == (frame_count, 3, 3, weights.size)
assert mask.shape == (frame_count, 1, 1, weights.size)

params = np.sum(weights * mask * params, axis=3) / np.sum(weights * mask)

if target:
    target, = target
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    container = av.open(target, mode='w', format='mp4')

    stream = container.add_stream('h264', rate=24)
    stream.height, stream.width = destination_shape
    stream.pix_fmt = 'gray'

    for frame, param in zip(decoded, params):
        frame = decode(frame)
        tform = transform.AffineTransform(matrix=param)
        frame = transform.warp(frame, tform, output_shape=destination_shape)
        frame = util.img_as_ubyte(frame)
        frame = av.VideoFrame.from_ndarray(frame, format='gray')
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)

else:
    import matplotlib.pyplot as plt
    from matplotlib import animation, widgets
    fig, (ax1, ax2) = plt.subplots(2)

    show1 = ax1.imshow(dummy.astype(np.float32))
    show2 = ax2.imshow(np.zeros(destination_shape, dtype=np.float32))

    def animate(param):
        frame = next(decoded)
        frame_index = frame.index
        frame = decode(frame)

        closest = np.argmin(np.abs(frame_index - valid_indexes))
        closest = keypoints[closest]
        
        frame1 = np.array(frame)
        for center in closest:
            rr, cc = draw.disk(center, 10)
            frame1[rr, cc] = 0
        show1.set_data(np.broadcast_to(frame1.reshape(*frame1.shape, 1), (*frame1.shape, 3)))

        tform = transform.AffineTransform(matrix=param)
        frame2 = transform.warp(frame1, tform, output_shape=destination_shape)
        show2.set_data(np.broadcast_to(frame2.reshape(*frame2.shape, 1), (*frame2.shape, 3)))

        return [show1, show2]
    ani = animation.FuncAnimation(fig, animate, frames=params, cache_frame_data=False, blit=True, interval=40)
    plt.show()
