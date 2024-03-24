import sys
from functools import partial
import numpy as np
from numpy import ma
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation, widgets
from skimage import draw, transform
from numpy.lib.stride_tricks import sliding_window_view
import cv2
import imageio.v3 as iio
import io
import time
from pathlib import Path
from scipy.ndimage import convolve1d
import itertools
import av
import json

colors = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
]

destination_shape = (224, 224)
keypoint_destinations = np.array([
    [75, 180], # left eye, left
    [75, 134], # right eye, right
    [75, 90], # right, eye, right
    [75, 45], # right eye, left
    [150, 112], # bottom of nose
])

_, source = sys.argv

rotation = json.loads(Path('data/rotation.json').read_text())
rotation = rotation[source]
rotate = lambda x: np.rot90(x, -rotation // 90)
decode = lambda frame: rotate(frame.to_ndarray(format='rgb24'))

container = av.open(str(Path('data/init', source)))
stream, = container.streams.video
decoded = iter(container.decode(stream))
source_rows, source_cols = stream.height, stream.width
target_rows = 480
target_cols = target_rows * source_cols // source_rows

dummy = rotate(np.zeros((source_rows, source_cols, 3)))
rows, cols, _ = np.shape(dummy)

loaded = np.load(Path('data/positions', source).with_suffix('.npz'))
valid_indexes, keypoints = loaded['frames'], loaded['keypoints']

frame_count = int(1.2 * np.max(valid_indexes))

mask = np.full((frame_count, 1, 1), False)
mask[valid_indexes] = True

fixup = np.array([source_rows / target_rows, source_cols / target_cols])
keypoints = np.stack(keypoints) / fixup # fixup for mistake made in positions.py
keypoints = np.flip(keypoints, axis=2)
keypoints = np.stack(keypoints) * fixup
params = np.zeros((frame_count, 3, 3))
for keypoint, i in zip(keypoints, valid_indexes):
    tform = transform.AffineTransform()
    ok = tform.estimate(np.fliplr(keypoint), np.fliplr(keypoint_destinations))
    mask[i] &= ok
    if ok:
        params[i] = np.linalg.inv(tform.params)

half_weights = 5
weights = np.geomspace(1, 2 ** 3, half_weights + 1)
weights = np.concatenate((weights, np.flip(weights[:-1])))
weights = np.reshape(weights, (1, 1, 1, -1))
print(weights)

mask = np.pad(mask, ((half_weights, half_weights), (0, 0), (0, 0)))
params = np.pad(params, ((half_weights, half_weights), (0, 0), (0, 0)))
mask = sliding_window_view(mask, weights.size, axis=0)
params = sliding_window_view(params, weights.size, axis=0)
assert weights.shape == (1, 1, 1, weights.size)
assert params.shape == (frame_count, 3, 3, weights.size)
assert mask.shape == (frame_count, 1, 1, weights.size)

params = np.sum(weights * mask * params, axis=3) / np.sum(weights * mask)

fig, (ax1, ax2) = plt.subplots(2)

show1 = ax1.imshow(dummy)
show2 = ax2.imshow(np.zeros((*destination_shape, 3)))

def animate(param):
    frame = next(decoded)
    frame_index = frame.index
    frame = decode(frame)

    closest = np.argmin(np.abs(frame_index - valid_indexes))
    closest = keypoints[closest]
    
    frame1 = np.array(frame)
    for center, color in zip(closest, colors):
        rr, cc = draw.disk(center, 10)
        frame1[rr, cc, :] = color
    show1.set_data(frame1)

    tform = transform.AffineTransform(matrix=param)
    frame2 = transform.warp(frame, tform, output_shape=destination_shape)
    show2.set_data(frame2)

    return [show1, show2]
ani = animation.FuncAnimation(fig, animate, frames=params, cache_frame_data=False, blit=True, interval=40)
plt.show()

