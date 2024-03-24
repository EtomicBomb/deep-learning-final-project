import sys
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation, widgets
from skimage import draw, transform
import cv2
import imageio.v3 as iio
import io
import time
from pathlib import Path
import itertools
import av
import json

colors = [
    [255, 0, 0], # left left
    [0, 255, 0], # left right
    [0, 0, 255], # right right
    [255, 255, 0], # right left
    [0, 255, 255], # nose
]

keypoint_destinations = np.array([
    [75, 180],
    [75, 134],
    [75, 90],
    [75, 45],
    [150, 112],
])

class Main:
    def __init__(self, source):
        rotation = Path('data', 'rotation.json')
        rotation = json.loads(rotation.read_text())
        rotation = rotation[source]
        rotate = lambda x: np.rot90(x, -rotation // 90)
        self.decode = lambda frame: rotate(frame.to_ndarray(format='rgb24'))

        container = av.open(str(Path('data', 'init', source)))
        self.stream, = container.streams.video
        self.decoded = iter(container.decode(self.stream))
        source_rows, source_cols = self.stream.height, self.stream.width
        target_rows = 480
        target_cols = target_rows * source_cols // source_rows

        dummy = rotate(np.zeros((source_rows, source_cols, 3)))
        rows, cols, _ = np.shape(dummy)
        print(cols)

        loaded = np.load(Path('data', 'positions', source).with_suffix('.npz'))
        self.frames, self.keypoints = loaded['frames'], loaded['keypoints']
        fixup = np.array([source_rows / target_rows, source_cols / target_cols])
        self.keypoints = np.stack(self.keypoints) / fixup # fixup for mistake made in positions.py
        self.keypoints = np.flip(self.keypoints, axis=2)
        self.keypoints = np.stack(self.keypoints) * fixup

        fig, (ax1, ax2) = plt.subplots(2)

        self.show1 = ax1.imshow(dummy)
        self.show2 = ax2.imshow(np.zeros((224, 224, 3)))

        ani = animation.FuncAnimation(fig, self.animate, cache_frame_data=False, blit=True, interval=40)
        plt.show()

    def animate(self, frame_count):
        frame = next(self.decoded)
        frame_index = frame.index
        frame = self.decode(frame)

        closest = np.argmin(np.abs(frame_index - self.frames))
        closest = self.keypoints[closest]
        
        show1 = np.array(frame)
        for center, color in zip(closest, colors):
            rr, cc = draw.disk(center, 10)
            show1[rr, cc, :] = color
        self.show1.set_data(show1)

        tform = transform.SimilarityTransform()
        ok = tform.estimate(np.fliplr(closest), np.fliplr(keypoint_destinations))
        if ok:
            show2 = transform.warp(frame, tform.inverse, output_shape=(224, 224))
            self.show2.set_data(show2)

        return [self.show1, self.show2]
            
_, source = sys.argv

Main(source)
