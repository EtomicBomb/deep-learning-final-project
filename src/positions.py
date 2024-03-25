#!/usr/bin/env python

from pathlib import Path
import sys
import av
import numpy as np
import dlib
from skimage import transform, util
import matplotlib.pyplot as plt
import imageio
import subprocess
from itertools import islice
import sys
import json

_, source, target = sys.argv

rotation = Path('data/rotation.json')
rotation = json.loads(rotation.read_text())
rotation = rotation[str(Path(source).relative_to('data/init'))]
rotate = lambda x: np.rot90(x, -rotation // 90)

detector = dlib.cnn_face_detection_model_v1('data/mmod_human_face_detector.dat')
predictor = dlib.shape_predictor('data/shape_predictor_5_face_landmarks.dat')

frames = []
keypoints = []

drop_count = 0
last_decoded = -np.inf

container = av.open(source)
stream, = container.streams.video
source_rows, source_cols = stream.height, stream.width
target_rows = 480
target_cols = target_rows * source_cols // source_rows
for frame in container.decode(stream):
    frame_index = frame.index
    if frame_index % 1000 == 0 and frame_index > 0:
        print(f'source {frame_index=} okay_count={len(frames)} {drop_count=}', file=sys.stderr)
    if frame_index - last_decoded < 2: # skip
        continue
    frame = frame.to_ndarray(format='gray', height=target_rows, width=target_cols, interpolation='FAST_BILINEAR')
    frame = rotate(frame)
    points = detector(frame)
    if len(points) != 1:
        drop_count += 1
        continue
    last_decoded = frame_index
    frames.append(frame_index)
    points = predictor(frame, points[0].rect)
    keypoints.append(np.vstack([(p.x, p.y) for p in points.parts()]))

frames = np.array(frames)
keypoints = np.stack(keypoints) * [source_rows / target_rows, source_cols / target_cols]

print(f'source okay_count={len(frames)} {drop_count=}', file=sys.stderr)

Path(target).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(target, frames=frames, keypoints=keypoints)
