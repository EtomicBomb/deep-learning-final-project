#!/usr/bin/env python

from pathlib import Path
import sys
from skimage import transform

_, init_source, positions_source, target = sys.argv

# positions_source = np.load(positions_source)
# valids = positions_source['valids']
# keypoints = positions_source['keypoints']
# 
# Path(target).parent.mkdir(parents=True, exist_ok=True)
# 
# with av.open(init_source, mode='r') as init_source:
#     with av.open(target, mode='w') as target:
#         init_stream, = init_source.streams.video
#         target = test_output.add_stream(init_stream.codec_context.name, str(init_stream.codec_context.rate))
#         target.codec_context = init_source.codec_context
# 
#         for valids, keypoints, frame in zip(valids, keypoints, init_source.decode(init_stream)):
#             frame = frame.to_ndarray(format='rgb24')
#             affine = tranform.AffineTransform()
#             affine.estimate(src, dst)
#             out = ski.transform.warp(frame, inverse_map=tform.inverse)
#             out = av.VideoFrame.from_image(out)
#             out = target.encode(out)
#             target.mux(out)
# 
#         out = target.encode(None)
#         target.mux(out)
# 
# test_input.close()
# test_output.close()


Path(target).parent.mkdir(parents=True, exist_ok=True)
Path(target).write_text('a')
