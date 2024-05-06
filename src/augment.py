from skimage import transform
import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np
from collections import namedtuple
import keras_cv
import warnings
from abc import ABC, abstractmethod

from dimensions import Dimensions

class PreprocessingLayer(keras.layers.Layer, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, trainable=False, **kwargs)
        self.s = None
    
    def build(self, shape):
        self.s = Dimensions(
            batch_size=shape[0],
            frame_count=shape[1],
            height=shape[2],
            width=shape[3],
            channels=shape[4],
        )
        
    @abstractmethod
    def operation(self, x, s):
        ...

    def call(self, x):
        return self.operation(x, self.s)

class Scale(PreprocessingLayer):
    def operation(self, x, s):
        dims = s.shape
        print(f"running Scale...")
        x = tf.reshape(x, [dims[0]*dims[1], dims[2], dims[3], dims[4]]) # flatten batch & frames dimensions 
        x = tf.image.resize_with_pad(x, target_height=224, target_width=224) 
        x = tf.reshape(x, [dims[0], dims[1], 224, dims[3], dims[4]])
        return x

class Gray2RGB(PreprocessingLayer):
    def operation(self, x, dims):
        print(f"running Gray2RGB...")
        rgb_dims = [1, 1, 1, 1, 3]
        x = tf.tile(x, rgb_dims)
        return x

class ClipZeroOne(PreprocessingLayer):
    def operation(self, x, s):
        return tf.clip_by_value(x, 0.0, 1.0)

class VideoCropAndResize(PreprocessingLayer):
    def operation(self, x, s):
        l = s.batch_size * s.frame_count
        x = tf.reshape(x, (l, s.height, s.width, s.channels))
        cut_from_height = 40
        cut_from_width = 40
        x = tf.image.crop_to_bounding_box(
            x, 
            cut_from_height, 
            cut_from_width, 
            s.height - 2 * cut_from_height, 
            s.width - 2 * cut_from_width,
        )
        x = tf.image.resize(x, (s.height, s.width))
        x = tf.reshape(x, s.shape)
        return x

class VideoRandomAugmentation(PreprocessingLayer, ABC):
    def __init__(self, *args, rng=None, augment_fraction=0.5, smooth_base=None, smooth_mean=None, smooth_std=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = rng
        self.augment_fraction = augment_fraction
        self.smooth_base = smooth_base
        self.smooth_mean = smooth_mean
        self.smooth_std = smooth_std
        self.s = None

    def smooth(self, x):
        """
        returns a smoothly varying curve with length x
        """
        w = x
        w = self.smooth_base ** tf.range(w, dtype=tf.float32)
        w = tf.concat((w[:-1], tf.reverse(w, axis=(0,))), axis=0)
        w = w / tf.reduce_sum(w)
        w = tf.reshape(w, (-1, 1, 1))
        x = x + w.shape[0] - 1
        x = tf.random.truncated_normal(shape=(1, x, 1), mean=self.smooth_mean, stddev=self.smooth_std)
        x = tf.nn.convolution(x, w)
        x = tf.reshape(x, (-1,))
        return x

    @abstractmethod
    def operation(self, x, s, rng):
        ...

    def call(self, x, training=True):
        probs = self.augment_fraction if training else 0.0
        if probs == 0.0:
            return x
        mask = tf.random.normal(shape=(self.s.batch_size, 1, 1, 1, 1)) < 0.0
#         mask = tf.random.binomial(shape=(self.s.batch_size, 1, 1, 1, 1), counts=1., probs=probs)
        x_augmented = self.operation(x, self.s, self.rng)
        return tf.where(mask, x, x_augmented)

class VideoRandomNoise(VideoRandomAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def operation(self, x, s, rng):
        frame_count = tf.reshape(tf.range(s.frame_count), (s.frame_count, 1, 1))
        frame_count = tf.where(frame_count < 2, 1.0, 0.0)
        height = tf.reshape(tf.range(s.height), (1, s.height, 1))
        height = tf.where(height < 3, 8.0, 0.0) + tf.where(height > 100, 1.0, 0.0)
        width = tf.reshape(tf.range(s.width), (1, 1, s.width))
        width = tf.where(width < 3, 8.0, 0.0) + tf.where(width > 200, 1.0, 0.0)
        w = tf.random.normal((s.channels, s.batch_size, s.frame_count, s.height, s.width))
        w = width * height * frame_count * w
        w = tf.signal.ifft3d(tf.complex(w, w))
        w = tf.math.real(w)
        w = tf.transpose(w, (1, 2, 3, 4, 0))
        return x + w * 2e2

class VideoRandomFlip(VideoRandomAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def operation(self, x, s, rng):
        return tf.reverse(x, axis=(3,))

class VideoRandomContrast(VideoRandomAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, smooth_base=1.1, smooth_mean=1., smooth_std=1.5, **kwargs)
    def operation(self, x, s, rng):
        mask = self.smooth(s.batch_size * s.frame_count)
        mask = tf.reshape(mask, (s.batch_size, s.frame_count, 1, 1, 1))
        x = (x - 0.5) * mask + 0.5
        return x

class VideoRandomMultiply(VideoRandomAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, smooth_base=1.1, smooth_mean=1., smooth_std=1.5, **kwargs)
    def operation(self, x, s, rng):
        mask = self.smooth(s.batch_size * s.frame_count)
        mask = tf.reshape(mask, (s.batch_size, s.frame_count, 1, 1, 1))
        x = x * mask
        return x

class VideoRandomAdd(VideoRandomAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, smooth_base=1.1, smooth_mean=0., smooth_std=0.5, **kwargs)
    def operation(self, x, s, rng):
        mask = self.smooth(s.batch_size * s.frame_count)
        mask = tf.reshape(mask, (s.batch_size, s.frame_count, 1, 1, 1))
        x = x + mask
        return x

class VideoRandomPerspective(VideoRandomAugmentation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, smooth_base=1.1, smooth_mean=0., smooth_std=5.0, **kwargs)
        corners = {(0, 0): (-1, 0), (0, 1): (0, 2), (1, 0): (1, -1), (1, 1): (2, 1)}
        src = list(corners)
        warp_basis = []
        for corner, tweak in corners.items():
            dst = [tweak if s == corner else s for s in src]
            tform = transform.ProjectiveTransform()
            success = tform.estimate(np.array(dst) - 0.5, np.array(src) - 0.5, )
            assert success
            warp_basis += [tform.params]
        warp_basis = tf.stack(warp_basis)
        warp_basis = tf.cast(warp_basis, tf.float32)
        warp_basis = tf.reshape(warp_basis, (4, 1, 3, 3))
        self.warp_basis = warp_basis

    @staticmethod
    def scale(sr: tf.float32, sc: tf.float32):
        return tf.convert_to_tensor([[[sc, 0, 0], [0, sr, 0], [0, 0, 1]]], dtype=tf.float32)

    @staticmethod
    def translation(tr: tf.float32, tc: tf.float32):
        return tf.convert_to_tensor([[[1, 0, tc], [0, 1, tr], [0, 0, 1]]], dtype=tf.float32)

    def operation(self, x, s, rng):
        l = s.batch_size * s.frame_count
        warp = [self.smooth(l), self.smooth(l), self.smooth(l), self.smooth(l)]
        warp = tf.reshape(warp, (4, l, 1, 1))
        warp = tf.math.maximum(warp, 0.0)
        warp = warp * self.warp_basis + (1 - warp) * tf.eye(3)
        warp = tf.reduce_sum(warp, axis=0)
        warp = (
            self.scale(s.height-1, s.width-1) 
            @ self.translation(0.5, 0.5)
            @ warp
            @ self.translation(-0.5, -0.5)
            @ self.scale(1/(s.height-1), 1/(s.width-1))
        )
        warp = tf.reshape(warp, (l, 3 * 3))
        warp = warp[:, :8] / warp[:, 8:]
        x = tf.reshape(x, (l, s.height, s.width, s.channels))
        x = keras_cv.src.utils.preprocessing.transform(x, warp, fill_mode='constant')
        x = tf.reshape(x, s.shape)
        return x
