import tensorflow_graphics.image.transformer as tfg
from skimage import transform
import tensorflow as tf
from tensorflow import keras
import numpy as np

class VideoRandomOperation(keras.layers.Layer):
    def __init__(self, rng=None, *args, **kwargs):
        super().__init__(trainable=False, *args, **kwargs)
        if rng is None:
            rng = tf.random.Generator.from_non_deterministic_state()
        self.rng = rng
    def smooth_exponential(self, x, base, mean, stddev):
        """
        Ran
        x: length 
        w: size of the window (convolution window has length 2 * w - 1)
        base: larger means sharper exponential
        """
        w = x
        w = base ** tf.range(w, dtype=tf.float32)
        w = tf.concat((w[:-1], tf.reverse(w, axis=(0,))), axis=0)
        w = w / tf.reduce_sum(w)
        w = tf.reshape(w, (-1, 1, 1))
        x = x + w.shape[0] - 1
        x = self.rng.truncated_normal(shape=(1, x, 1), mean=mean, stddev=stddev)
        x = tf.nn.convolution(x, w)
        x = tf.reshape(x, (-1,))
        return x
    def operation(self, x, rng):
        pass
    def call(self, x, training=True):
        return self.operation(x, self.rng) if training else x

class VideoRandomFlip(VideoRandomOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def operation(self, x, rng):
        batch_size, frame_count, height, width, channels = x.shape
        mask = rng.binomial(shape=(batch_size,), counts=1., probs=0.5)
        mask = tf.reshape(mask, (batch_size, 1, 1, 1, 1))
        return tf.where(mask == 1, x, tf.reverse(x, axis=(3,)))

class VideoRandomContrast(VideoRandomOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def operation(self, x, rng):
        batch_size, frame_count, height, width, channels = x.shape
        mask = self.smooth_exponential(x=batch_size * frame_count, base=1.1, mean=1., stddev=1.5)
        mask = tf.reshape(mask, (batch_size, frame_count, 1, 1, 1))
        x = (x - 0.5) * mask + 0.5
        return tf.clip_by_value(x, 0.0, 1.0)

class VideoRandomBrightness(VideoRandomOperation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def operation(self, x, rng):
        batch_size, frame_count, height, width, channels = x.shape
        mask = self.smooth_exponential(x=batch_size * frame_count, base=1.1, mean=1., stddev=1.5)
        mask = tf.reshape(mask, (batch_size, frame_count, 1, 1, 1))
        x = x * mask
        return tf.clip_by_value(x, 0.0, 1.0)

class VideoRandomPerspective(VideoRandomOperation):
    def __init__(self):
        super().__init__()
        corners = {(0, 0): (-1, 0), (0, 1): (0, 2), (1, 0): (1, -1), (1, 1): (2, 1)}
        src = list(corners)
        warp_basis = []
        for corner, tweak in corners.items():
            dst = [tweak if s == corner else s for s in src]
            tform = transform.ProjectiveTransform()
            success = tform.estimate(np.array(dst) - 0.5, np.array(src) - 0.5, )
            assert success
            warp_basis += [tform.params]
        warp_basis = tf.convert_to_tensor(warp_basis, dtype=np.float32)
        warp_basis = tf.reshape(warp_basis, (4, 1, 3, 3))
        self.warp_basis = warp_basis

    @staticmethod
    def scale(sr: tf.float32, sc: tf.float32):
        return tf.convert_to_tensor([[[sc, 0, 0], [0, sr, 0], [0, 0, 1]]], dtype=tf.float32)

    @staticmethod
    def translation(tr: tf.float32, tc: tf.float32):
        return tf.convert_to_tensor([[[1, 0, tc], [0, 1, tr], [0, 0, 1]]], dtype=tf.float32)

    def operation(self, x, rng):
        batch_size, frame_count, height, width, channels = x.shape
        length = batch_size * frame_count
        a = self.smooth_exponential(x=length, base=1.1, mean=0., stddev=5.0)
        b = self.smooth_exponential(x=length, base=1.1, mean=0., stddev=5.0)
        c = self.smooth_exponential(x=length, base=1.1, mean=0., stddev=5.0)
        d = self.smooth_exponential(x=length, base=1.1, mean=0., stddev=5.0)
        warp = tf.reshape([a, b, c, d], (4, length, 1, 1))
        warp = tf.math.maximum(warp, 0.0)
        warp = warp * self.warp_basis + (1 - warp) * tf.eye(3)
        warp = tf.reduce_sum(warp, axis=0)
        warp = (
            self.scale(height-1, width-1) 
            @ self.translation(0.5, 0.5)
            @ warp
            @ self.translation(-0.5, -0.5)
            @ self.scale(1/(height-1), 1/(width-1))
        )
        x = tf.reshape(x, (length, height, width, channels))
        x = tfg.perspective_transform(x, warp)
        x = tf.reshape(x, (batch_size, frame_count, height, width, channels))
        return x

