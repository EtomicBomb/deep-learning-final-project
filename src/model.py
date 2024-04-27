import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Softmax

# need to change the values for optimizer, loss & metrics -- copied from documentation

class Model(tf.keras.Model):
    def __init__(
        self, 
        batch_size: int, 
        frames_per_example: int,
        video_height: int, 
        video_width: int, 
        channels: int, 
        num_classes: int,
    ):
        super().__init__()
        self.hidden_size1 = 1
        self.hidden_size2 = 1
        self.hidden_size3 = 1
        self.num_classes = num_classes
    # layers here

        self.cnn = Sequential([
            keras.Input(shape=(frames_per_example, video_height, video_width, channels), batch_size=batch_size),
        # model here
            Conv2D(self.hidden_size1, kernel_size=(3,3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(self.hidden_size2, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(self.hidden_size3, activation=('relu')),
            Dense(units=self.num_classes, activation = 'softmax')
        ])
    def call(self, inputs, training=False):
        out = self.cnn(inputs)
        return out

if __name__ == '__main__':
    # dataset related imports
    from main import get_dataset
    import json
    from pathlib import Path

    batch_size = 4
    frames_per_example = 30 * 3
    video_height = 112
    video_width = 224
    model = Model(
        batch_size=batch_size,
        frames_per_example=frames_per_example,
        video_height=video_height,
        video_width=video_width,
        channels=1,
        num_classes=3,
    ) # classifying 3 levels of drowsiness: low, med, high
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_cross_entropy',
    metrics=['accuracy'],
    )

    model.summary()

    splits = json.loads(Path('data/split.json').read_text())
    dataset = splits['train'][:1]
    dataset = get_dataset(
        dataset, 
        frames_per_example, 
        shuffle_batch=1000, 
        video_height=video_height, 
        video_width=video_width,
    ).prefetch(4).batch(batch_size)
    model.fit(dataset, steps_per_epoch=10, epochs=10)

    test_loss, test_acc = model.evaluate(dataset)
