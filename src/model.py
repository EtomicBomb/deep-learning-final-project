import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Softmax
# dataset related imports
from main import get_dataset
import json
from pathlib import Path



class Model(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.hidden_size1 = None
        self.hidden_size2 = None
        self.hidden_size3 = None
        self.num_classes = num_classes
    # layers here

        self.cnn = Sequential([
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


model = Model(num_classes=3) # classifying 3 levels of drowsiness: low, med, high

# need to change the values for optimizer, loss & metrics -- copied from documentation
model.compile(
optimizer=keras.optimizers.Adam(learning_rate=1e-3),
loss='binary_cross_entropy',
metrics=['accuracy'],
)

model.summary()

splits = json.loads(Path('data/split.json').read_text())
dataset = get_dataset(splits['train'][:4])
model.fit(dataset, epochs=10)

test_loss, test_acc = model.evaluate(dataset)


