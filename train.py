from datetime import datetime, timezone, timedelta

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard


def main():
    # load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # one-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # create a model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # 32 3x3 filters, ReLU activation
        MaxPooling2D((2, 2)),  # 2x2 max pooling
        Conv2D(64, (3, 3), activation='relu'),  # 64 3x3 filters, ReLU activation
        MaxPooling2D((2, 2)),  # 2x2 max pooling
        Flatten(),
        Dense(128, activation='relu'),  # 128 neurons, ReLU activation
        Dense(10, activation='softmax')  # 10 neurons, softmax activation
    ])

    # compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # create a TensorBoard callback
    now_time = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    log_dir = f'logs/fit/{now_time}'
    tensor_board = TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(x_train, y_train, epochs=5, callbacks=[tensor_board])
    model.save(f'models/model-{now_time}.keras')


if __name__ == '__main__':
    main()
