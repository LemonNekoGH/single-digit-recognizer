import sys
import tensorflow as tf

from tensorflow.keras.models import load_model

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: predict.py <model-file> <image-file>')
        sys.exit(1)

    model_file = sys.argv[1]
    image_file = sys.argv[2]

    model = load_model(model_file)
    image = tf.io.read_file(image_file)  # read image file
    image = tf.image.decode_image(image, channels=1)  # grayscale image
    image = tf.image.resize(image, (28, 28))  # resize to 28x28 pixels
    image = tf.cast(image, tf.float32) / 255.0  # normalize pixel values in the range [0, 1]
    image = tf.expand_dims(image, 0)  # add extra dimension, https://www.tensorflow.org/api_docs/python/tf/expand_dims

    prediction = model.predict(image)

    print(f'Predicted digit: {tf.argmax(prediction, axis=1).numpy()[0]}')
