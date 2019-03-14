from keras.datasets import mnist
import tensorflow as tf
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print(train_images.shape)

a = tf.zeros((32, 10))
b = tf.zeros(10,)

x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))

z = np.maximum(x, y)

print(z.shape)




