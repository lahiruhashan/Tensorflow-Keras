from keras.datasets import mnist
import tensorflow as tf
import numpy as np
from keras import layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# print(train_images.shape)

a = tf.zeros((32, 10))
b = tf.zeros(10,)

x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))

z = np.maximum(x, y)


l = x[0:, 0:, 0, 0]
print(l.shape)


# reshaping
n = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])

print(n.shape)
print(n.reshape(6, 1))
print(n.reshape(2, 3))
print(np.transpose(n))



