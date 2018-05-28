from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("MNIST_data",one_hot=True)

print(mnist.train.images.shape)
print(mnist.train.labels.shape)

image=mnist.train.images[1,:]
image=image.reshape(28,28)

print(mnist.train.labels[1])

plt.figure()
plt.imshow(image)
plt.show()