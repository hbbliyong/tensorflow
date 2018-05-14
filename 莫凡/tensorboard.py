import numpy as np
import tensorflow as tf


# 使用with tf.name_scope('inputs')可以将xs和ys包含进来，形成一个大的图层，图层的名字就是with tf.name_scope()方法里的参数。
with tf.name_scope('inputs'):
    # define placeholder for inputs to network
    xs = tf.placeholder(tf.float32, [None, 1], name='x_in')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_in')


def add_layer(inputs, in_size, out_size, activation_function=None):
   # define layer name
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal(
                [in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size])+0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


# 我们开始定义隐藏层,利用之前的add_layer()函数，这里使用 Tensorflow 自带的激励函数tf.nn.relu。
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# 我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络
prediction = add_layer(l1, 10, 1, activation_function=None)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(ys-prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter('logs/', sess.graph)

init = tf.global_variables_initializer()
sess.run(init)
