#莫凡教程

## 1.第一课
### 目标
目标：输出参数训练参数
学习的目标是：
f(x)=x*01+0.3
### 地址
[地址](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/)

## 2.第二课:variable
注意点：
1.如果定义变量,一定要调用init= tf.global_variables_initializer()进行初始化，然后sess.run(init)进行激活
2.定义的变量不能直接输出，一定要把 sess 的指针指向 变量 再进行 print 才能得到想要的结果！
## 3.第三课：placehold
placeholder 是 Tensorflow 中的占位符，暂时储存变量.

Tensorflow 如果想要从外部传入data, 那就需要用到 tf.placeholder(), 然后以这种形式传输数据 sess.run(***, feed_dict={input: **}).

```python

import tensorflow as tf

#在 Tensorflow 中需要定义 placeholder 的 type ，一般为 float32 形式
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul = multiply 是将input1和input2 做乘法运算，并输出为 output 
ouput = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
# [ 14.]    
```
## 4.第四课 ：激励函数(activation function)
>激励函数运行时激活神经网络中某一部分神经元，将激活信息向后传入下一层的神经系统。激励函数的实质是非线性方程。 Tensorflow 的神经网络 里面处理较为复杂的问题时都会需要运用激励函数 activation function 