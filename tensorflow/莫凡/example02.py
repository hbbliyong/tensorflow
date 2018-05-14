import tensorflow as tf
import numpy as np

#create data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

### create tensorflow structure start ###
#权重为一个-1到1的随机数组
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases=tf.Variable(tf.zeros([1]))

#预测的值
y=Weights*x_data+biases

#误差
loss=tf.reduce_mean(tf.square(y-y_data))
#优化器,参数为学习效率，小于1的数字
optimizer=tf.train.GradientDescentOptimizer(0.5)
#用优化器减少误差
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
### create tensorflow structure end ###

sess=tf.Session()
sess.run(init) #Very important

#开始训练神经网络
for step in range(201):
    sess.run(train)
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))
