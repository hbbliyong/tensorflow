import tensorflow as tf
import numpy as np
import matplotlib as plt

def add_layer(inputs, in_size,out_size,activation_function=None):
    #因为在生成初始参数时，随机变量(normal distribution)会比全部为0要好很多，所以我们这里的weights为一个in_size行, out_size列的随机变量矩阵。
    Weights=tf.Variable(tf.random_uniform([in_size,out_size]))
    #在机器学习中，biases的推荐值不为0，所以我们这里是在0向量的基础上又加了0.1。
    biases=tf.Variable(tf.zeros([1,out_size]))+0.1
    #神经网络未激活的值
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if(activation_function is None):
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs       


x_data=np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
#加入噪点，使数据更像真实数据
noise=np.random.normal(0,0.05,x_data.shape).astype(np.float32)
y_data=np.square(x_data)-0.5+noise

xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#我们开始定义隐藏层,利用之前的add_layer()函数，这里使用 Tensorflow 自带的激励函数tf.nn.relu。
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
#我们构建的是——输入层1个、隐藏层10个、输出层1个的神经网络
prediction=add_layer(l1,10,1,activation_function=None)
#计算预测值prediction和真实值的误差，对二者差的平方求和再取平均
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),eduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#生成图片框
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.show()

for i in range(10000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if(i%500==0):
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
