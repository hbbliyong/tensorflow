import tensorflow as tf

#定义一个变量
state=tf.Variable(0,name='counter')
print(state.name)
#定义一个常量
one=tf.constant(1)
# 定义加法步骤 (注: 此步并没有直接计算)
new_value=tf.add(state,one)
# 将 State 更新成 new_value
update=tf.assign(state,new_value)

# must have if define variable
#init =tf.initialize_all_variables() #将要被废止
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
