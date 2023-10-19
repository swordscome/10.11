# 在tensorflow框架下编写python代码，创建一个简单的神经网络，并预测结果。
# (一)导入tensorflow模块（8分）
import numpy as np
import tensorflow as tf
# (二)准备数据集（共12分，各6分）
x_data = [[0, 0, 1, 0],
          [1, 1, 1, 1],
          [1, 0, 1, 1],
          [0, 1, 1, 0]]
y_data = [[0],
          [1],
          [1],
          [0]]
print(np.array(x_data).shape)
print(np.array(y_data).shape)

# (三)定义变量X（8分）和Y（8分）
X = tf.placeholder(dtype=tf.float32,shape=[None,4])
Y = tf.placeholder(dtype=tf.float32,shape=[None,1])
# (四)定义预测值，可用sigmoid函数（8分）
w1 = tf.Variable(tf.random_normal(shape=[4,128]))
b1 = tf.Variable(tf.random_normal(shape=[128]))
h1 = tf.sigmoid(tf.matmul(X,w1)+b1)

w2 = tf.Variable(tf.random_normal(shape=[128,1]))
b2 = tf.Variable(tf.random_normal(shape=[1]))
h = tf.sigmoid(tf.matmul(h1,w2)+b2)

hh = tf.cast(h>0.5,dtype=tf.int32)
# (五)定义代价函数（8分）
loss = -tf.reduce_mean(Y*tf.log(h)+(1-Y)*tf.log(1-h))
# (六)使用梯度下降优化器（8分）
op = tf.train.AdamOptimizer(0.01).minimize(loss)
# (七)创建会话（8分）
with tf.Session() as sess:
# (八)全局变量初始化（8分）
    sess.run(tf.global_variables_initializer())
# (九)迭代10001次，每500次输出一次代价值（8分）
    for i in range(10001):
        op_,loss_ = sess.run([op,loss],feed_dict={X:x_data,Y:y_data})
        if i % 500 == 0:
            print(i,loss_)
# (十)对数据[1,0,0,0]进行预测，输出结果。（8分）
    print(sess.run(hh,feed_dict={X:[[1,0,0,0]]}))
# (十一)对数据[1,1,1,1]进行预测，输出结果。（8分）
    print(sess.run(hh,feed_dict={X:[[1,1,1,1]]}))