#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# Epochs
#epochs = 20000
epochs = 1500
#epochs = 100

# Data
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
num_attributes = x_data.shape[1]
#print(num_attributes)

X = tf.placeholder(tf.float32, shape=[None, num_attributes])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W_h1 = tf.Variable(tf.random_normal([num_attributes, num_attributes * 2]))
b_h1 = tf.Variable(tf.random_normal([num_attributes * 2]))
#H1 = tf.sigmoid(tf.matmul(X, W_h1) + b_h1)
H1 = tf.nn.relu(tf.matmul(X, W_h1) + b_h1)

W_h2 = tf.Variable(tf.random_normal([num_attributes * 2, num_attributes * 4]))
b_h2 = tf.Variable(tf.random_normal([num_attributes * 4]))
#H2 = tf.sigmoid(tf.matmul(H1, W_h2) + b_h2)
H2 = tf.nn.relu(tf.matmul(H1, W_h2) + b_h2)

W_h3 = tf.Variable(tf.random_normal([num_attributes * 4, num_attributes * 8]))
b_h3 = tf.Variable(tf.random_normal([num_attributes * 8]))
#H3 = tf.sigmoid(tf.matmul(H2, W_h3) + b_h3)
H3 = tf.nn.relu(tf.matmul(H2, W_h3) + b_h3)

W_h4 = tf.Variable(tf.random_normal([num_attributes * 8, num_attributes * 4]))
b_h4 = tf.Variable(tf.random_normal([num_attributes * 4]))
#H4 = tf.sigmoid(tf.matmul(H3, W_h4) + b_h4)
H4 = tf.nn.relu(tf.matmul(H3, W_h4) + b_h4)

W_h5 = tf.Variable(tf.random_normal([num_attributes * 4, num_attributes * 2]))
b_h5 = tf.Variable(tf.random_normal([num_attributes * 2]))
#H5 = tf.sigmoid(tf.matmul(H4, W_h5) + b_h5)
H5 = tf.nn.relu(tf.matmul(H4, W_h5) + b_h5)

#W = tf.Variable(tf.random_normal([num_attributes, 1]), name='W')
#b = tf.Variable(tf.random_normal([1]), name='b')
W_o = tf.Variable(tf.random_normal([num_attributes * 2, 1]))
b_o = tf.Variable(tf.random_normal([1]))


#model = tf.sigmoid(tf.matmul(H5, W_o) + b_o)
model  = tf.matmul(H5, W_o) + b_o
#cost = tf.reduce_mean(-Y * tf.log(model) - (1 - Y) * tf.log(1 - model))
# cost 계산 수식을 이미 만들어진 함수를 사용
cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=Y)

#train = tf.train.GradientDescentOptimizer(0.01).minimize(cost);
train = tf.train.AdamOptimizer(0.01).minimize(cost);

predict = tf.cast(model > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=tf.float32))

# Session
with tf.Session() as sess:
	# Training
	sess.run(tf.global_variables_initializer())
	for step in range(epochs+1):
		cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
		print(step, cost_val)
		#if step % 2000 == 0:
		#	print(step, cost_val)
	# Testing
	m, p, a = sess.run([model, predict, accuracy], feed_dict={X: x_data, Y: y_data})
	print(m,p,a)

