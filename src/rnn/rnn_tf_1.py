# -*- coding: utf-8 -*-
# Tensorflow 를 이용한 RNN 구현 테스트 예제-1

import tensorflow as tf
import numpy as np

epoch = 200

idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]] # hihell
y_data = [[1, 0, 2, 3, 3, 4]] # ihello
x_onehot = [[[1,0,0,0,0],
			 [0,1,0,0,0],
			 [1,0,0,0,0],
			 [0,0,1,0,0],
			 [0,0,0,1,0],
			 [0,0,0,1,0]]]

num_classes = 5
input_dim = 5
hidden_size = 5
batch_size = 1
seq_length = 6

X = tf.placeholder(tf.float32, [None, seq_length, input_dim])
Y = tf.placeholder(tf.int32, [None, seq_length])

# Model, Cost, Train
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
model, _ = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
model = tf.reshape(model, [batch_size, seq_length, num_classes])
weights = tf.ones([batch_size, seq_length])
cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=model, targets=Y, weights=weights))
train = tf.train.AdamOptimizer(0.1).minimize(cost)

pred = tf.argmax(model, axis=2)


fWrite = open("Test1.result", "a")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(epoch):
		c, _ = sess.run([cost, train], feed_dict={X: x_onehot, Y: y_data})
		result = sess.run(pred, feed_dict={X: x_onehot})
		print(i, "cost: ", c, "pred: ", result)
		result_str = [idx2char[c] for c in np.squeeze(result)]
		print("Prediction String: ", ''.join(result_str))
		
		if i == epoch - 1:
			for j, w in enumerate(result_str):
				fWrite.write(w)
			fWrite.write("\n")
			fWrite.flush()

fWrite.close()



