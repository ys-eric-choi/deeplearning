# -*- coding: utf-8 -*-
# Tensorflow 를 이용한 RNN 구현 테스트 예제-2

import tensorflow as tf
import numpy as np

epoch = 5000

sentence = " My name is Yunsu Choi."
idx2char = list(set(sentence))
print(idx2char)

char2idx = {c: i for i, c in enumerate(idx2char)}
print(char2idx)

input_dim = len(idx2char)
hidden_size = len(idx2char)
num_classes = len(idx2char)
batch_size = 1
seq_length = len(sentence) - 1
sentence_idx = [char2idx[c] for c in sentence]
print(sentence_idx)

x_data = [sentence_idx[:-1]]
y_data = [sentence_idx[1:]]

X = tf.placeholder(tf.int32, [None, seq_length])
Y = tf.placeholder(tf.int32, [None, seq_length])
x_onehot = tf.one_hot(X, num_classes)

# Model, Cost, Train
cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
model, _ = tf.nn.dynamic_rnn(cell, x_onehot, initial_state=initial_state, dtype=tf.float32)
modle = tf.reshape(model, [batch_size, seq_length, num_classes])
weights = tf.ones([batch_size, seq_length])
cost = tf.reduce_mean(tf.contrib.seq2seq.sequence_loss(logits=model, targets=Y, weights=weights))
train = tf.train.AdamOptimizer(0.1).minimize(cost)

pred = tf.argmax(model, axis=2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(epoch):
		c, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
		result = sess.run(pred, feed_dict={X: x_data})
		print(i, "cost: ", c, "pred: ", result)
		result_str = [idx2char[c] for c in np.squeeze(result)]
		print("Prediction String: ", ''.join(result_str))

