# -*- coding: utf -*-

# keras 테스트 코드

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Data
x_train = np.array([[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Model
model = Sequential()
model.add(Dense(1, input_dim = 2, activation='sigmoid'))

# Learning Methos
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# Training
model.fit(x_train, y_train, epochs=10000, verbose=1) # verbose, 1: show the training
model.summary()
print(model.get_weights()) # print weight values

# Evaluation
y_predict = model.predict(x_train)
print(y_predict)

