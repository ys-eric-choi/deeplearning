import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Epochs
#epochs = 1000
epochs = 500

# Data
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
num_attributes = x_data.shape[1]
#print(num_attributes)

# Model
model = Sequential()
#model.add(Dense(16, input_dim = num_attributes, activation='sigmoid'))
#model.add(Dense(8, input_dim = num_attributes, activation='relu'))
#model.add(Dense(8, input_dim = 8, activation='sigmoid'))
#model.add(Dense(8, input_dim = 16, activation='sigmoid'))
#model.add(Dense(8, input_dim = 16, activation='relu'))
#model.add(Dense(1, input_dim = 8, activation='sigmoid'))
model.add(Dense(num_attributes * 2, activation='relu'))
model.add(Dense(num_attributes * 4, activation='relu'))
model.add(Dense(num_attributes * 8, activation='relu'))
model.add(Dense(num_attributes * 4, activation='relu'))
model.add(Dense(num_attributes * 2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Learning Methods
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(x_data, y_data, epochs=epochs, verbose=1)
model.summary()
