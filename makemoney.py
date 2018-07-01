import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
prices = datasets.load_boston()

# Use only one feature
prices_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
prices_X_train = prices_X[:-20]
prices_X_test = prices_X[-20:]

# Split the targets into training/testing sets
prices_y_train = prices.target[:-20]
prices_y_test = prices.target[-20:]

# Create linear regression object
regr = linear_model.SupportVectorMachine()

# Train the model using the training sets
regr.fit(prices_X_train, prices_y_train)

# Make predictions using the testing set
prices_y_pred = regr.predict(prices_X_test)

from keras.models import Sequential
from keras.layers import Dense
from sklearn import datasets,
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load pima price dataset
dataset = datasets.load_boston()
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

import gym
env = gym.make('sairen-v0')
for i_episode in xrange(20):
    observation = env.reset()
    for t in xrange(100):
        env.render()
        print observation
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break
