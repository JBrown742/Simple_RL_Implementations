
import warnings
warnings.filterwarnings('ignore')
import random
import gym 
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam

env = gym.make('Pendulum-v0').unwrapped
best_model = tf.keras.models.load_model("Models/DDPG-Actor-Best")
# Has an average over 10 runs of 300.
avg_reward_array = np.zeros(1)

for i in range(3):
	model = tf.keras.models.load_model("Models/DDPG-Actor-v0-episode-{}".format(i*25))
	for k in range(1):
		state = env.reset()
		reward_total=0
		epsilon=0.01
		for j in range(1000):
			env.render(mode='human')
			if np.random.uniform(0, 1) > epsilon:
				a = 2 * model(state.reshape((1,3)))[0]
			else:
				#a = np.clip(2*best_model(state.reshape((1,3)))[0] + np.random.normal(loc=0.0, scale=0.2),-2,2)
				a = np.random.uniform(-2,2, size=(1,))
			state, reward, done, _ = env.step(a)
			done=False
			reward_total+=reward
			if done:
				break  
		env.close()
		avg_reward_array[k]=reward_total
	
	print(np.mean(avg_reward_array))	