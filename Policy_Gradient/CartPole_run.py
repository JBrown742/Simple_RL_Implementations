import random
import gym 
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam

env = gym.make('CartPole-v1')
best_model = tf.keras.models.load_model("Models/PG/PGAgent-v2-best")
# Has an average over 10 runs of 300.
avg_reward_array = np.zeros(5)
epsilon=1
for i in range(8):
	model = tf.keras.models.load_model("Models/PG/PGAgent-v2-episode{}".format(i*50))
	for k in range(5):
		state = env.reset()
		reward_total=0
		for j in range(500):
			env.render(mode='human')
			prob = model(state[np.newaxis])
# 			a = np.argmax(prob[0].numpy())
			a = np.random.choice(np.array([0,1]), p = prob[0].numpy())
			state, reward, done, _ = env.step(a)
			reward_total+=reward
			if done:
				break  
		env.close()
		avg_reward_array[k]=reward_total
	
	print(np.mean(avg_reward_array))	