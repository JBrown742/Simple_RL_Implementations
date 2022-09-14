#!/usr/bin/env python
# coding: utf-8

# In[73]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Rescaling, BatchNormalization, GaussianNoise, LSTM, GRU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import math


"""
Build a class for the actor and critic agents
"""
class MADDPG:
	"""
	Define a class that returns the actor and critic (as well as duplicate targets) for a single
	agent within the MADDPG framework. Here we assume each agent will control a single parameter
	"""
	def __init__(self, observation_sizes, n_actions, gamma=0.97, actor_structs=[64,64], critic_structs=[256,32],
						agent_names='default'):
		self.gamma = gamma
		self.observation_sizes = observation_sizes
		self.n_agents = len(observation_sizes)
		self.critic_input_size = sum(observation_sizes) + sum(n_actions)

		if agent_names=='default':
			self.agent_names = ['agent_{}'.format(i) for i in range(self.n_agents)]
		else:
			self.agent_names = agent_names

		self.actor_structs = actor_structs
		self.critic_structs = critic_structs

		self.n_actions = n_actions


	def build_actor(self, obs_size, struct, num_actions):
		"""
		Function for initializing and building the main and target actor networks for a single agent.

		Parameters
		----------
		observation_shape: (Tuple) Of type (None, observation_legnth). None here denoted a yet undefined
								   quantity, i.e 1 for single actions, and batch_size when training
		num_actions: (int) Number of parameters the agent is responsible for. Default is 1. Each
						   agent is responsible for one single control parameter.
		hidden_arch: (list) Defines both the number of nodes, and number of hidden layers. Layers are Dense
							by default.

		Returns
		-------
		actor: (keras.model object) The network to be used as the actor withing the agent.
		actor_: (keras.model object) Identical copy of actor to be used as the agents actor target network.
		"""
		obs_shape = (None, obs_size)
		# observation shape should be (1,obs_size) or (batch_size, obs_size)
		actor = Sequential()
		actor.add(LayerNormalization())
		for h in struct:
			actor.add(Dense(h, activation="relu"))
		#actor.add(GRU(64))
		actor.add(Dense(64, activation="relu"))
		actor.add(Dense(num_actions, activation="tanh"))
		actor.build(input_shape=obs_shape)
		actor_ = tf.keras.models.clone_model(actor)
		actor_.set_weights(actor.get_weights())
		return actor, actor_

	def build_critic(self, struct):
		"""
		Function for initializing and building the main and target critic networks for a single agent.

		Parameters
		----------
		observation_shape: (Tuple) Of type (None, observation_legnth). None here denoted a yet undefined
								   quantity, i.e 1 for single actions, and batch_size when training
		num_actions: (int) Number of parameters the agent is responsible for. Default is 1. Each
						   agent is responsible for one single control parameter.
		hidden_arch: (list) Defines both the number of nodes, and number of hidden layers. Layers are Dense
							by default.

		Returns
		-------
		critic: (keras.model object) The network to be used as the actor withing the agent.
		critic_: (keras.model object) Identical copy of actor to be used as the agents actor target network.
		"""
		obs_shape = (None,self.critic_input_size)
		critic = Sequential()
		for h in struct:
			critic.add(Dense(h, activation="relu"))
		# critic.add(GRU(64))
		critic.add(Dense(64,  activation="relu"))
		critic.add(Dense(1, activation=None))
		critic.build(input_shape=obs_shape)
		critic_ = tf.keras.models.clone_model(critic)
		critic_.set_weights(critic.get_weights())
		return critic, critic_

	def init_agent_networks(self):
		self.actors = {}
		self.actor_targets = {}
		self.critics = {}
		self.critic_targets = {}
		for i, name in enumerate(self.agent_names):
			actor, actor_ = self.build_actor(obs_size=self.observation_sizes[i], struct=self.actor_structs,
											num_actions=self.n_actions[i])
			critic, critic_ = self.build_critic(struct=self.critic_structs)
			self.actors[name] = actor
			self.actor_targets[name] = actor_
			self.critics[name] = critic
			self.critic_targets[name] = critic_
		return

	def action_noise(self, action, scale):
		return np.clip(action + np.random.normal(loc=0.0, scale=scale, size=action.shape), a_min=0, a_max=1).astype(np.float32)

	def get_action_async(self, observation, agent_name, noise=False, scale=0.01):
		if not noise:
			action = (self.actors[agent_name](observation[np.newaxis]).numpy() + 1)/2
		else:
			actor_output = (self.actors[agent_name](observation[np.newaxis]).numpy() + 1)/2
			action = self.action_noise(actor_output, scale)
		return action[0]

	def get_action_sync(self, observations, noise=False, scale=0.1):
		actions = []
		if not noise:
			for i, name in enumerate(self.agent_names):
				actions.append(self.actors[name](observations[i].reshape((1,self.observation_sizes[i]))).numpy())
		else:
			for i, name in enumerate(self.agent_names):
				actor_output = self.actors[name](observations[i].reshape((1, self.observation_sizes[i])))
				noisy_action = self.action_noise(actor_output, scale)
				actions.append(noisy_action)
		return actions

	def train_critic(self, name, obs, actions, next_obs, rewards,
						dones,	optimizer=Adam(learning_rate=0.001, epsilon=1e-4)):
		# get the critic target and main critic networks corresponding to index
		critic_target_net = self.critic_targets[name]
		critic_net = self.critics[name]
		batch_size = rewards[name].shape[0]
		next_state = list(next_obs.values()).copy()
		state = list(obs.values()).copy()
		# Initialize a tensor to get the target predicted next actions from next observations
		# as well as the main actor predicted current actions from current observations
		# to be used in the target and main critic networks
		targ_actions = []
		for tag in self.agent_names:
			next_ob = next_obs[tag]
			action_prime = self.actors[tag](next_ob)
			targ_actions.append(action_prime)
			next_state.append(action_prime)
			state.append(actions[tag])
		# First concatenate the list of actor observation arrays into one single array of shape
		# (batch_size, obs_len_1 + ..... + obs_len_N), then concatenate all actions to the end of this to
		# make a tensor of shape (batch_size, (obs_len_1 + .... + obs_len_N + num_acts_1 + .... + num_acts_N))
		concated_next_obs_and_targ_acts = tf.concat(next_state, axis=1)
		concated_obs_and_acts = tf.concat(state, axis=1)
		# Calculate the estimated Q values as per the target critic network on the next observations and actions
		estimated_target_Qs = critic_target_net(concated_next_obs_and_targ_acts)
		# obtain a batch of bellman targets with shape (batch_size,1)
		y = rewards[name]+ self.gamma * (1-dones[name]) * tf.squeeze(estimated_target_Qs)
		with tf.GradientTape() as tape:
			current_predicted_Qs = tf.squeeze(critic_net(concated_obs_and_acts))
			loss = mse(y,current_predicted_Qs)
		grads = tape.gradient(loss, critic_net.trainable_variables)
		optimizer.apply_gradients(zip(grads, critic_net.trainable_variables))

	def train_actor(self, name, obs, rewards, dones,
						optimizer=Adam(learning_rate=0.005, epsilon=1e-5)):
		# Get a the batch of observations corresponding to agent index
		# Note obs will be a list of batches
		learning_actor = self.actors[name]
		critic = self.critics[name]
		state = list(obs.values()).copy()
		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(learning_actor.trainable_variables)
			for tag in self.agent_names:
				action = self.actors[tag](obs[tag])
				state.append(action)
			concat_obs_and_acts = tf.concat(state, axis=1)
			Q_values = critic(concat_obs_and_acts)
			loss = 1 / tf.reduce_mean(Q_values)
		grads = tape.gradient(loss, learning_actor.trainable_variables)
		optimizer.apply_gradients(zip(grads, learning_actor.trainable_variables))

	def soft_update(self, tau = 0.001):
		for agent_idx in self.agent_names:
			current_crit_targ_weights = self.critic_targets[agent_idx].get_weights()
			current_crit_weights = self.critics[agent_idx].get_weights()
			current_act_targ_weights = self.actor_targets[agent_idx].get_weights()
			current_act_weights = self.actors[agent_idx].get_weights()

			new_act_targ_weights = []
			for i in range(len(current_act_targ_weights)):
				new_act_targ_weights.append(tau * current_act_weights[i] + (1 - tau) * current_act_targ_weights[i])
			new_crit_targ_weights = []
			for i in range(len(current_crit_targ_weights)):
				new_crit_targ_weights.append(tau * current_crit_weights[i] + (1 - tau) * current_crit_targ_weights[i])
			self.critic_targets[agent_idx].set_weights(new_crit_targ_weights)
			self.actor_targets[agent_idx].set_weights(new_act_targ_weights)

	def sigma_scheduler(self,episode, num_episodes, init_sigma=0.2, final_sigma=0.01):
		half_life = num_episodes/4
		sigma = max(init_sigma*(0.5)**(episode/half_life), final_sigma)
		return sigma

#	  def train_critic(self, critic_main, critic_targ, actor_main, actor_targ):


class MultiAgentReplayBuffer:
	def __init__(self, max_size, observation_sizes, n_actions, batch_size, agent_names=None):
		# observations sizes is a list of observation_lengths for each agent ****Careful of ordering
		self.n_agents = len(observation_sizes)
		if agent_names==None:
			self.agent_tags = ['agent_{}'.format(i) for i in range(self.n_agents)]
		else:
			self.agent_tags = agent_names
		self.action_sizes = n_actions
		self.obs_sizes = observation_sizes
		# Max memory size
		self.mem_size = max_size
		# keeps track of how full the memory is
		self.mem_cntr = 0
		# Batch_size used in the learning process
		self.batch_size = batch_size

		# Each replay memory will be a dictionary keyed by the agent name, with the corresponding
		# memory array
		self.reward_memory = {}
		self.terminal_memory = {}
		self.obs_memory = {}
		self.next_obs_memory = {}
		self.action_memory = {}
		for i, size in enumerate(self.obs_sizes):
		# Will here use the standard default agent_i tags just to build the dict
			self.obs_memory[self.agent_tags[i]] = np.zeros((self.mem_size, size), dtype=np.float32)
			self.next_obs_memory[self.agent_tags[i]] = np.zeros((self.mem_size, size), dtype=np.float32)
			self.action_memory[self.agent_tags[i]] = np.zeros((self.mem_size, n_actions[i]), dtype=np.float32)
			self.reward_memory[self.agent_tags[i]] = np.zeros((self.mem_size,), dtype=np.float32)
			self.terminal_memory[self.agent_tags[i]] = np.zeros((self.mem_size,), dtype=bool)




	def store_transition(self, raw_obs, actions, rewards, next_raw_obs, dones):
		# states sould be a list of arrays for each agent, so that the state_memory structure with be
		# arrays embedded in a lists embedded in an array mem[elem_idx][agent_idx][obs_elem_idx]
		# Should turn actions and rewards and dones to lists  also so that each has the same structure
		# as the states

		# Each input to this function must be a list
		idx = self.mem_cntr % self.mem_size
		for i in range(self.n_agents):
			self.obs_memory[self.agent_tags[i]][idx,:] = raw_obs[i][:]
			self.next_obs_memory[self.agent_tags[i]][idx,:] = next_raw_obs[i][:]
			self.action_memory[self.agent_tags[i]][idx,:] = actions[i][:]
			self.reward_memory[self.agent_tags[i]][idx] = rewards[i]
			self.terminal_memory[self.agent_tags[i]][idx] = dones[i]

		self.mem_cntr += 1

	def store_transition_async(self, raw_obs, actions, reward, done, agent):
		# states sould be a list of arrays for each agent, so that the state_memory structure with be
		# arrays embedded in a lists embedded in an array mem[elem_idx][agent_idx][obs_elem_idx]
		# Should turn actions and rewards and dones to lists  also so that each has the same structure
		# as the states
		# Reinsert next_raw_obs if needed

		# Each input to this function must be a list
		idx = self.mem_cntr % self.mem_size
		self.obs_memory[agent][idx,:] = raw_obs
		#self.next_obs_memory[self.agent_tags[i]][idx,:] = next_raw_obs
		self.action_memory[agent][idx,:] = actions
		self.reward_memory[agent][idx] = reward
		self.terminal_memory[agent][idx] = done

		#self.mem_cntr += 1


	def sample_buffer(self):
		"""
		returns list of batches of observations one for each agent, list of batches of next observations
		one for each agent, batch of rewards, batch of actions, batch of done bools
		"""
		max_mem = min(self.mem_cntr, self.mem_size)

		batch = np.random.choice(max_mem, self.batch_size, replace=False)
		batch_obs = {}
		batch_next_obs = {}
		batch_actions = {}
		batch_rewards = {}
		batch_terminals = {}
		for i, agent_tag in enumerate(self.agent_tags):
			batch_obs[agent_tag] = self.obs_memory[agent_tag][batch,:]
			batch_next_obs[agent_tag] = self.next_obs_memory[agent_tag][batch, :]
			batch_actions[agent_tag] = self.action_memory[agent_tag][batch, :]
			batch_rewards[agent_tag] = self.reward_memory[agent_tag][batch]
			batch_terminals[agent_tag] = self.terminal_memory[agent_tag][batch]

		return batch_obs, batch_next_obs, batch_rewards, batch_actions, batch_terminals

	def ready(self):
		if self.mem_cntr>= self.batch_size*50:
			return True
		return False


# %%
