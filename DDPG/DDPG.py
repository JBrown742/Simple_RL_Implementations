#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, LayerNormalization
from tensorflow.keras import Sequential, Model
from tensorflow.keras.optimizers import Adam
import math
import gym
from collections import deque
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


class DDPG:
    
    def __init__(self, n_actions, state_size, buffer_size=10000, learning_batch_size=64, 
    act_struct = [32,32], crit_struct=[126,32]):
        
        self.num_actions = n_actions
        self.state_size = state_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = learning_batch_size
        
        # Actor Network
        # ctor takes the state as input and outputs num_actions. Here tanh suits as it is between -1 and 1
        # so scales well to the specific proiblem
        input_actor = Input(self.state_size)
        layer_1_actor = Dense(act_struct[0], activation="relu")(input_actor)
        layer_2_actor = Dense(act_struct[1], activation="relu")(layer_1_actor)
        tanh_layer = Dense(self.num_actions, activation='tanh')(layer_2_actor)

        self.main_actor = Model(inputs=input_actor, outputs=tanh_layer)

        # Critic Network
        input_critic = Input((self.num_actions + self.state_size))
        layer_1_critic = Dense(crit_struct[0], activation="relu")(input_critic)
        layer_2_critic = Dense(crit_struct[1], activation="relu")(layer_1_critic)
        output_critic = Dense(1, activation=None)(layer_2_critic)

        self.main_critic = Model(inputs=input_critic, outputs=output_critic)

        # Copy these networks to initialize the target networks
        self.target_actor = tf.keras.models.clone_model(self.main_actor)
        self.target_critic = tf.keras.models.clone_model(self.main_critic)
        # When cloning it seems weights a re initialized randomly so we need to set the weights of the target networks
        # those of the main networks
        self.target_actor.set_weights(self.main_actor.get_weights())
        self.target_critic.set_weights(self.main_critic.get_weights())

    """
    Define Utility functions for obtaining the Q value target, loss, the objective func for the actor, the soft 
    updates and storing and sampling from the buffer.
    """
    def get_critic_target(self, next_states, rewards, dones, gamma=0.99):
        # at this stage we are using a batch of states so states should have shape (batch_size, state_shape[0])
        # Compute the target networks prediction for next states i.e a' = mu(s')
        actions_by_target = 2 * self.target_actor(next_states)
        # Calculate target critics estimation of the value of s', a'
        # First we concatenate the next_states and actions
        inputs_to_val_net = tf.concat([next_states, actions_by_target], axis=1)
        targ_Q_vals = self.target_critic(inputs_to_val_net)
        # Use target critic estimation of Q value for a' and s' to build bellman target
        y = rewards + (1-dones) * gamma * targ_Q_vals
        return y

    def get_critic_loss(self, targets, states):
        # To gt the bellman error we now use main actor to find a=mu(s)
        actions_by_main = 2 * self.main_actor(states)
         # Input a=mu(s) and s into the critic network
        inputs_to_val_net = tf.concat([states, actions_by_main], axis=1)
        # Get the main critic estimation of the current state and action
        predicted_Q_vals = self.main_critic(inputs_to_val_net)
        # calculate the MSE between the predicted q value and the bellman target
        mean_squared_errors = tf.keras.losses.MSE(tf.squeeze(targets), tf.squeeze(predicted_Q_vals))
        return mean_squared_errors

    def actor_objective(self, states):
        # The actor objective here should calculate the Q value the whole way from actor through the critic so we can
        # get Q(s, mu(s)), which we the find gradients wrt to actor parameters and perform gradient ascent step 
        actions = 2 * self.main_actor(states)
        q_val_net_inputs = tf.concat([states, actions], axis=1)
        Q_vals = self.main_critic(q_val_net_inputs)
        # Want to carry out a gradient ascent step here so include -1, since optimizer.apply exclusively implements 
        # gradient descent
        loss = -1 * tf.reduce_mean(Q_vals)
        return loss

    def train_critic(self, states, rewards, next_states, dones, optimizer=Adam(lr=0.01)):
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        done = tf.convert_to_tensor(dones, dtype=tf.bool)

        with tf.GradientTape() as tape:
            Q_targets = self.get_critic_target(next_states,rewards,dones)
            loss = self.get_critic_loss(Q_targets, states)
        grads = tape.gradient(loss, self.main_critic.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.main_critic.trainable_variables))

    def train_actor(self, states, optimizer=Adam(lr=0.001)):
        states = tf.convert_to_tensor(states, tf.float32)
        with tf.GradientTape() as tape:
            Q_vals = self.actor_objective(states)
        grads = tape.gradient(Q_vals, self.main_actor.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.main_actor.trainable_variables))


    def soft_update(self, tau=0.001):
        # implement the soft target update by list wise computing the update
        actor_target_weights = self.target_actor.get_weights()
        actor_main_weights = self.main_actor.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        critic_main_weights = self.main_critic.get_weights()

        new_actor_target_weights = []
        for t in range(len(actor_target_weights)):
            new_actor_target_weights.append(tau * actor_main_weights[t] + (1 - tau) * actor_target_weights[t])
        self.target_actor.set_weights(new_actor_target_weights)
        
        new_critic_target_weights = []
        for t in range(len(critic_target_weights)):
            new_critic_target_weights.append(tau * critic_main_weights[t] + (1 - tau) * critic_target_weights[t])
        self.target_critic.set_weights(new_critic_target_weights)


    def store_sample(self, state, action, next_state, reward, done):
        # Store experience instances in the replay buffer
        if len(self.buffer)< self.buffer_size:
            self.buffer.append((state, action, next_state, reward, done))
        else:
            self.buffer.popleft()
            self.buffer.append((state, action, next_state, reward, done))

    def sample_experiences(self):
        """
        Returns 5 np.array() objects containing batch_size elements corresponding to the minibatch for 
        each element recorded in the replay buffer
        """
        states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        actions = np.zeros((self.batch_size, self.num_actions), dtype=np.float32)
        rewards = np.zeros((self.batch_size,1), dtype=np.float32)
        next_states = np.zeros((self.batch_size, self.state_size), dtype=np.float32)
        dones = np.zeros((self.batch_size,1), dtype=bool)
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        for i, elem in enumerate(indices):
            states[i,:] = self.buffer[elem][0][:]
            actions[i,:] = self.buffer[elem][1][:]
            next_states[i,:] = self.buffer[elem][2][:]
            rewards[i] = self.buffer[elem][3]
            dones[i] = self.buffer[elem][4]
        return states, actions, rewards, next_states, dones

    """
    define a scheduler for the noise sigma
    """

    def sigma_scheduler(self, episode, num_episodes, init_sigma=0.1, final_sigma=0.05):
        half_life = num_episodes/2
        sigma = max(init_sigma*(0.5)**(episode/half_life), final_sigma)
        return sigma

    """
    define an random process for exploration
    """

    def get_noisy_output(self, net_outputs, sig):
        action = net_outputs[0]
        return tf.clip_by_value(action+np.random.normal(loc=0.0, scale=sig, size = action.shape[0]), clip_value_min=-1, 
                                    clip_value_max=1)

