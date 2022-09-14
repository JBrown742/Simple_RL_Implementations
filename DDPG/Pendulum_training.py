import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import gym 
from DDPG import DDPG

env = gym.make("LunarLander-v2")
acts = env.action_space.shape[0]
state_size = env.observation_space.shape[0]



ddpg = DDPG(n_actions=1, state_size=state_size, learning_batch_size=64, act_struct=[64,64], crit_struct=[32,32])

num_episodes =500
returns_array = np.zeros(num_episodes)
counter = 0
episode_length=100
training_start=1000#batch_size*5
returns_benchmark = -3000
for episode in range(num_episodes):
    state = env.reset()
    done=False
    returns=0.
    sigma = ddpg.sigma_scheduler(episode=episode, num_episodes=num_episodes)
    t=0
    while not done:
        env.render()
        counter+=1
        main_actor_prediction = ddpg.main_actor(state.reshape((1,state_size)))
        noisy_predict = ddpg.get_noisy_output(main_actor_prediction, sig=sigma)
        action = 2 * noisy_predict
        next_state, reward, done, _ = env.step(action.numpy())
        t+=1
        if t==episode_length:
            done=True
        returns+=reward
        ddpg.store_sample(state, action, next_state, reward, done)
        state=next_state

        if counter > training_start:
            states, actions, rewards, next_states, dones = ddpg.sample_experiences()
            ddpg.train_critic(states=states, rewards=rewards, 
                                next_states=next_states, dones=dones)
            ddpg.train_actor(states=states)
            ddpg.soft_update()
    returns_array[episode] = returns
    print(returns)
env.close()