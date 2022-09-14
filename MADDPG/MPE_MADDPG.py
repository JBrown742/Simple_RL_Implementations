#!/usr/bin/env python
# coding: utf-8

# In[49]:


from MADDPG import MADDPG, MultiAgentReplayBuffer
import pettingzoo
import numpy as np
from pettingzoo.mpe import simple_tag_v2, simple_v2, simple_adversary_v2, simple_speaker_listener_v3, simple_push_v2
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

#env = simple_tag_v2.env(num_good=1, num_adversaries=3,
#                     num_obstacles=2, max_cycles=25,
#                     continuous_actions=True)
env = simple_speaker_listener_v3.env(max_cycles=25, continuous_actions=True)
#env = simple_push_v2.env(max_cycles=25, continuous_actions=True)
#env = simple_v2.env(max_cycles=25, continuous_actions=True)
#env = simple_adversary_v2.env(N=2, max_cycles=25, continuous_actions=True)
# Get all the relevant params for the MADDPG lass


def single_run(env, seed, batch_size, actor_arc, critic_arc, actor_lr, critic_lr, tau):
    np.random.seed(seed)
    names = env.possible_agents
    n_actions = []
    obs_sizes = []
    for name in names:
        act_size = env.action_spaces[name].shape[0]
        n_actions.append(act_size)
        obs_size = env.observation_spaces[name].shape[0]
        obs_sizes.append(obs_size)

    maddpg = MADDPG(observation_sizes=obs_sizes, gamma=0.98, n_actions=n_actions, agent_names=names,
                    actor_structs=actor_arc, critic_structs=critic_arc)
    memory = MultiAgentReplayBuffer(max_size=5000, observation_sizes=obs_sizes,
                                n_actions=n_actions, batch_size=batch_size, agent_names=names)
    maddpg.init_agent_networks()

    MAX_EPISODES = 100000
    episode_rewards = np.zeros((MAX_EPISODES, env.max_num_agents))
    learn_every=25
    sum_rewards=[]

    for episode in range(MAX_EPISODES):
        env.reset()
        ep_rewards = 0
        sigma=maddpg.sigma_scheduler(episode=episode, num_episodes=MAX_EPISODES)

        for agent in env.agent_iter():
            env.render()
            obs, reward, done, info = env.last(observe=True)

            if done:
                action=None
            else:
                action = maddpg.get_action_async(obs, agent, noise=True, scale=sigma)
            memory.store_transition_async(obs, action, reward, done, agent=agent)
            state = env.state()
            env.step(action)
            state_ = env.state()
            state_update = np.sum(state-state_)
            if bool(env.agents)==False:
                break
            if state_update!=0.:
                memory.mem_cntr+=1
                reward = sum(list(env.rewards.values()))
                ep_rewards+=reward
                if memory.mem_cntr%learn_every==0 and memory.ready():
                    batch_obs, batch_next_obs, batch_rewards, batch_actions, batch_terminals \
                    = memory.sample_buffer()
                    print("training.....")
                    for learning_agent in env.agents:
                        maddpg.train_critic(name=learning_agent, obs=batch_obs, actions=batch_actions,
                                next_obs=batch_next_obs, rewards=batch_rewards,
                                dones=batch_terminals, optimizer=Adam(learning_rate=critic_lr, epsilon=1e-3))
                        maddpg.train_actor(name=learning_agent, obs=batch_obs, rewards=batch_rewards,
                                dones=batch_terminals, optimizer=Adam(learning_rate=actor_lr, epsilon=1e-3))
                # print("learning done for agent: {}".format(learning_agent))
                maddpg.soft_update(tau=tau)
        sum_rewards.append(ep_rewards)
        # if episode%10==0:
#             print("EPISODE: {}, REWARD: {}".format(episode, this_ep_reward))
    env.close()


  #   for name in names:
#         maddpg.actors[name].save(str(env)+name+"_finalmodel_"+run)
    this_run_rewards = np.array(sum_rewards)
    return this_run_rewards

n = len(env.possible_agents)
batch = 32
critic_lr = 1e-3
actor_lr = 1e-4
actor_arc = [64]
critic_arc = [64]
tau =  0.005
seed = 12345
learning_run_reward =  single_run(env=env, seed=seed, batch_size=batch,
                                actor_arc=actor_arc, critic_arc=critic_arc,
                                actor_lr=actor_lr,
                                critic_lr=critic_lr,
                                tau=tau)
plt.figure()
plt.plot(learning_run_reward)
plt.savefig(fname="simple_speaker_listener_v3_{}_{}_{}_{}_{}_{}_return.png".format(batch, actor_lr, critic_lr, actor_arc, critic_arc, tau), dpi=300)
