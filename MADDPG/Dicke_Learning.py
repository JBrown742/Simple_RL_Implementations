import numpy as np
import matplotlib.pyplot as plt
from MADDPG import MADDPG, MultiAgentReplayBuffer
from DickeEnv import DickeEnv
from UltraStrongNetwork import UltraStrongNetwork


Num_qubits = 2
dicke_k = 1


env = DickeEnv(N=Num_qubits, dicke_k=dicke_k, max_steps=11)
# env = UltraStrongNetwork(Num_qubits, dicke_k,max_steps=101)
n_actions = [2] * Num_qubits
observation_sizes = [8] * Num_qubits

names = ['qubit_{}'.format(i) for i in range(Num_qubits)]

maddpg = MADDPG(observation_sizes=observation_sizes, n_actions=n_actions, agent_names=names)
memory = MultiAgentReplayBuffer(max_size=10000, 
								observation_sizes=observation_sizes, 
								n_actions=n_actions, 
								batch_size=32,
								agent_names=names)
								
maddpg.init_agent_networks()
MAX_EPISODES = 20000
max_ep_time = 50
ep_steps = env.max_steps
episode_rewards = np.zeros((MAX_EPISODES,ep_steps))
learn_every=25
max_rewards=np.zeros(MAX_EPISODES)
times = np.linspace(0,max_ep_time,ep_steps)
duration = (max_ep_time + 1)/ep_steps
best_model_count = 0
for episode in range(MAX_EPISODES):
	obs = env.reset()
	ep_rewards = 0
	sigma=maddpg.sigma_scheduler(episode=episode, num_episodes=MAX_EPISODES, init_sigma=0.2)
	for i, t in enumerate(times):
		actions = maddpg.get_action_sync(obs, noise=True, scale=sigma)
		next_obs, rewards, dones, info = env.step(actions=actions, duration=duration, init_time=t, rate=0)
		if t==times[-1]:
		    dones=[True] * env.N
		memory.store_transition(raw_obs=obs, actions=actions, rewards=rewards, 
										next_raw_obs=next_obs, dones=dones)
		episode_rewards[episode,i]=rewards[0]
		
		
		if memory.mem_cntr%learn_every==0 and memory.ready():
			batch_obs, batch_next_obs, batch_rewards, batch_actions, batch_terminals = memory.sample_buffer()
			for learning_agent in maddpg.agent_names:
				maddpg.train_critic(name=learning_agent, obs=batch_obs, actions=batch_actions, 
										next_obs=batch_next_obs, rewards=batch_rewards, 
										dones=batch_terminals)
				maddpg.train_actor(name=learning_agent, obs=batch_obs, rewards=batch_rewards, 
										dones=batch_terminals)
			maddpg.soft_update(tau=0.01)
	
	max_rewards[episode] = np.max(episode_rewards[episode,:])
	if episode%20==0:
		print("episode: {}, AVG reward: {}, MAX Reward: {}, Final reward: {}".format(episode, np.average(episode_rewards[episode,:]),
		                                                                             max_rewards[episode], rewards[0]))
	if rewards[0]>0.95:
	    best_model_count+=1
	    for learning_agent in maddpg.agent_names:
		    maddpg.actors[learning_agent].save("N"+str(Num_qubits)+"K"+str(dicke_k)+"/"+learning_agent+'_'+str(episode)+"best_"+str(best_model_count))
	if best_model_count==5:
	    break
			
plt.figure()
plt.plot(max_rewards)
plt.title("maximum fidelity acheived in each episode \n during learning (n={},k={})".format(env.N, env.cav_dim-1))
plt.xlabel("Episodes")
plt.ylabel("fidelity")
plt.savefig("Learning Curve; (n={},k={})".format(env.N, env.cav_dim-1))
plt.show()