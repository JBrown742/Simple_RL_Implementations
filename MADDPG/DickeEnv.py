#!/usr/bin/env python
# coding: utf-8

# In[291]:


import numpy as np
import tensorflow as tf
import qutip as qt
from qutip import identity, basis, projection, tensor, create, destroy
import itertools
import matplotlib.pyplot as plt


class DickeEnv:
	
	def __init__(self, N, dicke_k=1, max_steps=11, max_coupling=1, max_qubit_driving=10):
		"""
		attempt at coding n qubits in full generality
		"""
		self.max_steps = max_steps
		self.N=N
		self.cav_dim=dicke_k + 3
		self.qubit_names = ["qubit_{}".format(num) for num in range(self.N)]
		self.anharm = 0.200
		self.cav_freq = (2 * np.pi) * 6
		self.g = (2 * np.pi) * 0.100
		self.max_coupling = max_coupling
		self.max_qubit_driving = max_qubit_driving
		"""
		Initialize qubit operators accesible via dictionaries
		"""

		self.op01 = {}
		self.op11 = {}
		for qubit_idx, qubit_name in enumerate(self.qubit_names):
			op01_ls = []
			op11_ls = []
			for i in range(len(self.qubit_names)):
				if i==qubit_idx:
					op01_ls.append(projection(2,0,1))
					op11_ls.append(projection(2,1,1))
				else:
					op01_ls.append(identity(2))
					op11_ls.append(identity(2))
			op01_ls.append(identity(self.cav_dim))
			op11_ls.append(identity(self.cav_dim))
			self.op01[qubit_name]=tensor(op01_ls)
			self.op11[qubit_name]=tensor(op11_ls)

		"""
		initialize the creation op for the resonator
		"""

		a_ls = [identity(2) for _ in range(self.N)]
		a_ls.append(destroy(self.cav_dim))
		self.a = tensor(a_ls)
		

		
		"""
		get the list of strings corresponding to all strings of n bits with k ones
		"""
		result = []
		for bits in itertools.combinations(range(self.N), dicke_k):
			s = ['0'] * self.N
			for bit in bits:
				s[bit] = '1'
			result.append(''.join(s))  
		targ_pure_state = qt.tensor([qt.zero_ket(2) for i in range(self.N)])
		for string in result:
			targ_pure_state+=qt.ket(string)
		targ_state = (1/np.sqrt(len(result))) * targ_pure_state
		self.target_state_dm = targ_state * targ_state.dag()

		
	def decode_actions(self, actions_ls):
		"""
		The actions list will be of the form [[delta_q_0, gamma_q_0], [delta_q_1, gamma_q_1],.... etc]
		"""
		deltas = []
		gammas = []
		for idx, name in enumerate(self.qubit_names):
			coupling = actions_ls[idx][0,0]*self.max_coupling
			gamma = actions_ls[idx][0,1]*self.max_qubit_driving
			deltas.append(coupling)
			gammas.append(gamma)
		return deltas, gammas
		
	def get_obs_from_state(self, state):
		observations = []
		for idx in range(self.N):
			reduced = state.ptrace(idx)
			reduced_array = reduced.full().flatten()
			imaginary = reduced_array.imag
			real = reduced_array.real
			obs = np.concatenate((real,imaginary))
			observations.append(obs)
		return observations
	
	def exp_plus(self, t, args):
		return np.exp(1j * self.cav_freq * t)
	
	def exp_minus(self, t, args):
		return np.exp(-1j * self.cav_freq * t)
		
	def build_hamiltonian(self, couplings, gammas, rate=0.1):
		H_ls = [qt.zero_ket(2) * qt.zero_ket(2).dag() for _ in range(self.N)]
		H_ls.append(qt.zero_ket(self.cav_dim) * qt.zero_ket(self.cav_dim).dag())
		H_const = tensor(H_ls)
		rate=0.1 * self.g
		H_full = []
		for idx, name in enumerate(self.qubit_names):
			H_const+= (0 * self.op11[name] + 
				 self.g * couplings[idx] * (self.op01[name] * self.a.dag() + self.op01[name].dag() * self.a))
					   
		H_full.append(H_const)
		collapse_ops = []
		for idx, name in enumerate(self.qubit_names):
			H_full.append([self.g * gammas[idx] * self.op01[name], self.exp_minus])
			H_full.append([self.g * gammas[idx] * self.op01[name].dag(), self.exp_plus])
			collapse_ops.append(np.sqrt(rate) * self.op01[name])
		return H_full, collapse_ops
	
	def reset(self):
		self.dones = [False] * self.N
		state_ls = []
		for i in range(self.N):
			state_ls.append(basis(2,0))
		state_ls.append(basis(self.cav_dim,0))
		self.state = tensor(state_ls)
		self.step_cnt=0
		observations = self.get_obs_from_state(self.state)
		return observations
	
	def step(self, actions, init_time, duration, rate=0.1):
		couplings, gammas = self.decode_actions(actions)
		inner_tlist = np.linspace(init_time, init_time+duration, 11)
		H, collapse = self.build_hamiltonian(couplings, gammas, rate=rate)
		results = qt.mesolve(rho0=self.state, H = H, tlist=inner_tlist, c_ops=collapse)
		self.state = results.states[-1]
		state_histories = results.states

		self.step_cnt+=1
		state_of_qubit_net = self.state.ptrace([i for i in range(self.N)])
		fidelity = qt.fidelity(state_of_qubit_net, self.target_state_dm)
		info = {"state his": state_histories, "fidelity":fidelity}
		observations = self.get_obs_from_state(self.state)
		reward=fidelity
		self.dones = [False] * self.N
		rewards = [reward] * self.N
		return observations, rewards, self.dones, info
	
	def close(self):
		return
	
	def render(self):
		return

