#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import tensorflow as tf
import qutip as qt
from qutip import sigmax, sigmaz, destroy, basis, tensor, identity, projection
import itertools

class UltraStrongNetwork:
    def __init__(self, n_qubits, dicke_k, max_steps=11):
        self.N = n_qubits
        self.cav_dim = dicke_k + 1
        self.max_steps = max_steps
        self.qubit_names = ["qubit_{}".format(num) for num in range(self.N)]
        self.anharm = 0.200
        self.cav_freq = (2 * np.pi) * 10
        self.g =  self.cav_freq
        """
        Initialize qubit operators accesible via dictionaries
        """

        self.sigma_z = {}
        self.sigma_x = {}
        for qubit_idx, qubit_name in enumerate(self.qubit_names):
            sigma_z_ls = []
            sigma_x_ls = []
            for i in range(len(self.qubit_names)):
                if i==qubit_idx:
                    sigma_x_ls.append(sigmax())
                    sigma_z_ls.append(sigmaz())
                else:
                    sigma_x_ls.append(identity(2))
                    sigma_z_ls.append(identity(2))
            sigma_x_ls.append(identity(self.cav_dim))
            sigma_z_ls.append(identity(self.cav_dim))
            self.sigma_x[qubit_name]=tensor(sigma_x_ls)
            self.sigma_z[qubit_name]=tensor(sigma_z_ls)

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
        The actions list will be of the form [[freq_0, tunnelling_0, coupling_0], 
        [freq_1, tunnelling_1, coupling_1],.... etc]
        """
        freqs = []
        tunnelings = []
        for idx, name in enumerate(self.qubit_names):
            freq = self.cav_freq + actions_ls[idx][0,0]* self.g
            tunnel = actions_ls[idx][0,1] * 0.1 * self.cav_freq
            freqs.append(freq)
            tunnelings.append(tunnel)
        return freqs, tunnelings
    
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
        
    def build_hamiltonian(self, trans_freqs, tunneling_strengths):
        """
        let all of these input params be lists or arrays of len = self.N
        """
        H = self.cav_freq * self.a * self.a.dag()
        
        for idx, name in enumerate(self.qubit_names):
            H += ((trans_freqs[idx]/2) * self.sigma_z[name] + (tunneling_strengths[idx]/2) * self.sigma_x[name]
                  + self.g * (self.a.dag() + self.a) * self.sigma_z[name])
        
        return H
    
    def step(self, actions, duration=1, resolution=10):
        freqs, tunnelings = self.decode_actions(actions)
        
        H = self.build_hamiltonian(trans_freqs=freqs, tunneling_strengths=tunnelings)
        
        times = np.linspace(0,duration, resolution)
        
        result = qt.mesolve(H=H, rho0=self.state, tlist=times)
        
        final_state = result.states[-1]
        
        self.state = final_state
        self.step_cnt+=1 
        full_state_obs = final_state.full().flatten()
        full_state_proc = np.concatenate((full_state_obs.real, full_state_obs.imag))
        
        local_obs = self.get_obs_from_state(final_state)
        state_of_qubit_net = self.state.ptrace([i for i in range(self.N)])
        fidelity = qt.fidelity(state_of_qubit_net, self.target_state_dm)
        reward= fidelity * 10
        if fidelity > 0.99:
            self.dones = [True] * self.N
            reward+=1000
        else:
            self.dones = [False] * self.N
        rewards = [reward] * self.N	
        info = {"state_histories": result.states}
        return full_state_proc, local_obs, rewards, self.dones, info
    
    def reset(self):
        self.dones = [False] * self.N
        state_ls = []
        for i in range(self.N):
            state_ls.append(basis(2,0))
        state_ls.append(basis(self.cav_dim,self.cav_dim-1))
        self.state = tensor(state_ls)
        
        full_state_obs = self.state.full().flatten()
        full_state_proc = np.concatenate((full_state_obs.real, full_state_obs.imag))
        
        self.step_cnt=0
        local_observations = self.get_obs_from_state(self.state)
        return full_state_proc, local_observations
        
        

