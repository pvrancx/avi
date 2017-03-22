import numpy as np
from sklearn.base import clone
from timeit import default_timer as timer
import cPickle as pickle
import matplotlib.pyplot as plt

class ApproximateValueIteration(object):
    '''Approximate Value Iteration with sklearn estimators'''

    def __init__(self,sim, gamma, sampler, regressor, v0=0.):
        ''' Approximate Value Iteration
            Inputs:
            sim: environment simulator should support set_state and step methods
            gamma: discount factor
            sampler: callable to sample states
            regressor: estimator implementing sklearn interface
            v0: initial value (default=0)

        '''
        self.env = sim
        self.n_actions = sim.action_space.n
        self.sampler = sampler
        self.gamma = gamma
        self.prototype = clone(regressor)
        self.regressors = [clone(regressor) for _ in range(self.n_actions)]
        self.trained = False
        self.v0 = v0

    def save_value_function(self,filename):
        '''save action values to file'''
        with open(filename,'wb') as f:
            pickle.dump(self.regressors,f,-1)

    def load_value_function(self,filename):
        '''load action values from file'''
        with open(filename,'rb') as f:
            self.regressors=pickle.load(f)

    def generate_samples(self,n_samples, n_transitions):
        states = self.sampler(self.env,n_samples)
        nstates = np.zeros((n_samples,self.n_actions, n_transitions,states.shape[1]))
        rewards = np.zeros((n_samples,self.n_actions,n_transitions))
        dones = np.zeros((n_samples,self.n_actions,n_transitions), dtype=bool)
        for s in range(n_samples):
            state = states[s,:]
            for a in range(self.n_actions):
                for t in range(n_transitions):
                    self.env.set_state(state)
                    obs,rew,done,_ = self.env.step(a)
                    nstates[s,a,t,:] = obs
                    rewards[s,a,t]= rew
                    dones[s,a,t]= done
        return states, rewards, nstates, dones

    def train(self,n_samples,n_transitions, n_iter, resample=-1):
        ''' Run value iterations for given number of iterations'''
        start_time = timer()
        avg_time = 0.
        for it in range(n_iter):
            regressors = [clone(self.prototype) for _ in range(self.n_actions)]
            if (it==0) or ((resample > 0 ) and (it % resample == 0)):
                states, rewards, nstates, dones = self.generate_samples(n_samples,n_transitions)
            for a in range(self.n_actions):
                nvals = self.get_multiple_state_values(nstates[:,a,:,:].reshape((-1,states.shape[1]))).reshape((-1,n_transitions))
                vals = rewards[:,a,:] + self.gamma * nvals * (np.logical_not(dones[:,a,:]))
                regressors[a].fit(states,np.mean(vals,-1))
            end_time = timer()
            dt = end_time - start_time
            avg_time += (dt-avg_time)/float(it+1)
            print('ended iteration %d -- time %f --avg time %f -- remaining %f'%(it,dt,avg_time,avg_time*(n_iter-it-1)))
            start_time = end_time

            self.regressors = regressors
            self.trained = True

    def get_multiple_state_values(self,states):
        '''get values of action in set of states'''
        res = self.get_multiple_state_action_values(states,0)
        for a in range(1,self.n_actions):
            res = np.maximum(self.get_multiple_state_action_values(states,a),res)
        return res


    def get_multiple_state_action_values(self,states,action):
        '''get values of action in set of states'''
        if self.trained:
            return self.regressors[action].predict(states).flatten()
        else:
            return np.ones(states.shape[0])*self.v0

    def get_action_value(self,state,action):
        '''get single state action value '''
        if self.trained:
            return self.regressors[action].predict(state.reshape((1,-1)))[0]
        else:
            return self.v0

    def get_all_values(self,state):
        '''get values for all actions in given state'''
        vals = [self.get_action_value(state,a) for a in range(self.n_actions)]
        return np.array(vals)

    def get_value(self,state):
        '''get (greedy) state value'''
        return np.max(self.get_all_values(state))
