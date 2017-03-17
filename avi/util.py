import numpy as np

def uniform_sampler(env,n_samples=1):
    '''uniform state space sampler for gym environments'''
    samples=[]
    for i in range(n_samples):
        samples.append(env.observation_space.sample())
    return np.array(samples)

def trajectory_sampler(env,n_samples=1):
    '''trajector based state sampler for gym environments'''
    samples=[]
    obs = env.reset()
    for i in range(n_samples):
        samples.append(obs)
        obs,_,done_ = env.step(env.action_space.sample())
        if done:
            obs = env.reset()
    return np.array(samples)
