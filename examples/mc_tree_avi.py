import gym
import gym.envs.classic_control

from sklearn.tree import DecisionTreeRegressor

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys

sys.path.append('../avi')
from avi import ApproximateValueIteration
from util import uniform_sampler

'''Aprroximate Value Iteration on Mountaincar with regression trees'''

parser = argparse.ArgumentParser(description='Approximate Value Iteration')
parser.add_argument('--niter', type=int, default=1000,
                    help='number of iterations')
parser.add_argument('--nsamples', type=int, default=1000,
                    help='sample to generate per iteration ')

args = parser.parse_args()



# mountaincar simulator (add option to set to random state)
class MountainCarSim(gym.envs.classic_control.MountainCarEnv):
    def set_state(self,state):
        self.state = state

# prototype for regressor to use: regression tree
regressor = DecisionTreeRegressor(max_depth=15,min_samples_leaf=3)


# run AVI
sim = MountainCarSim()
mc_avi = ApproximateValueIteration(sim,1.,uniform_sampler,regressor)

mc_avi.train(n_samples=args.nsamples,n_transitions=1, n_iter=args.niter)
mc_avi.save_value_function('temp.pkl')

# apply learnt policy to real env
env = gym.make('MountainCar-v0')
obs = env.reset()
total_rew = 0.

traj = []
for i in range(500):
    traj.append(obs)
    # q-values for current state
    vals = mc_avi.get_all_values(obs)
    #e-greedy
    if np.random.rand() < 0.05:
        act = np.random.choice(mc_avi.n_actions)
    else:
        act = np.random.choice(np.arange(mc_avi.n_actions)[vals==np.max(vals)])

    #step environment
    obs, rew, done, _ = env.step(act)
    total_rew += rew
    #uncomment to observe trajectory
    #env.render()
    if done:
        print('Goal reached -- steps %d -- total reward %f'%(i,total_rew))
        break

#plot result
n_actions = env.action_space.n
low, high = env.observation_space.low, env.observation_space.high
Xs, Ys = np.meshgrid(np.linspace(low[0],high[0],100),
                     np.linspace(low[1],high[1],100))
Qs = np.zeros( Xs.shape+(n_actions,))

for a in range(env.action_space.n):
    Qs[:,:,a].flat = mc_avi.get_multiple_state_action_values(np.c_[Xs.flatten(),Ys.flatten()],a)


fig = plt.figure()
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.plot_surface(Xs, Ys, -np.max(Qs,axis=2))
ax.set_title('value function')

ax = fig.add_subplot(1, 3, 2)
ax.matshow(np.argmax(Qs,axis=-1))
ax.set_title('policy')

traj =np.array(traj)
ax = fig.add_subplot(1, 3, 3)
ax.plot(traj[:,0],traj[:,1])
ax.set_title('trajectory')

plt.show()
