import gym
import numpy as np
import random
import tensorflow as tf

env = gym.make('FrozenLake-v0')
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 5000
e = [0.5,0.1,0.05,0.01,0.0]
#create lists to contain total rewards and steps per episode
for iter in range( len(e)):
    JList = []
    rList = []

    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while not d:
            j+=1
            #Choose an action by greedily (with noise) picking from Q table
            #a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
            if random.random() < e[iter]:
                a = env.action_space.sample()
            else:
                a = np.argmax( Q[s,:])
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a)
            #Update Q-Table with new knowledge
            Q[s,a] = (1-lr) * Q[s,a] + lr*(r + y*np.max(Q[s1,:]))
            rAll += r
            s = s1

        JList.append(j)
        rList.append(rAll)

    print 'Iteration ', iter
    print "Score over time: " +  str(sum(rList)/num_episodes)
    print "Length over time: " + str(sum(JList) / num_episodes)
