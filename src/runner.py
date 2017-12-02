import gym
from agent_base import RandomAgent
from agent_nn import AgentNN
from time import sleep
import numpy as np

def run( environmentName, agent, converter, episodes, resultFrequency, render = False, delay=0.0 ):

    converter = converter or ( lambda x : x )
    env = gym.make( environmentName)
    total = 0.0


    for i in xrange( episodes ):
        s = env.reset()
        done = False

        if render:
            env.render()

        while not done:
            a = agent.getAction( converter( s ), env.action_space )
            s, r, done, _ = env.step( a )
            agent.feedback( converter( s ), 0.0, done )
            total += r
            if render:
                env.render()
                sleep( delay )

        #print 'DONE'
        if render:
            sleep( delay*10)

        if i%resultFrequency==0:
            print '%d - Reward Total %f, Average %f'%( i, total, total/resultFrequency)
            total = 0.0

def one_hot( n ):
    def f( s ):
        retval = np.zeros( ( n ) )
        retval[ s ] = 1.0
        return retval

    return f

#run( 'CartPole-v0', AgentNN( 4, 2, 4), None, 1000, 10, False, 0.5 )
run( 'FrozenLake-v0', AgentNN( 16, 4, 16), one_hot( 16 ), 1000, 100, False, 0.5 )