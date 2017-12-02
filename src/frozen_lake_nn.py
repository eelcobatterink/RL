import gym
import numpy as np
import random
import tensorflow as tf

import matplotlib.pyplot as plt

def createOneHot( s ):
    result = np.zeros( ( 1, 16 ) )
    result[0][s] = 1.0
    return result

def createCartesian( states ):
    result = np.array( [ [ float( s%4 ), float( s/4 ) ] for s in states ])
    return result


env = gym.make('FrozenLake-v0')

tf.reset_default_graph()


#These lines establish the feed-forward part of the network used to choose actions
inputs = tf.placeholder(shape=[None,2],dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([2,16],0,0.01))
W2 = tf.Variable(tf.random_uniform([16,4],0,0.01))
Z1 = tf.matmul( inputs, W1 )
A1 = tf.sigmoid( Z1 )
Z2 = tf.matmul( A1, W2 )
Qout = Z2
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
loss = tf.losses.mean_squared_error(Qout, nextQ)
#loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()
batch_size=8

# Set learning parameters
y = .99
e = 0.1
num_episodes = 200
#create lists to contain total rewards and steps per episode
jList = []
rList = []

memory = []
with tf.Session() as sess:
    sess.run(init)
    for iter in range( 10 ):
        for i in range(num_episodes):
            #Reset environment and get first new observation
            s = env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Network
            while not d:
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                a,allQ = sess.run([predict,Qout],feed_dict={inputs:createCartesian([s])})
                if np.random.rand(1) < e:
                    a[0] = env.action_space.sample()
                #Get new state and reward from environment
                s1,r,d,_ = env.step(a[0])
                memory.append(( s, a[0], s1, r) )
                memSlice = memory[-batch_size:]

                priorStates = [ x[0] for x in memSlice ]
                nextStates = [ x[2] for x in memSlice ]
                actions = [ x[1] for x in memSlice ]
                rewards = [ x[3] for x in memSlice ]
                #Obtain the Q' values by feeding the new state through our network
                Q0 = sess.run(Qout, feed_dict={inputs: createCartesian(priorStates)})
                Q1 = sess.run(Qout,feed_dict={inputs:createCartesian(nextStates)})

                for i in xrange( len( Q0 )):
                    Q0[priorStates[i]] = rewards[i] + y * max( Q1[nextStates[i]])
                #Obtain maxQ' and set our target value for chosen action.
                #Train our network using target and predicted Q values
                _ = sess.run([updateModel],feed_dict={inputs:createCartesian(priorStates),nextQ:Q0})
                rAll += r
                s = s1
                if d == True:
                    #Reduce chance of random action as we train the model.
                    e = 1./((i/50) + 10)
                    break
            jList.append(j)
            rList.append(rAll)
        print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"
        print 'Length', np.mean(jList)