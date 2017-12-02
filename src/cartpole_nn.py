import gym
from time import sleep
import numpy as np
import random
import tensorflow as tf

env = gym.make('CartPole-v0')
env.reset()
#env.render()

alpha = 0.1
gamma = 0.9

done = False
total = 0.0

print( env.action_space, env.action_space.n )

inputs = tf.placeholder(shape=[1,4],dtype=tf.float32)
W1 = tf.Variable(tf.random_uniform([4,8],0,0.01))
W2 = tf.Variable(tf.random_uniform([8,2],0,0.01))
Z1 = tf.matmul( inputs, W1 )
A1 = tf.sigmoid( Z1 )
Z2 = tf.matmul( A1, W2 )
Qout = Z2
predict = tf.argmax(Qout,1)

nextQ = tf.placeholder(shape=[1,2],dtype=tf.float32)
loss = tf.losses.mean_squared_error(Qout, nextQ)
#loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.2)
updateModel = trainer.minimize(loss)

init = tf.global_variables_initializer()

a = 0
with tf.Session() as sess:
    sess.run( init )
    for i in xrange( 100000):
        done = False
        total = 0.0
        s = env.reset()
        while not done:
            #a = env.action_space[ env.action_space.sample() ]
            Q, a = sess.run( [Qout, predict], feed_dict={ inputs : s.reshape( 1, 4 ) })
            s1, reward, done, _ = env.step( a[0] )
            Q1 = sess.run( Qout, feed_dict = { inputs : s1.reshape( 1, 4 ) })
            Q[0,a[0]] = ( 1-alpha) * Q[0,a[0]] + alpha * ( reward * gamma * np.max( Q1
                                                                                    ))
            sess.run( updateModel, feed_dict={ inputs: s.reshape( 1,4 ), nextQ : Q})
            total += reward
            #env.render()
            #sleep(0.02)
        if i%1000 == 0:
            print(i)
            print( 'Total reward: ', total)

