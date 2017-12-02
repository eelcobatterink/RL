from agent_base import Agent
import tensorflow as tf
import numpy as np
from copy import copy
import random

class AgentNN( Agent):

    def __init__(self, input_count, output_count, hidden_count, gamma=0.9, alpha=0.5):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 0.5

        self.output_count = output_count
        self.input_count = input_count
        self.hidden_count = hidden_count

        self.memory = []

        self.inputs = tf.placeholder(shape=[None, input_count], dtype=tf.float32)
        self.W1 = tf.get_variable("W1", shape=[input_count,hidden_count],
                            initializer=tf.contrib.layers.xavier_initializer())
        self.W2 = tf.get_variable("W2", shape=[hidden_count, output_count],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.Z1 = tf.matmul(self.inputs, self.W1)
        self.A1 = tf.nn.relu(self.Z1)
        self.Z2 = tf.matmul(self.A1, self.W2)
        self.Qout = self.Z2

        self.nextQ = tf.placeholder(shape=[None, output_count], dtype=tf.float32)
        self.loss = tf.losses.mean_squared_error(self.Qout, self.nextQ)
        self.trainer = tf.train.AdamOptimizer( learning_rate = 0.01).minimize( self.loss )

        self.sess = tf.Session()
        self.sess.run( tf.global_variables_initializer() )
        self.state = None
        self.action = None
        self.step = 0

        Q = self.sess.run(self.Qout, feed_dict={self.inputs: np.identity(16)})
        print Q
        print np.abs(Q).mean().mean()


    def getAction(self, state, action_space):
        #self.epsilon = max( 0.01, self.epsilon * 1.0 )
        self.state = state

        if random.random < self.epsilon:
            return action_space.sample()
        else:
            Q = self.sess.run( self.Qout, feed_dict={self.inputs : state.reshape( 1, self.input_count)} )
            self.action = np.argmax(Q)
            #print( state, '->', Q, '->', self.action)

            return self.action

    def feedback(self, nextState, reward, done):
        self.memory.append( ( self.state, self.action, nextState, reward ) )
        self.train()
        if done:
            self.step += 1


    def train(self):
        samples = self.memory[ -128:]
        prevStates = np.array( [x[0] for x in samples ])
        nextStates = np.array( [x[2] for x in samples])
        actions = [ x[1] for x in samples ]
        rewards = [ x[3] for x in samples ]

        assert prevStates.shape == ( len( samples ), self.input_count ), prevStates.shape
        assert nextStates.shape == ( len( samples ), self.input_count ), nextStates.shape

        Qorig = self.sess.run( self.Qout, feed_dict={self.inputs : prevStates} )
        Qnext = self.sess.run( self.Qout, feed_dict={self.inputs : nextStates} )

        Qtarget = copy(Qorig)

        assert Qorig.shape == ( len( samples), self.output_count)
        assert Qnext.shape == (len(samples), self.output_count)

        #print( reward, Qorig, Qnext )

        for i, ( action , reward ) in enumerate( zip(actions, rewards) ):
            assert all( Qtarget[i] == Qorig[i]), "%s %s"%( Qtarget[i], Qorig[i] )
            assert self.gamma < 1.0
            assert self.alpha < 1.0
            Qtarget[i,action] = (1.0-self.alpha) * Qtarget[i,action] + self.alpha * ( reward + self.gamma * np.max(Qnext[i:]))
            assert reward == 0.0
            if reward == 0.0:
                assert abs( Qtarget[i,action] )  <= max( abs(Qorig[i,action]), abs(np.max( Qnext[i:]) ) ) , "%f  %f  %f"%(Qtarget[i,action], Qorig[i,action], np.max( Qnext[i:]) )
            if reward > 0.0:
                print action, reward, prevStates[i], nextStates[i]
                print Qorig[ i,action ], Qtarget[ i, action ], np.max( Qnext[i:])
                print

        Qtarget = Qtarget * 0.0
        #self.sess.run( self.trainer, feed_dict={ self.inputs: prevStates, self.nextQ: Qtarget})
        self.sess.run(self.trainer, feed_dict={self.inputs: np.identity(16), self.nextQ: np.zeros((16, 4))})

        if self.step % 100 == 0:
            #print Qtarget - Qorig
            Q = self.sess.run(self.Qout, feed_dict={self.inputs: np.identity(16)})
            #print Q
            print np.abs(Q).mean().mean()
            print 'cost', self.sess.run( self.loss, feed_dict={ self.inputs: np.identity(16), self.nextQ: np.zeros( ( 16, 4 ))} )
            #print self.step
            #print self.sess.run( self.Qout, feed_dict = { self.inputs : np.identity( 16 ) })

