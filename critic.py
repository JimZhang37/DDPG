import tensorflow as tf
import numpy as np
import gym
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """
    #def __init__(self, state_size, action_size): #old
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        #net = tflearn.fully_connected(net, 128)
        #net = tflearn.layers.normalization.batch_normalization(net)
        #net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        #w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        #w_init = tflearn.initializations.xavier(uniform=False, seed=None, dtype=tf.float32)
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

'''
from keras import layers, models, optimizers
from keras import backend as K
from keras import regularizers

class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        #Initialize other variables
        
        self.build_model()
        
    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        net_states = layers.Dense(units=32, 
                                  activation='relu', 
                                   use_bias = True,
                                   kernel_initializer='glorot_uniform', 
                                   bias_initializer='zeros', 
                                   kernel_regularizer=None, 
                                   bias_regularizer=None, 
                                   activity_regularizer=None, 
                                   kernel_constraint=None, 
                                   bias_constraint=None)(states)
        net_states = layers.Dense(units=64, 
                                  activation='relu', 
                                  use_bias = True,
                                  kernel_initializer='glorot_uniform', 
                                  bias_initializer='zeros', 
                                  kernel_regularizer=None, 
                                  bias_regularizer=None, 
                                  activity_regularizer=None, 
                                  kernel_constraint=None, 
                                  bias_constraint=None)(net_states)
       
        
        net_actions = layers.Dense(units = 32, 
                                   activation='relu', 
                                   use_bias = True,
                                   kernel_initializer='glorot_uniform', 
                                   bias_initializer='zeros', 
                                   kernel_regularizer=None, 
                                   bias_regularizer=None, 
                                   activity_regularizer=None, 
                                   kernel_constraint=None, 
                                   bias_constraint=None)(actions)
        net_actions = layers.Dense(units = 64, 
                                   activation='relu', 
                                   use_bias = True,
                                   kernel_initializer='glorot_uniform', 
                                   bias_initializer='zeros', 
                                   kernel_regularizer=None, 
                                   bias_regularizer=None, 
                                   activity_regularizer=None, 
                                   kernel_constraint=None, 
                                   bias_constraint=None)(net_actions)
        
        
        net = layers.Add()([net_states, net_actions])
        
        net = layers.Activation('relu')(net)
        
        Q_values = layers.Dense(units = 1, 
                                name='q_values', 
                                activation= None, 
                                use_bias = True,
                                kernel_initializer='glorot_uniform', 
                                bias_initializer='zeros', 
                                kernel_regularizer=None, 
                                bias_regularizer=None, 
                                activity_regularizer=None, 
                                kernel_constraint=None, 
                                bias_constraint=None)(net)
        
        self.model = models.Model(inputs = [states, actions], outputs = Q_values)
        
        optimizer = optimizers.Adam()
        self.model.compile(optimizer = optimizer, loss ='mse')
        
        action_gradients = K.gradients(Q_values, actions)
        
        self.get_action_gradients = K.function(inputs = [*self.model.input, K.learning_phase()], 
                                               outputs = action_gradients)
'''            