import tensorflow as tf
import numpy as np
import tflearn
from replay_buffer import ReplayBuffer
#import argparse
#import pprint as pp



# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """
    #def __init__(self, state_size, action_size, action_low, action_high):#old interface
    #def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):#new
    def __init__(self, sess, state_dim, action_dim, action_low, action_high, learning_rate, tau, batch_size):#new
    
        self.sess = sess#new
        self.s_dim = state_dim
        self.a_dim = action_dim
        #self.action_bound = action_bound            #delete
        self.action_low = action_low                 #new
        self.action_high = action_high               #new
        self.action_range = action_high - action_low #new
        self.learning_rate = learning_rate           #new
        self.tau = tau                               #new
        self.batch_size = batch_size                 #new

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        #net = tflearn.fully_connected(net, 128)
        #net = tflearn.layers.normalization.batch_normalization(net)
        #net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        #w_init = tflearn.initializations.xavier(uniform=False, seed=None, dtype=tf.float32)
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='sigmoid', weights_init=w_init)    #modified from tangh to sigmoid
        # Scale output to action_low and action_high
        scaled_out = tf.multiply(out, self.action_range) + self.action_low #modified for the quadcopter project
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


'''

from keras import layers, models, optimizers
from keras import backend as K
from keras import regularizers

class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, 
                           activation='relu', 
                           use_bias = True,
                           kernel_initializer='glorot_uniform', 
                           bias_initializer='zeros', 
                           kernel_regularizer=None, 
                           bias_regularizer=None, 
                           activity_regularizer=None, 
                           kernel_constraint=None, 
                           bias_constraint=None)(states)
        net = layers.Dense(units=64, 
                           activation='relu', 
                           use_bias = True,
                           kernel_initializer='glorot_uniform', 
                           bias_initializer='zeros', 
                           kernel_regularizer=None, 
                           bias_regularizer=None, 
                           activity_regularizer=None, 
                           kernel_constraint=None, 
                           bias_constraint=None)(net)


        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, 
                                   activation='sigmoid',
                                   name='raw_actions', 
                                   use_bias = True,
                                   kernel_initializer='glorot_uniform', 
                                   bias_initializer='zeros', 
                                   kernel_regularizer=None, 
                                   bias_regularizer=None, 
                                   activity_regularizer=None, 
                                   kernel_constraint=None, 
                                   bias_constraint=None)(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
'''