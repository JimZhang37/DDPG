import tensorflow as tf
import numpy as np
import tflearn
from replay_buffer import ReplayBuffer
#from ounoise import OrnsteinUhlenbeckActionNoise
from actor import ActorNetwork
from critic import CriticNetwork
from ounoise import OUNoise
class DDPG():
    def __init__(self, task, sess):
        self.sess = sess
        self.env = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
            
        self.actor_lr = 0.0001
        self.tau = 0.001
        self.minibatch_size = 64
        self.critic_lr = 0.001
        self.gamma = 0.99
        self.buffer_size = 1000000
        self.random_seed = 1234
        self.summary_dir = "/"
        #self.max_episode = 100
        #self.max_episode_len = 100
        self.mu = 0  
        
        
        
        self.actor = ActorNetwork(self.sess, 
                                  self.state_size, 
                                  self.action_size, 
                                  self.action_low, 
                                  self.action_high, 
                                  self.actor_lr, 
                                  self.tau, 
                                  self.minibatch_size)

        self.critic = CriticNetwork(self.sess, 
                                    self.state_size, 
                                    self.action_size,
                                    self.critic_lr, self.tau,
                                    self.gamma,
                                    self.actor.get_num_trainable_vars())

        

        
        # Initialize replay memory
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.random_seed)
        self.sess.run(tf.global_variables_initializer())        
        self.actor.update_target_network()
        self.critic.update_target_network()
        
        self.noise = OUNoise(self.action_size, self.mu)
        
        self.sess.run(tf.global_variables_initializer())
        
    def reset_episode(self):
        #self.actor_noise.reset()
        state = self.env.reset()
        self.last_state = state
        self.ep_ave_max_q = 0
        self.ep_reward = 0
        return state
    
    def step(self, s, a, r, terminal, s2):
         # Save experience / reward
        #self.memory.add(self.last_state, action, reward, next_state, done)
        #summary_ops, summary_vars = self.build_summaries()
        self.replay_buffer.add(np.reshape(s, (self.actor.s_dim,)), np.reshape(a, (self.actor.a_dim,)), r,
                              terminal, np.reshape(s2, (self.actor.s_dim,)))
        # Learn, if enough samples are available in memory
        if self.replay_buffer.size() > self.minibatch_size:
            
            s_batch, a_batch, r_batch, t_batch, s2_batch = self.replay_buffer.sample_batch(self.minibatch_size)
            #self.train(s_batch, a_batch, r_batch, t_batch, s2_batch)
            target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

            y_i = []
            for k in range(self.minibatch_size):
                if t_batch[k]:
                    y_i.append(r_batch[k])
                else:
                    y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

                        # Update the critic given the targets
            predicted_q_value, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i, (self.minibatch_size, 1)))

            #self.ep_ave_max_q += np.amax(predicted_q_value)

            # Update the actor policy using the sampled gradient
            a_outs = self.actor.predict(s_batch)
            grads = self.critic.action_gradients(s_batch, a_outs)
            self.actor.train(s_batch, grads[0])

            # Update target networks
            self.actor.update_target_network()
            self.critic.update_target_network()

        # Roll over last state and action
        self.last_state = s2
        '''
        self.ep_reward +=r
        
        if terminal:
            
            summary_str = self.sess.run(
            , feed_dict={summary_vars[0]: self.ep_reward, summary_vars[1]: self.ep_ave_max_q / float(j)})

            writer.add_summary(summary_str, i)
            #writer.flush()
            
            print('| Reward: {:d} |Qmax: {:.4f}'.format(int(self.ep_reward), \
                             (self.ep_ave_max_q / float(j))))
             '''       
        
    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        states = np.reshape(states, [-1, self.state_size])

        actions = self.actor.predict(states)[0]
        #actornoises = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_size))
        #print(actions)                            

        return actions + self.noise.sample()  # add some noise for exploration

        
    def train(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        
        
                    
        target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch))

        y_i = []
        for k in range(self.minibatch_size):
            if t_batch[k]:
                y_i.append(r_batch[k])
            else:
                y_i.append(r_batch[k] + self.critic.gamma * target_q[k])

                    # Update the critic given the targets
        predicted_q_value, _ = self.critic.train(s_batch, a_batch, np.reshape(y_i, (self.minibatch_size, 1)))
        
        #self.ep_ave_max_q += np.amax(predicted_q_value)

                    # Update the actor policy using the sampled gradient
        a_outs = self.actor.predict(s_batch)
        grads = self.critic.action_gradients(s_batch, a_outs)
        self.actor.train(s_batch, grads[0])

                    # Update target networks
        self.actor.update_target_network()
        self.critic.update_target_network()

    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar("Reward", episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar("Qmax Value", episode_ave_max_q)

        summary_vars = [episode_reward, episode_ave_max_q]
        summary_ops = tf.summary.merge_all()

        return summary_ops, summary_vars

'''
# TODO: your agent here!
from task import Task
from actor import Actor
from critic import Critic
from ounoise import OUNoise
from replaybuffer import ReplayBuffer
import numpy as np


class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 50000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.001  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state
    
    def pretrain(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)



        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        #print("state")
        #print(state)
        action = self.actor_local.model.predict(state)[0]
        print(action)
        #print("action in agent")
        #print(action)
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)
'''