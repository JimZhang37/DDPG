import numpy as np
import gym

class Task():

    def __init__(self):
        
        # Simulation
        
        self.sim = gym.make('Pendulum-v0')  
        self.sim.seed(1234)
        self.state_size = self.sim.observation_space.shape[0]
        self.action_size = self.sim.action_space.shape[0]
        self.action_high = self.sim.action_space.high
        self.action_low = self.sim.action_space.low
        assert (self.sim.action_space.high == -self.sim.action_space.low)
        
        print(self.state_size, self.action_size, self.action_high, self.action_low)
        self.steps = 0
        self.maxsteps = 1000
        
        '''
        self.sim = gym.make('MountainCarContinuous-v0')
        self.action_high = self.sim.action_space.high
        self.action_low = self.sim.action_space.low
        self.action_size = 1
        self.state_low = self.sim.observation_space.low
        self.state_high = self.sim.observation_space.high
        self.state_size = 2
        '''
        # Goal
        



    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        observation, reward, done, info = self.sim.step(action)
        self.steps +=1
        #print("self.steps", self.steps)
        if self.steps == self.maxsteps:
            done = True
            print("self.steps == self.maxsteps.",self.steps)
        return observation, reward, done
    
    

    def reset(self):
        """Reset the sim to start a new episode."""
        state = self.sim.reset()
        self.steps = 0
        self.maxsteps = 100
        return state