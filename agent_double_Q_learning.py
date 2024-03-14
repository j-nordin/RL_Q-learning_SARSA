import numpy as np


# DOUBLE Q-LEARNING AGENT
class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.discount_factor = 0.95
        self.learning_rate = 0.1
        self.epsilon = 0.05
        self.Q1_table = np.random.rand(state_space, action_space) # table
        self.Q2_table = np.random.rand(state_space, action_space) 
        self.Q_table = None # Sum of Q1 and Q2
        self.last_state = None
        self.last_action = None

    def observe(self, observation, reward, done):
        
        prob = np.random.uniform(0,1)
        if prob < 0.5:
            if not done:
                self.Q1_table[self.last_state, self.last_action] += self.learning_rate * (reward + self.discount_factor * self.Q2_table[observation, np.argmax(self.Q1_table[observation,:])] - self.Q1_table[self.last_state, self.last_action]) 
            else:
                self.Q1_table[self.last_state, self.last_action] += self.learning_rate * (reward - self.Q1_table[self.last_state, self.last_action])
        else:
            if not done:
                self.Q2_table[self.last_state, self.last_action] += self.learning_rate * (reward + self.discount_factor * self.Q1_table[observation, np.argmax(self.Q2_table[observation,:])] - self.Q2_table[self.last_state, self.last_action]) 
            else:
                self.Q2_table[self.last_state, self.last_action] += self.learning_rate * (reward - self.Q2_table[self.last_state, self.last_action])
    
    def act(self, observation):
        # Save current state
        try:
            self.last_state = observation[0]
        except Exception:
            self.last_state = observation

        self.Q_table = self.Q1_table + self.Q2_table
                    
        s = np.random.uniform(0,1)
        if s < self.epsilon:
            # Choose action randomly with prob epsilon
            a = np.random.randint(self.action_space)
        else:
            # Greedy
            a = np.argmax(self.Q_table[self.last_state,:])
        
        # Save last action
        self.last_action = a
        return a
    