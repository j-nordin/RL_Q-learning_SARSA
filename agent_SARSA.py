import numpy as np


# SARSA AGENT
class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.Q = np.random.rand(state_space, action_space)
        self.epsilon = 0.05
        self.discount_factor = 0.95
        self.learning_rate = 0.1
        self.last_state = None
        self.last_action = None

    def observe(self, observation, reward, done):
        if not done:
            # Use greedy policy to choose a_prime
            s = np.random.uniform(0,1)
            if s < self.epsilon:
                # Choose action randomly with prob epsilon
                a_prime = np.random.randint(self.action_space)
            else:
                # Greedy
                a_prime = np.argmax(self.Q[self.last_state,:])
            
            self.Q[self.last_state, self.last_action] += self.learning_rate * (reward + self.discount_factor * self.Q[observation, a_prime] - self.Q[self.last_state, self.last_action]) 
        else:
            self.Q[self.last_state, self.last_action] += self.learning_rate * (reward - self.Q[self.last_state, self.last_action])
        
    def act(self, observation):
        # Save current state
        try:
            self.last_state = observation[0]
        except Exception:
            self.last_state = observation
                    
        s = np.random.uniform(0,1)
        if s < self.epsilon:
            # Choose action randomly with prob epsilon
            a = np.random.randint(self.action_space)
        else:
            # Greedy
            a = np.argmax(self.Q[self.last_state,:])
        
        # Save last action
        self.last_action = a
        return a