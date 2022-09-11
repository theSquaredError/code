import gym
env = gym.make('CartPole-v0')
env.observation_space
env.action_space

''' In general we use neural network to represent the parameterised policy but here the problem is very simple
    so we can use a logistic expression to parameterise the probabilities of moving left and right.'''

'''Additionally, we will use this simple form of a policy to manually derive the gradients for the policy gradient update rule.'''

'''Let's write this up into a Python class that will act as our Logistic policy agent. 
The class will have all the methods required for calculating action probabilities and acting based on those, 
calculating grad-log-prob gradients and temporally adjusted discounted rewards and 
updating the policy parameters after the end of an episode (we will update the parameters after every episode, 
but for more difficult problems a gradient update is typically performed after a batch of episodes to make training more stable):'''

import numpy as np

class LogisticPolicy:

    def __init__(self, theta, alpha, gamma):
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma

    def logistic(self, y):
        # defintion of logistic function
        return 1/(1+np.exp(-y))
    
    def probs(self, x):
        # return the probabilities of two actions

        y = x @ self.theta
        prob0 = self.logistic(y)
        return np.array([prob0, 1-prob0])
    
    def act(self, x):
        # sample an action in proportion to probabilities

        probs = self.probs(x)
        action = np.random.choice([0,1], p = probs)

        return action, probs[action]
    
    def grad_log_p(self, x):
        # calculate grad-log-probs
        y = x @ self.theta
    
