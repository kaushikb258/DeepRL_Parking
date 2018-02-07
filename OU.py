# Ornstein-Uhlenbeck noise


import random
import numpy as np 

class OU(object):

    def function(self, x, mu, theta, sigma, dt):
        return theta * (mu - x) + sigma/np.sqrt(dt) * np.random.randn()
