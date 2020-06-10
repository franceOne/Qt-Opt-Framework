import numpy as np
class CEM():
    def __init__(self, theta_dim, init_mean=0.0, init_std=1.0):
        self.theta_dim = theta_dim
        self._init_mean = init_mean
        self._init_std = init_std

        self.mean = init_mean* np.ones(theta_dim)
        self.std = init_std* np.ones(theta_dim)


    def reset(self):
        self.mean = self._init_mean* np.ones(self.theta_dim)
        self.std = self._init_std* np.ones(self.theta_dim)

    def sample(self):
        theta = self.mean + np.random.normal(size=self.theta_dim)*self.std
        return theta
    
    def sample_multi(self, n):
        theta_list = []
        for i in range(n):
            theta_list.append(self.sample())
        return np.array(theta_list)

    def update(self, selected_samples):
        self.mean = np.mean(selected_samples, axis= 0)
        self.std = np.std(selected_samples, axis = 0)
        return self.mean, self.std
