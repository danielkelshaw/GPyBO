class GP:

    def __init__(self, mean, kernel, likelihood):

        self.mean = mean
        self.kernel = kernel

        self.likelihood = likelihood

    def log_likelihood(self):
        raise NotImplementedError('GP::log_likelihood()')

    def train(self):
        raise NotImplementedError('GP::train()')
