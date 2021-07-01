import numpy as np
import pickle

class MinMaxScaler():
    """Reimplementation of scikit.learn MinMaxSaler. Avoids the need for pickling scikitlearn objects."""
    def __init__(self,x_scale,x_min):
        self.x_scale = x_scale
        self.x_min = x_min

    def transform(X):
        X*=self.x_scale
        X+=self.x_min
        return X

    def inverse_transform(X):
        X -= self.x_min
        X /= self.scale
        return X



