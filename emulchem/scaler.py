import numpy as np
import pickle

class MinMaxScaler():
    """Reimplementation of scikit.learn MinMaxSaler. Avoids the need for pickling scikitlearn objects."""
    def __init__(self,x_scale,x_min):
        self.x_scale = np.array(x_scale).astype(float)
        self.x_min = np.array(x_min).astype(float)

    def transform(self,X):
        X = np.array(X).astype(float)
        print(X)
        print(self.x_scale)
        X*=self.x_scale
        X+=self.x_min
        return X

    def inverse_transform(self,X):
        X = np.array(X).astype(float)
        X -= self.x_min
        X /= self.x_scale
        return X



