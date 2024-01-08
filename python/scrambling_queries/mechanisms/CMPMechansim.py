import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
from .AbstractMechanism import AbstractMechanism
import threading

class CMPMechanism(AbstractMechanism):
    """
    Oluwaseyi Feyisetan, Borja Balle, Thomas Drake, and Tom Diethe. 2020. Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate
    Perturbations. In Proceedings of the 13th International Conference on Web Search and Data Mining (WSDM '20). 178–186.
    https://doi.org/10.1145/3336191.3371856
    """

    def __init__(self, m, epsilon=1, **kwargs):
        super().__init__(m, epsilon, **kwargs)

    def noise_sampling(self):
        N = npr.multivariate_normal(np.zeros(self.m), np.eye(self.m))
        X = N / np.sqrt(np.sum(N ** 2))
        #X = N/np.sum(N)
        Y = npr.gamma(self.m, 1 / self.epsilon)
        Z = Y * X
        return Z
