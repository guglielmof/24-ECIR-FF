import numpy as np
import numpy.random as npr
from sklearn.preprocessing import normalize
from scipy.linalg import sqrtm
from AbstractMechanism import AbstractMechanism
import scipy


class MahalanobisMechanism(AbstractMechanism):
    """
    Xu, Zekun, Abhinav Aggarwal, Oluwaseyi Feyisetan, and Nathanael Teissier. "A Differentially Private Text Perturbation Method Using Regularized
    Mahalanobis Metric." In Proceedings of the Second Workshop on Privacy in NLP, pp. 7-17. 2020.

    The mechanism is specifically designed to work with embeddings trained in an hyperbolic space and therefore it is not really sure whether it could
    be used also for vectors trained in an euclidean space as in the case of glove.
    Furthermore the mathematical part is fundamentally poorly written with many unclear passages that make it challenging to reproduce it
    """

    def __init__(self, epsilon=1, **kwargs):
        super().__init__(epsilon, **kwargs)

        self.m = kwargs['m']
        self.b = kwargs['b'] if 'b' in kwargs else 10

        self.hyp2f1_val = scipy.special.hyp2f1(1, self.epsilon, 2 + self.epsilon, -1)

    def enorm(self, x):
        """euclidean norm"""
        return np.sqrt(np.sum(x ** 2))

    def proba(self, x):
        """EQ 7 on the papaer"""
        return (1 + self.epsilon) / (2 * self.hyp2f1_val) * (-2 / (self.enorm(x)) - 1) ** (-self.epsilon)

    def lorentzTranslation(self, x):
        """EQ 4 on the paper"""
        return np.sqrt(1 + np.sum(x)), x

    def poincarreTransation(self, x):
        x_0, x_p = x[0], x[1]
        return x_p / (1 + x_0)

    def noise_sampling(self, k):
        x_0 = np.zeros(self.m)
        x_0[0] = 1
        x_t = x_0

        outputs = []
        for i in range(k + self.b):

            x_p = npr.multivariate_normal(np.zeros(self.m), np.eye(self.m))
            x_p = self.poincarreTransation(self.lorentzTranslation(x_p))
            alpha = self.proba(x_p) / self.proba(x_t)

            if npr.random() <= alpha:
                x_t = x_p
            else:
                x_t = x_t

            if i > self.b:
                outputs.append(x_t)

        return outputs
