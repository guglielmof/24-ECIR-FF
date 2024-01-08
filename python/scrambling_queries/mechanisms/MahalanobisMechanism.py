import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm
from .AbstractMechanism import AbstractMechanism


class MahalanobisMechanism(AbstractMechanism):
    """
    Xu, Zekun, Abhinav Aggarwal, Oluwaseyi Feyisetan, and Nathanael Teissier. "A Differentially Private Text Perturbation Method Using Regularized
    Mahalanobis Metric." In Proceedings of the Second Workshop on Privacy in NLP, pp. 7-17. 2020.
    """

    def __init__(self, m, epsilon=1, **kwargs):
        super().__init__(m, epsilon, **kwargs)

        self.lam = kwargs['lam'] if 'lam' in kwargs else 1
        self.emb_matrix = kwargs['embeddings']
        _, self.m = self.emb_matrix.shape
        cov_mat = np.cov(self.emb_matrix.T, ddof=0)
        self.sigma = cov_mat/ np.mean(np.var(self.emb_matrix.T, axis=1))
        self.sigma_loc = sqrtm(self.lam * self.sigma + (1 - self.lam) * np.eye(self.m))
        #self.sigma_loc = sqrtm(np.linalg.inv(self.lam * self.sigma + (1 - self.lam) * np.eye(self.m)))

    def noise_sampling(self):
        N = npr.multivariate_normal(np.zeros(self.m), np.eye(self.m))
        X = N / np.sqrt(np.sum(N ** 2)) #direction
        X = np.matmul(self.sigma_loc, X)
        X = X / np.sqrt(np.sum(X ** 2))

        Y = npr.gamma(self.m, 1 / self.epsilon) #distance

        Z = X*Y
        #Z = Y * np.matmul(self.sigma_loc, X)
        return Z
