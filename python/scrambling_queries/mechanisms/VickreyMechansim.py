import numpy as np
import numpy.random as npr
from sklearn.preprocessing import normalize
from scipy.linalg import sqrtm
from .MahalanobisMechanism import MahalanobisMechanism
from utils.Timer import Timer


class VickreyMechanism(MahalanobisMechanism):
    """
    Zekun Xu, Abhinav Aggarwal, Oluwaseyi Feyisetan, Nathanael Teissier: On a Utilitarian Approach to Privacy Preserving Text Generation.
    CoRR abs/2104.11838 (2021)
    """

    def __init__(self, m, epsilon=1, **kwargs):
        super().__init__(m, epsilon, **kwargs)
        self.lam = kwargs['lambda'] if 'lambda' in kwargs else 0.75

    def get_protected_vectors(self, embeddings):

        n_words = len(embeddings)
        noisy_embeddings = []
        for e in embeddings:
            noisy_embeddings.append(e + self.noise_sampling())

        def euclidean_distance_matrix(x, y):
            x_expanded = x[:, np.newaxis, :]  # Shape: [n, 1, d]
            y_expanded = y[np.newaxis, :, :]  # Shape: [1, m, d]

            return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))

        noisy_embeddings = np.array(noisy_embeddings)
        distance = euclidean_distance_matrix(noisy_embeddings, self.emb_matrix)

        closest = np.argpartition(distance, 2, axis=1)[:, :2]
        dist_to_closest = distance[np.tile(np.arange(n_words).reshape(-1, 1), 2), closest]

        p = ((1 - self.lam) * dist_to_closest[:, 1]) / (self.lam * dist_to_closest[:, 0] + (1 - self.lam) * dist_to_closest[:, 1])

        vickrey_choice = np.array([npr.choice(2, p=[p[w], 1 - p[w]]) for w in range(n_words)])
        noisy_embeddings = self.emb_matrix[closest[np.arange(n_words), vickrey_choice]]

        return noisy_embeddings
