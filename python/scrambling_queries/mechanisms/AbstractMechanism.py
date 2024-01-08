import numpy as np
import numpy.random as npr
import os

class AbstractMechanism:

    def __init__(self, m, epsilon=1, **kwargs):
        self.epsilon = epsilon
        self.m = m

    def get_protected_vectors(self, embeddings):

        os.environ["OMP_NUM_THREADS"] = "1"

        noisy_embeddings = []
        for e in embeddings:
            noise = self.noise_sampling()

            noisy_embeddings.append(e+noise)

        return np.array(noisy_embeddings)
