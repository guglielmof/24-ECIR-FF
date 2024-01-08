import numpy as np
from multiprocessing import Pool
import utils
from scipy.spatial import distance


class glove:

    def __init__(self, path="glove.42B.300d.txt", workers=1, vocab=None):

        self.vocab = None

        if vocab is not None:
            with open(vocab, 'r', encoding='utf-8') as fp:
                self.vocab = set([w.strip() for w in fp.readlines()])

        with open(path, 'r', encoding='utf-8') as fp:
            lines = fp.readlines()

            lchunks = utils.chunk_based_on_number(lines, workers)

            with Pool(processes=workers) as pool:
                futureEmbeddings = [pool.apply_async(read_lines, [c, self.vocab]) for c in lchunks]
                temp_embeddings = [fr.get() for fr in futureEmbeddings]
                self._embeddings = {k: v for d in temp_embeddings for k, v in d.items()}

            self._embsize = len(self._embeddings[list(self._embeddings.keys())[0]])
            self._word2int = {w: i for i, w in enumerate(self._embeddings.keys())}
            self._int2word = {i: w for w, i in self._word2int.items()}
            self._embeddings_matrix = np.zeros((len(self._embeddings), self._embsize))

            for w, i in self._word2int.items():
                self._embeddings_matrix[i] = self._embeddings[w]

    def encode(self, word):
        return self._embeddings.get(word)

    def encode_sentence(self, sentence):
        return [self.encode(word) for word in sentence.split() if not self.encode(word) is None]

    def get_size(self):
        return self._embsize

    def get_embeddings_matrix(self):
        return self._embeddings_matrix

    def get_k_closest_terms(self, vectors, k):
        distance = euclidean_distance_matrix(vectors, self._embeddings_matrix)
        found = np.argpartition(distance, k, axis=1)[:, :k]

        closest_terms = self.indexes_to_words(found)
        return closest_terms

    def indexes_to_words(self, indexes):
        return [self._int2word[e] for f in indexes for e in f]

    def get_vocabulary(self):
        return list(self._word2int.keys())


def euclidean_distance_matrix(x, y):
    x_expanded = x[:, np.newaxis, :]  # Shape: [n, 1, d]
    y_expanded = y[np.newaxis, :, :]  # Shape: [1, m, d]

    return np.sqrt(np.sum((x_expanded - y_expanded) ** 2, axis=2))


def read_lines(chunk_lines, vocab=None):
    embeddings = {}

    for line in chunk_lines:
        values = line.split()
        word = values[0]
        if word.isalpha() and (vocab is None or word in vocab):
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings
