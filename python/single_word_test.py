import os
from embeddings import glove
import argparse
import importlib
from utils.Timer import Timer
import pandas as pd
import numpy as np
import random

basePath = "/ssd/data/faggioli/24-ECIR-FF"
dataPath = f"{basePath}/data"
savePath = f"{dataPath}/protected-queries"

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epsilon", default=5, type=int)  # "convdr"
parser.add_argument('-m', '--mechanism', default="MahalanobisMechanism")
parser.add_argument("-n", "--ntests", default=10, type=int)
parser.add_argument('-r', '--repetitions', default=100, type=int)
parser.add_argument('-v', '--vectors', default="glove.6B.300d")

args = parser.parse_args()

representation = glove(f"{dataPath}/vectors/{args.vectors}.txt", vocab="/ssd/data/faggioli/24-ECIR-FF/data/vocabularies/robust04-vocab-30000.txt")

class_ = getattr(importlib.import_module("scrambling_queries.mechanisms"), args.mechanism)
mechanism = class_(representation.get_size(), epsilon=args.epsilon, embeddings=representation.get_embeddings_matrix())

words = list(representation._word2int.keys())

freqs = []
for i in range(args.ntests):
    w = random.choice(words)
    reps = " ".join([w] * args.repetitions)
    perturbed = mechanism.get_protected_vectors(representation.encode_sentence(reps))
    encodings = representation.get_k_closest_terms(perturbed, 1)

    freq = np.sum([1 if t == w else 0 for t in encodings]) / args.repetitions
    freqs.append(freq)

    print(f"word: {w}: freq not changed: {freq:.3f} -- curr avg: {np.mean(freqs):.3f}")

print(np.mean(freqs))
