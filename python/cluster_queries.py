import os
import pyterrier as pt
from embeddings import glove
import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


basePath = "/ssd/data/faggioli/24-ECIR-FF"
dataPath = f"{basePath}/data"
savePath = f"{dataPath}/simulated_users"

collections = {'robust04': 'irds:disks45/nocr/trec-robust-2004'}

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection", default="robust04")
parser.add_argument("-n", "--nclusters", default=10, type=int)  # "convdr"
args = parser.parse_args()


os.environ['JAVA_HOME'] = '/ssd/data/faggioli/SOFTWARE/jdk-11.0.11'
if not pt.started():
    pt.init()

dataset = pt.get_dataset(collections[args.collection])
# (optionally other pipeline components)
queries = dataset.get_topics('title')

representation = glove(f"{dataPath}/vectors/glove.42B.300d.txt", workers=60)
queries['embeddings'] = queries['query'].apply(representation.encode_sentence)
print(queries)
queries['embeddings'] = queries['embeddings'].apply(lambda x: np.mean(x, axis=0))

print(queries)

X = queries['embeddings'].to_list()

print(X)

labels = KMeans(n_clusters=args.nclusters, random_state=0, n_init="auto").fit_predict(X)

queries['cluster'] = labels

queries[['qid', 'query', 'cluster']].to_csv(f"{savePath}/{args.collection}-{args.nclusters}.csv", header=False, index=False)