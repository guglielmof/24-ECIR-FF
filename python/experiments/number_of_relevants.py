import argparse

import pandas as pd
import pyterrier as pt
from glob import glob
import numpy as np
from tqdm import tqdm
from time import time
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection")
args = parser.parse_args()

collections = {'robust04': 'irds:disks45/nocr/trec-robust-2004',
               'trec-covid': 'irds:beir/trec-covid',
               'msmarco-passage': 'irds:msmarco-passage/trec-dl-2019/judged'}

if not pt.started():
    pt.init()

dataset = pt.get_dataset(collections[args.collection])
paths = glob(f"../../data/runs/{args.collection}_*")
paths = [p for p in paths if "glove" not in p]

qrels = dataset.get_qrels().drop(["iteration"], axis=1).rename({"qid": "query_id", "docno": "doc_id", "label": "relevance"}, axis=1)
qrels.query_id = qrels.query_id.astype(str)
qrels.doc_id = qrels.doc_id.astype(str)

relevant = qrels[qrels.relevance >= 1]


def parse_run(path):
    run = pd.read_csv(path, names=["query_id", "doc_id", "score"], usecols=[0, 2, 4], sep="\t", dtype={"query_id": str, "doc_id": str})
    run = run.sort_values(['score'], ascending=False).groupby('query_id').head(100)
    if "topics.csv" not in path:
        run[['query_id', 'rep']] = run.query_id.str.split("_", expand=True)
    else:
        run['rep'] = 0

    run = run.merge(relevant[['query_id', 'doc_id']], on=['query_id', 'doc_id'])
    idx = path.rsplit(".", 1)[0].split("/")[-1].replace("_TF_IDF_", "_TFIDF_").split("_")
    if len(idx) == 4:
        collection, model, mechanism, epsilon = idx
    else:
        collection, model, mechanism = idx
        epsilon = np.NAN

    run['model'] = model
    run['mechanism'] = mechanism
    run['epsilon'] = float(epsilon)
    return run


with Pool(processes=70) as pool:
    future = [pool.apply_async(parse_run, [p]) for p in paths]
    runs = [fr.get() for fr in future]

runs = pd.concat(runs)
relevant_retrieved = runs[['query_id', 'doc_id', 'model', 'mechanism', 'epsilon']].drop_duplicates().groupby(
    ['query_id', 'model', 'mechanism', 'epsilon'], dropna=False).count().reset_index().rename({"doc_id": 'retrieved'}, axis=1)
recall_base = relevant[['query_id', 'doc_id']].groupby(["query_id"]).count().reset_index().rename({"doc_id": "recall_base"}, axis=1)
relevant_retrieved = relevant_retrieved.merge(recall_base)
relevant_retrieved['perc'] = relevant_retrieved['retrieved'] / relevant_retrieved['recall_base']
print(relevant_retrieved[relevant_retrieved.mechanism.str.contains("Mechanism")][['mechanism', 'epsilon', 'model', 'perc']]
      .groupby(['mechanism', 'epsilon', 'model'], dropna=False)
      .mean().reset_index()
      .pivot_table(index=["model", "mechanism"], columns="epsilon", values="perc")
      .to_string())
print(relevant_retrieved[~relevant_retrieved.mechanism.str.contains("Mechanism")][['mechanism', 'model', 'perc']]
      .groupby(['model', 'mechanism'], dropna=False)
      .mean().reset_index()
      .pivot_table(index="mechanism", columns="model", values="perc")
      .to_string())
# for each model and for each query, take the documents retrieved at least once
'''
# for each mechanism, how many unique relevants are retrieved?
rel_retrieved_times = runs[["query_id", "model", "doc_id", "mechanism", "epsilon", "rep"]].groupby(
    ["query_id", "model", "doc_id", "mechanism", "epsilon"], dropna=False).count().reset_index()
rel_retrieved_times = relevant.merge(rel_retrieved_times, how="left", on=["query_id", "doc_id"])


def plot_hist(df):
    out = df[['query_id', 'doc_id', 'rep']].fillna(0)
    print(out)


for model in ['BM25', 'TF-IDF', 'glove', 'tasb', 'contriever']:
    for mechanism in ['CMPMechanism', 'MahalanobisMechanism', 'VikreyCMPMechanism', 'VikreyMMechanism', 'ArampatzisEtAl2011']:
        if "Mechanism" in mechanism:
            for epsilon in [1., 5., 10., 12.5, 15., 17.5, 20., 35., 50.]:
                plot_hist(rel_retrieved_times[((rel_retrieved_times.model == model) & (rel_retrieved_times.mechanism == mechanism) & (
                        rel_retrieved_times.epsilon == epsilon))])
        else:
            plot_hist(rel_retrieved_times[((rel_retrieved_times.model == model) & (rel_retrieved_times.mechanism == mechanism))])
        break
    break
'''
