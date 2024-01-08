import argparse

import pandas as pd
import pyterrier as pt
from glob import glob
import numpy as np
from tqdm import tqdm
from time import time
from multiprocessing import Pool
from ir_measures import nDCG, iter_calc

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection")
parser.add_argument("-r", "--retrieval_model", default="tasb")
args = parser.parse_args()

collections = {'robust04': 'irds:disks45/nocr/trec-robust-2004',
               'trec-covid': 'irds:beir/trec-covid',
               'msmarco-passage': 'irds:msmarco-passage/trec-dl-2019/judged'}

if not pt.started():
    pt.init()

paths = glob(f"../../data/runs/{args.collection}_*")
paths = [p for p in paths if "glove" not in p]


def parse_run(path):
    run = pd.read_csv(path, names=["query_id", "doc_id", "score"], usecols=[0, 2, 4], sep="\t", dtype={"query_id": str, "doc_id": str})
    if "topics.csv" not in path:
        run = run.sort_values(['score'], ascending=False).groupby('query_id').head(100)
        run[['query_id', 'rep']] = run.query_id.str.split("_", expand=True)
    else:
        run['rep'] = 0

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

tot_runs = pd.concat(runs)

original = tot_runs.loc[(tot_runs['mechanism'] == 'topics') & (tot_runs['model'] == args.retrieval_model)]
runs = tot_runs.loc[(tot_runs['rep'] != '20')]
unique_retrieved = runs[['query_id', 'doc_id', 'model', 'mechanism', 'epsilon']].drop_duplicates()

filtered_runs = original[['query_id', 'doc_id', 'score']].merge(unique_retrieved)

dataset = pt.get_dataset(collections[args.collection])

qrels = dataset.get_qrels().drop(["iteration"], axis=1).rename({"qid": "query_id", "docno": "doc_id", "label": "relevance"}, axis=1)
qrels.query_id = qrels.query_id.astype(str)
qrels.doc_id = qrels.doc_id.astype(str)


def evaluate(run, qrels):
    out = pd.DataFrame(iter_calc([nDCG @ 10], qrels, run))
    out['measure'] = out['measure'].astype(str)
    return out


perf = filtered_runs.groupby(["model", "mechanism", "epsilon"], dropna=False).apply(lambda x: evaluate(x, qrels)).reset_index()

measures = perf[['model', 'mechanism', 'epsilon', 'value']].groupby(['model', 'mechanism', 'epsilon'], dropna=False).mean().reset_index()

print(measures[measures.mechanism.str.contains("Mechanism")].pivot_table(index=["model", "mechanism"], columns="epsilon", values="value").to_string())
print(measures[~measures.mechanism.str.contains("Mechanism")].pivot_table(index="mechanism", columns="model", values="value").to_string())
