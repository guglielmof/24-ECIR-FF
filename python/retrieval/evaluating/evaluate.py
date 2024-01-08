import argparse

import pandas as pd
import pyterrier as pt
from ir_measures import AP, nDCG, P, Recall, iter_calc
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection")
parser.add_argument("-r", "--run")
parser.add_argument("-v", "--verbose", type=bool, default=False)
parser.add_argument("-b", "--binary", type=int)
args = parser.parse_args()

collections = {'robust04': 'irds:disks45/nocr/trec-robust-2004',
               'trec-covid': 'irds:beir/trec-covid',
               'msmarco-passage': 'irds:msmarco-passage/trec-dl-2019/judged'}

if not pt.started():
    pt.init()

dataset = pt.get_dataset(collections[args.collection])
if args.run is None:
    paths = glob(f"../../data/runs/{args.collection}_*")
else:
    paths = glob(f"../../data/runs/{args.collection}_{args.run}*")
measures = [nDCG @ 10]

qrels = dataset.get_qrels().drop(["iteration"], axis=1).rename({"qid": "query_id", "docno": "doc_id", "label": "relevance"}, axis=1)
qrels.query_id = qrels.query_id.astype(str)
qrels.doc_id = qrels.doc_id.astype(str)
if args.binary is not None:
    qrels.loc[qrels.relevance < args.binary, 'relevance'] = 0
    qrels.loc[qrels.relevance >= args.binary, 'relevance'] = 1


def compute_measure(run, qrels):
    out = pd.DataFrame(iter_calc(measures, qrels, run))
    out['measure'] = out['measure'].astype(str)
    return out


for p in paths:
    run = pd.read_csv(p, names=["query_id", "doc_id", "score"], usecols=[0, 2, 4], sep="\t", dtype={"query_id": str, "doc_id": str})
    if not "topics.csv" in p:
        print(p)
        run[['query_id', 'rep']] = run['query_id'].str.split("_", expand=True)
        out = run.groupby("rep").apply(lambda x: compute_measure(x, qrels)).reset_index().drop('level_1', axis=1)
    else:
        out = pd.DataFrame(iter_calc(measures, qrels, run))

    out['measure'] = out['measure'].astype(str)

    filename = p.split("/").pop()
    if args.verbose:
        print(filename)
        print(out[['measure', 'value']].groupby("measure").mean())
    out.to_csv(f"../../data/measures/{filename}", sep="\t", index=False)
