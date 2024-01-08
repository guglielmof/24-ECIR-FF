import numpy as np
import pandas as pd
from glob import glob


def import_noisy_queries(collection, n_queries):
    '''
    mechanisms = ["MahalanobisMechanism", "CMPMechanism", "VickreyMMechanism", "VickreyCMPMechanism"]
    for m in mechanisms:
        df = import_pd_noisy_queries(collection, m, n_queries)
        df['qid'] = df.apply(lambda x: f"{x['qid']}_{x['rep']}", axis=1)
        for e in df['eps'].unique():
            df.loc[df.eps == e][['qid', 'query']].to_csv(f"../../data/queries/{collection}/valid/{m}_{e}.csv", header=False, index=False, sep=";")

    scramblers = ["ArampatzisEtAl2011", "FroebeEtAl2021"]
    '''
    scramblers = ["FroebeEtAl2021b"]
    for s in scramblers:
        df = import_sota_noisy_queries(collection, s, n_queries)
        df['qid'] = df.apply(lambda x: f"{x['qid']}_{x['rep']}", axis=1)
        df[['qid', 'query']].to_csv(f"../../data/queries/{collection}/valid/{s}.csv", header=False, index=False, sep=";")


def import_pd_noisy_queries(collection, mechanism, n_queries):
    files = glob(f"../../data/queries/{collection}/{mechanism}/*.csv")
    files = [f for f in files if int(f.rsplit("_", 1)[1].split(".")[0]) < n_queries]

    dfs = []
    for f in files:
        eps, _, rep = f.split("/")[-1].rsplit(".", 1)[0].split("_")
        df = pd.read_csv(f, header=None, names=["qid", "query"], sep=";")
        df['eps'] = eps
        df['rep'] = rep
        dfs.append(df)
    df = pd.concat(dfs)

    return df


def keep_k_median(df, k):
    if len(df) <= k:
        return df
    else:
        median = df['similarity'].median()
        df['similarity'] = 1 - np.absolute(df['similarity'] - median)

        return df.nlargest(k, ['similarity'])


def import_sota_noisy_queries(collection, scrambler, n_queries, **kwargs):
    filename = f"../../data/queries/{collection}/{scrambler}/{scrambler}.csv"
    df = pd.read_csv(filename, header=None, usecols=[0, 2, 3], names=["qid", "query", "similarity"], sep=";")

    if scrambler == "ArampatzisEtAl2011":
        df = df.groupby("qid").apply(lambda x: keep_k_median(x, n_queries)).reset_index(drop=True)
    else:
        df = df[~df[['qid', 'query']].duplicated()]
        df = df.sort_values("similarity", ascending=False).groupby("qid").head(n_queries)

    df['rep'] = df.groupby('qid').cumcount() + 1
    df = df[df['rep'] <= n_queries][["qid", "query", "rep"]]
    print(df)

    def replacing(s):
        return str(s).replace("_", " ")

    df['query'] = df['query'].apply(replacing)

    # print(df[["qid", "rep"]].groupby("qid").count())
    return df

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection")
args = parser.parse_args()
import_noisy_queries(args.collection, 20)
