import pandas as pd
from multiprocessing import Pool

import math


def get_duplicates(lines):
    dupset = set()
    for l in lines:
        dups = l.strip().split(":")
        if len(dups[1]) > 1:
            dups = set([did[6:] for did in dups[1].split(",")])
            dupset = dupset.union(dups)

    return dupset


def chunk_based_on_number(lst, chunk_numbers):
    n = math.ceil(len(lst) / chunk_numbers)

    chunks = []
    for x in range(0, len(lst), n):
        each_chunk = lst[x: n + x]
        chunks.append(each_chunk)

    return chunks


class msmarco:

    def __init__(self,
                 basepath="/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/MSMARCO-PASSAGES/",
                 path="collection/collection.tsv", duppath="duplicate_list_v1.0.txt", ncores=72):
        self.basepath = basepath
        self.duppath = basepath + duppath
        self.path = basepath + path
        self.ncores = ncores
        self.sep = "\t"
        self.dupset = set()

    def read_duplicates(self):
        self.dupset = set()

        with open(self.duppath, "r") as fp:
            lines = fp.readlines()
            lchunks = chunk_based_on_number(lines, self.ncores)
            with Pool(processes=self.ncores) as pool:
                futureDups = [pool.apply_async(get_duplicates, [c]) for c in lchunks]
                dups = [fr.get() for fr in futureDups]

            self.dupset = set().union(*dups)

        return self

    def read_collection(self):
        self.read_duplicates()
        dataset = pd.read_csv(self.path, sep=self.sep, header=None, names=["did", "body"],
                              dtype={"did": str, "body": str}).dropna()
        self.corpus = dataset[~dataset.did.isin(self.dupset)].reset_index(drop=True)

        return self

    def get_corpus(self):
        return self.corpus


if __name__ == "__main__":
    msmarco().read_collection()
