import pandas as pd
import argparse
from search_faiss import search_faiss
from search_pyterrier import search_pyterrier
import string

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection", default="robust04")
parser.add_argument("-q", "--queries")
parser.add_argument("-m", "--model", default="tasb")
args = parser.parse_args()
# read queries
queries = pd.read_csv(f"../../data/queries/{args.collection}/valid/{args.queries}.csv", header=None, names=["qid", "query"], sep=";", dtype={"query":str})

#remove punctuation from queries
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
queries['query'] = queries['query'].apply(lambda x: str(x).translate(translator))

if args.model in ["tasb", "contriever", "glove"]:
    out = search_faiss(queries, args.collection, args.model)
else:
    out = search_pyterrier(queries, args.collection, args.model)

out.to_csv(f"../../data/runs/{args.collection}_{args.model.replace('_', '-')}_{args.queries}.csv", header=None, index=None, sep="\t")