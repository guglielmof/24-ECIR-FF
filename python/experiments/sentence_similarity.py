


def sentence_similarity(s1, s2):

    w1 = set(s1.split())
    w2 = set(s2.split())

    return len(w1.intersection(w2))/len(w1.union(w2))


import pandas as pd
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection", default="robust04")
parser.add_argument("-s", "--scrambler")
args = parser.parse_args()


def clean(string):
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    translator = str.maketrans(punctuation, ' ' * len(punctuation))

    return string.lower().replace("-", "").translate(translator)


# import default run
queries = pd.read_csv(f"../../data/queries/{args.collection}/valid/topics.csv", sep=";", header=None, names=["qid", "query"], dtype={"qid": str})
queries['query'] = queries['query'].apply(clean)

paths = glob(f"../../data/queries/{args.collection}/valid/*.csv")
paths = [p for p in paths if "/topics.csv" not in p]

scrambled = []
for p in paths:
    # import scrambled queries
    scrambled.append(pd.read_csv(p, sep=";", header=None, names=["qid", "query"], dtype={"qid": str}))
    scrambled[-1][['qid', 'rep']] = scrambled[-1].qid.str.split("_", expand=True)
    scrambled[-1]['scrambling'] = p.split("/")[-1].rsplit(".", 1)[0]
scrambled = pd.concat(scrambled)

scrambled = queries.merge(scrambled, on="qid", suffixes=("", "_noisy"))

scrambled['similarity'] = scrambled.apply(lambda x: sentence_similarity(x['query'], x['query_noisy']), axis=1)
scrambled[['scrambling', 'epsilon']] = scrambled.scrambling.str.split("_", expand=True)
scrambled.epsilon = scrambled.epsilon.astype(float)


perf = scrambled.loc[scrambled.scrambling.str.contains("Mechanism")][['scrambling', 'epsilon', 'similarity']].groupby(["scrambling", "epsilon"],
                                                                                                             dropna=False).mean().pivot_table(
    index="scrambling", columns="epsilon", values="similarity")
sota = scrambled.loc[~scrambled.scrambling.str.contains("Mechanism")][['scrambling', 'similarity']].groupby(["scrambling"], dropna=False).mean()

print(perf.to_string())
print(sota.to_string())
