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


def create_language_model(df):
    lm = {}
    strings = df['query'].values
    tot_l = 0
    for s in strings:
        for w in s.split():
            tot_l += 1
            if w not in lm:
                lm[w] = 0
            lm[w] += 1
    lm = {k: v / tot_l for k, v in lm.items()}
    return lm


lms = scrambled.groupby(["qid", "scrambling"]).apply(create_language_model).reset_index().rename({0: "lm"}, axis=1)

queries = lms.merge(queries)


# V1: language models as should be done
def compute_proba(lm, string):
    tot_proba = 1
    for i in string.split():
        tot_proba *= lm[i] if i in lm else 1/(100000)

    return tot_proba


'''
#V2 average probability of generating each word
def compute_proba(lm, string):
    tot_proba = 0
    for i in string.split():
        tot_proba += lm[i] if i in lm else 0

    return tot_proba/len(string.split())
'''

queries['proba'] = queries.apply(lambda x: compute_proba(x["lm"], x["query"]), axis=1)
queries[['scrambling', 'epsilon']] = queries.scrambling.str.split("_", expand=True)
queries.epsilon = queries.epsilon.astype(float)
# print(queries[['qid', 'query', 'proba']].sort_values("proba", ascending=False).to_string())


perf = queries.loc[queries.scrambling.str.contains("Mechanism")][['scrambling', 'epsilon', 'proba']].groupby(["scrambling", "epsilon"],
                                                                                                             dropna=False).mean().pivot_table(
    index="scrambling", columns="epsilon", values="proba")
sota = queries.loc[~queries.scrambling.str.contains("Mechanism")][['scrambling', 'proba']].groupby(["scrambling"], dropna=False).mean()

print(perf.to_string())
print(sota.to_string())
