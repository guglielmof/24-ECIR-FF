import argparse
import pandas as pd
from embeddings import glove
import string
from glob import glob

basePath = "/ssd/data/faggioli/24-ECIR-FF"
dataPath = f"{basePath}/data"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection")
parser.add_argument("-v", "--vectors", default="glove.6B.300d")
args = parser.parse_args()

# import original queries
original_queries = pd.read_csv(f"{dataPath}/queries/{args.collection}/original/topics.csv", sep=";", header=None, names=["qid", "query"])

# remove words that are not in the glove/collection vocabulary
vocab = glove(f"{dataPath}/vectors/{args.vectors}.txt", vocab=f"{dataPath}/vocabularies/{args.collection}-vocab.txt").get_vocabulary()

translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
original_queries['query'] = original_queries['query'].apply(lambda x: [w for w in x.lower().translate(translator).split() if w in vocab])

# import perturbed queries
pq_files = glob(f"{dataPath}/queries/{args.collection}/**/*.csv", recursive=True)
pq_files = [p for p in pq_files if p != f"{dataPath}/queries/{args.collection}/original/topics.csv"]
perturbed_queries = []
for f in pq_files:
    mec, setting = f.split("/")[-2:]
    eps, vec = setting.split("_")

    perturbed_queries.append(pd.read_csv(f, header=None, names=["qid", "noisy_query"], sep=";"))
    perturbed_queries[-1]['eps'] = float(eps)
    perturbed_queries[-1]['mec'] = mec
    perturbed_queries[-1]['vec'] = vec

perturbed_queries = pd.concat(perturbed_queries)
perturbed_queries.noisy_query = perturbed_queries.noisy_query.str.split()
perturbed_queries = perturbed_queries.merge(original_queries, on="qid")


# perturbed_queries[perturbed_queries['noisy_query'].str.len()!=perturbed_queries['query'].str.len()][['noisy_query', 'query']]
def jaccard(x, y):
    return len(set(x).intersection(set(y))) / len(set(x).union(set(y)))


pd.options.display.float_format = '{:.3f}'.format

perturbed_queries['similarity'] = perturbed_queries.apply(lambda x: jaccard(x.noisy_query, x.query), axis=1)
mean_perf = perturbed_queries[['eps', 'mec', 'similarity']].groupby(['eps', 'mec']).mean()
mean_perf = mean_perf.reset_index().pivot_table(index='eps', columns='mec', values='similarity')
print(mean_perf.to_string())
