import os
import sys
sys.path.append(".")
import pyterrier as pt
from embeddings import glove
import argparse
import importlib
from pathlib import Path
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"

basePath = "/ssd/data/faggioli/24-ECIR-FF"
dataPath = f"{basePath}/data"
savePath = f"{dataPath}/queries"

collections = {'robust04': 'irds:disks45/nocr/trec-robust-2004',
               'trec-covid': 'irds:beir/trec-covid',
               'msmarco-passage': 'irds:msmarco-passage/trec-dl-2019/judged'}


def dp_mechanism(queries, args):
    print("    + importing representation")
    representation = glove(f"{dataPath}/vectors/{args.vectors}.txt",
                           vocab=f"/ssd/data/faggioli/24-ECIR-FF/data/vocabularies/{args.collection}-vocab.txt",
                           workers=1)

    print("    + representing queries")
    queries['embeddings'] = queries['query'].apply(representation.encode_sentence)

    print("    + initializing mechanism")
    class_ = getattr(importlib.import_module("scrambling_queries.mechanisms"), args.mechanism)
    mechanism = class_(representation.get_size(), epsilon=args.epsilon, embeddings=representation.get_embeddings_matrix())

    print("    + computing protected vectors and representations")
    for k in tqdm(range(args.num_perturbations)):
        queries['noisy_embeddings'] = queries['embeddings'].apply(mechanism.get_protected_vectors)

        queries['noisy_query'] = queries['noisy_embeddings'].apply(lambda x: " ".join(representation.get_k_closest_terms(x, 1)))

        # check if the path exists or create it
        Path(f"{savePath}/{args.collection}/{args.mechanism}").mkdir(parents=True, exist_ok=True)

        # save the query
        queries[["qid", "noisy_query"]].to_csv(f"{savePath}/{args.collection}/{args.mechanism}/{args.epsilon}_{args.vectors}_{k}.csv",
                                               index=False, header=False, sep=";")


def sota_approaches(queries, args, dataset):

    class_ = getattr(importlib.import_module("scrambling_queries.sota"), args.mechanism)
    scrambler = class_(dataset=dataset, collection=args.collection)

    scrambled_queries = scrambler.scramble(queries)

    # check if the path exists or create it
    Path(f"{savePath}/{args.collection}/{args.mechanism}").mkdir(parents=True, exist_ok=True)

    # save the query
    scrambled_queries.to_csv(f"{savePath}/{args.collection}/{args.mechanism}/{args.mechanism}.csv",
                                           index=False, header=False, sep=";")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", default="robust04")
    parser.add_argument("-e", "--epsilon", default=5, type=float)  # "convdr"
    parser.add_argument("-k", "--num_perturbations", default=100, type=int)  # "convdr"
    parser.add_argument('-m', '--mechanism', default="MahalanobisMechanism")
    parser.add_argument('-v', '--vectors', default="glove.6B.300d")
    args = parser.parse_args()

    print(f"computing perturbed queries for {args.collection} using {args.mechanism} with epsilon {args.epsilon}")
    os.environ['JAVA_HOME'] = '/ssd/data/faggioli/SOFTWARE/jdk-11.0.11'
    if not pt.started():
        pt.init()

    print("    + importing queries")
    qfield = {"robust04": "title", "msmarco-passage": "text", "trec-covid": "query"}
    dataset = pt.get_dataset(collections[args.collection])
    queries = dataset.get_topics(qfield[args.collection])

    # remove punctuation
    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    translator = str.maketrans(punctuation, ' ' * len(punctuation))
    queries['query'].apply(lambda x: x.replace("-", "").translate(translator))

    print("    + scrambling")
    if hasattr(importlib.import_module("scrambling_queries.mechanisms"), args.mechanism):
        dp_mechanism(queries, args)
    else:
        sota_approaches(queries, args, dataset)


