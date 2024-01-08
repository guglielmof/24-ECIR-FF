"""
This example starts multiple processes (1 per GPU), which encode
sentences in parallel. This gives a near linear speed-up
when encoding large text collections.
"""

from sentence_transformers import SentenceTransformer, LoggingHandler
import logging

import faiss
import argparse
import ir_datasets
from itertools import islice
import numpy as np
from time import time
from tqdm import tqdm
from pathlib import Path
import string
import pyterrier as pt

if not pt.started():
    pt.init()

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def process_field(txt):
    return txt.lower().replace("\n", " ")


def preprocess_document(doc, collection):
    if collection == "robust04":
        doc = {"doc_id": doc["docno"], "text": process_field(doc["title"] + " " + doc["body"])}
    else:
        doc["doc_id"] = doc["docno"]
    return doc["doc_id"], doc["text"]


def process_msmarco(doc):
    return doc['doc_id'], doc['text']


collid2irdspath = {
    'robust04': "irds:disks45/nocr/trec-robust-2004",
    'msmarco-passage': 'irds:msmarco-passage'
}

mname2modelid = {
    'contriever': "facebook/contriever-msmarco",
    'tasb': "sentence-transformers/msmarco-distilbert-base-tas-b",
    'glove': "sentence-transformers/average_word_embeddings_glove.6B.300d",
    'SGPT': "Muennighoff/SGPT-125M-weightedmean-nli-bitfit",
    'ance': "sentence-transformers/msmarco-roberta-base-ance-firstp"
}

# Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection")
    parser.add_argument("-m", "--model_name")
    args = parser.parse_args()

    dataset = pt.get_dataset(collid2irdspath[args.collection])
    iterator = dataset.get_corpus_iter()
    '''
    if args.collection == "msmarco-passage":
        dataset = pt.get_dataset(collid2irdspath[args.collection])
        iterator = dataset.get_corpus_iter()
    else:
        dataset = ir_datasets.load(collid2irdspath[args.collection])
        iterator = dataset.docs_iter()
    '''
    sentences = []
    doc_ids = []
    for d in iterator:
        doc_id, doc_text = preprocess_document(d, args.collection)
        sentences.append(doc_text)
        doc_ids.append(doc_id)

    # Define the model
    model = SentenceTransformer(mname2modelid[args.model_name])

    # Start the multiprocess pool on all available CUDA devices
    pool = model.start_multi_process_pool()

    # faiss_index = faiss.IndexFlatL2(model[1].word_embedding_dimension)  # L2 distance (Euclidean distance) index for 128-dimensional vectors
    faiss_index = faiss.IndexFlatIP(model[1].word_embedding_dimension)
    step = 500
    for i in tqdm(range(0, len(sentences), step)):
        # Compute the embeddings using the multiprocess pool
        faiss_index.add(model.encode_multi_process(sentences[i:i + step], pool))

    # Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)

    # create the directory where to save the files, if it does not exist
    Path(f"/home/ims/faggioli/24-ECIR-FF/data/indexes/{args.collection}/faiss/{args.model_name}").mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, f"/home/ims/faggioli/24-ECIR-FF/data/indexes/{args.collection}/faiss/{args.model_name}/{args.model_name}.faiss")
    with open(f"/home/ims/faggioli/24-ECIR-FF/data/indexes/{args.collection}/faiss/{args.model_name}/{args.model_name}.map", "w") as fp:
        for idx in doc_ids:
            fp.write(f"{idx}\n")
