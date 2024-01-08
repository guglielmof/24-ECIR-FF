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


def preprocess_document(doc, collection):
    if collection not in ["msmarco-passage"]:
        fields = [f for f in doc._fields if f != "doc_id"]

        #translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

        def process_field(txt): return txt.lower().replace("\n", " ")

        text = " ".join([process_field(getattr(doc, f)) for f in fields])
        idx = doc.doc_id
        doc = {"doc_id": idx, "text": text}
    else:
        doc["doc_id"] = doc["docno"]
    return doc["doc_id"], doc["text"]

def process_msmarco(doc):
    return doc['doc_id'], doc['text']

collid2irdspath = {
    'robust04': "disks45/nocr/trec-robust-2004",
    'msmarco-passage': 'irds:msmarco-passage'
}

mname2modelid = {
    'contriever': "facebook/contriever-msmarco",
    'tasb': "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
    'glove': "sentence-transformers/average_word_embeddings_glove.6B.300d"
}

# Important, you need to shield your code with if __name__. Otherwise, CUDA runs into issues when spawning new processes.
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection")
    parser.add_argument("-m", "--model_name")
    args = parser.parse_args()

    dataset = pt.get_dataset(collid2irdspath[args.collection])
    iterator = dataset.get_corpus_iter()
    #dataset = ir_datasets.load(collid2irdspath[args.collection])
    #iterator = dataset.docs_iter()
    sentences = []
    doc_ids = []
    for d in iterator:
        doc_id, doc_text = preprocess_document(d, args.collection)
        sentences.append(doc_text)
        doc_ids.append(doc_id)

    # Define the model
    model = SentenceTransformer(mname2modelid[args.model_name])

    # Start the multi-process pool on all available CUDA devices
    pool = model.start_multi_process_pool()
    docs_emb = []
    step = 100
    start = time()
    for i in tqdm(range(0, len(sentences), step)):
        # Compute the embeddings using the multi-process pool
        docs_emb.append(model.encode_multi_process(sentences[i:i + step], pool))

    docs_emb = np.concatenate(docs_emb)
    print(f"Embeddings computed in {time() - start:.2f}s. Shape: {docs_emb.shape}")

    # Optional: Stop the proccesses in the pool
    model.stop_multi_process_pool(pool)

    print(f"Writing the index")
    start = time()
    faiss_index = faiss.IndexFlatL2(docs_emb.shape[1])  # L2 distance (Euclidean distance) index for 128-dimensional vectors
    faiss_index.add(docs_emb)

    #create the directory where to save the files, if it does not exist
    Path(f"/home/ims/faggioli/24-ECIR-FF/data/indexes/{args.collection}/faiss/{args.model_name}").mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, f"/home/ims/faggioli/24-ECIR-FF/data/indexes/{args.collection}/faiss/{args.model_name}/{args.model_name}.faiss")
    with open(f"/home/ims/faggioli/24-ECIR-FF/data/indexes/{args.collection}/faiss/{args.model_name}/{args.model_name}.map", "w") as fp:
        for idx in doc_ids:
            fp.write(f"{idx}\n")

    print(f"Index written computed in {time() - start:.2f}s.")
