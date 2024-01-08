import os
import sys
sys.path.append("..")
import ir_datasets
from sentence_transformers import SentenceTransformer, util, InputExample
import faiss
import numpy as np
from tqdm import tqdm
from time import time
import torch
import argparse
from utils import chunk_based_on_number
import math

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--initial_index", type=int)
parser.add_argument("-s", "--step", type=int)
parser.add_argument("-l", "--length", type=int)
parser.add_argument("-c", "--collection")
args = parser.parse_args()


def preprocess_document(doc):
    title = doc.title.lower().replace("\n", " ")
    body = doc.body.lower().replace("\n", " ")
    text = (title + " " + body)  # namedtuple<doc_id, title, body, marked_up_doc>

    return doc.doc_id, text


coll2ir_sd = {"robust04": "disks45/nocr/trec-robust-2004"}

# Load the model
transformer = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
dataset = ir_datasets.load(coll2ir_sd[args.collection])

print(f"loading the data")
start = time()
data = []
for d in dataset.docs_iter():
    data.append(preprocess_document(d)[1])
# data = [preprocess_document(d)[1] for d in dataset.docs_iter()]

data = data[args.initial_index:args.initial_index + args.length]
print(f"done in in {time() - start:.2f}s")

n_chunks = int(math.ceil(args.length / args.step))
chunks = chunk_based_on_number(data, n_chunks)
docs_emb = []
for e, c in enumerate(chunks):
    start = time()
    docs_emb.append(transformer.encode(c, convert_to_tensor=True))
    print(f"chunk {e} done in {time() - start:.2f}s")

docs_emb = torch.cat(docs_emb, dim=0)

faiss_index = faiss.IndexFlatL2(768)  # L2 distance (Euclidean distance) index for 128-dimensional vectors

faiss_index.add(docs_emb)

faiss.write_index(faiss_index, f"/ssd/data/faggioli/24-ECIR-FF/data/indexes/{args.collection}/tasb_{args.initial_index}-{args.initial_index + args.length - 1}")
