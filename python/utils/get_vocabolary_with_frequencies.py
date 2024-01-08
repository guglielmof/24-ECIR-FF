import os
import pyterrier as pt
from pathlib import Path
import ir_datasets
from ir_datasets.util.download import DownloadConfig
import string
from tqdm import tqdm
from collections import Counter, namedtuple
import json
import argparse

if not pt.started():
    pt.init()

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection", default="robust04")  # "convdr"
parser.add_argument("-k", "--topk", default=-1, type=int)  # "convdr"
args = parser.parse_args()

def msmarco_generate():
    Doc = namedtuple('Doc', 'doc_id text')
    msmarcofile = '/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/MSMARCO-PASSAGES/collection/collection.tsv'
    with pt.io.autoopen(msmarcofile, 'rt') as corpusfile:
        for l in corpusfile:
            try:
                docno, passage = l.split("\t")
            except Exception as e:
                print(l)
                pass
            yield Doc(docno, passage)


os.environ['JAVA_HOME'] = '/ssd/data/faggioli/SOFTWARE/jdk-11.0.11'

os.environ['IR_DATASETS_HOME'] = "/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/IR_DATASETS_CORPORA/"

if args.collection == "robust04":
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
    iterator = dataset.docs_iter()
elif args.collection=="msmarco-passage":
    iterator = msmarco_generate()

vocpath = f"/ssd/data/faggioli/24-ECIR-FF/data/vocabularies/{args.collection}-vocab-frequencies.txt"

translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
def process_field(txt): return txt.lower().replace("\n", " ").translate(translator)

if not os.path.exists(vocpath):

    vocab = {}

    for doc in tqdm(iterator):
        fields = [f for f in doc._fields if f != "doc_id"]
        content = " ".join(list(map(process_field, [getattr(doc, f) for f in fields])))
        frequency = Counter(content.split())

        for w, f in frequency.items():
            if w in vocab:
                vocab[w] += f
            else:
                vocab[w] = f

    with open(vocpath, "w") as f:
        json.dump(vocab, f)

else:
    with open(vocpath, "r") as f:
        vocab = json.load(f)

topkwords = [w for w, _ in sorted(list(vocab.items()), key=lambda x: -x[1])][:args.topk]

with open(f"/ssd/data/faggioli/24-ECIR-FF/data/vocabularies/{args.collection}-vocab-{args.topk}.txt", "w") as f:
    for l in topkwords:
        f.write(f"{l}\n")
