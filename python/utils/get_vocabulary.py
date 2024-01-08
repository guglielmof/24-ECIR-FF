import os
import pyterrier as pt
from pathlib import Path
import ir_datasets
from ir_datasets.util.download import DownloadConfig
import string
from tqdm import tqdm

os.environ['JAVA_HOME'] = '/ssd/data/faggioli/SOFTWARE/jdk-11.0.11'

os.environ['IR_DATASETS_HOME'] = "/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/IR_DATASETS_CORPORA/"

dataset = ir_datasets.load("beir/trec-covid")

vocab = set()

docs = []

translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))


def process_field(txt): return txt.lower().replace("\n", " ").translate(translator)


for doc in tqdm(dataset.docs_iter()):
    fields = [f for f in doc._fields if f != "doc_id"]
    content = " ".join(list(map(process_field, [getattr(doc, f) for f in fields])))
    docs.append(content.split())

vocab = set().union(*docs)

with open("/ssd/data/faggioli/24-ECIR-FF/data/vocabularies/trec-covid.txt", "w") as f:
    for l in vocab:
        f.write(f"{l}\n")
