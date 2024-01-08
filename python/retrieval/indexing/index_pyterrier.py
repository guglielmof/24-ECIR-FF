import sys
sys.path.append("..")
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--collection")
args = parser.parse_args()


def preprocess_document(doc):
    title = doc.title.lower().replace("\n", " ")
    body = doc.body.lower().replace("\n", " ")
    text = (title + " " + body)  # namedtuple<doc_id, title, body, marked_up_doc>

    return doc.doc_id, text


collid2irdspath = {
    'robust04': "irds:disks45/nocr/trec-robust-2004",
    'msmarco-passage': '/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/MSMARCO-PASSAGES/collection',
    'trec-covid': 'irds:beir/trec-covid'
}

fields = {
    'robust04': ['title', 'body'],
    'msmarco-passage': ['text'],
    'trec-covid':  ['text', 'title', 'url', 'pubmed_id']
}

import pyterrier as pt
if not pt.started():
    pt.init()

index_path = f"/ssd/data/faggioli/24-ECIR-FF/data/indexes/{args.collection}/pyterrier"


def msmarco_generate():
    msmarcofile = '/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/MSMARCO-PASSAGES/collection/collection.tsv'
    with pt.io.autoopen(msmarcofile, 'rt') as corpusfile:
        for l in corpusfile:
            try:
                docno, passage = l.split("\t")
            except Exception as e:
                print(l)
                pass
            yield {'docno': docno, 'text': passage}

if args.collection in ['robust04', 'trec-covid']:
    dataset = pt.get_dataset(collid2irdspath[args.collection])
    iterator = dataset.get_corpus_iter()

else:
    iterator = msmarco_generate()

indexer = pt.IterDictIndexer(index_path)
indexref = indexer.index(iterator, fields=fields[args.collection])
