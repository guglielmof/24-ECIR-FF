import numpy as np
import numpy.random as npr
from nltk.corpus import wordnet as wn
from .wn_utils import get_proper_synset
from utils.corpora_generators import msmarco_generate
import sys
import os
import pandas as pd
import pyterrier as pt
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.parsing.porter import PorterStemmer
from gensim import similarities
import gensim
import itertools
from tqdm import tqdm
import math
from ir_measures import AP, nDCG, P, Recall, iter_calc
import ir_measures
from unidecode import unidecode
from multiprocessing import Pool
from multiprocessing import get_context


class FroebeEtAl2021:
    """
    Fröbe, Maik, Eric Oliver Schmidt, and Matthias Hagen. "Efficient query obfuscation with keyqueries."
    IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology. 2021.
    https://dl.acm.org/doi/pdf/10.1145/3486622.3493950
    """

    def __init__(self, dataset, collection):
        self.dataset = dataset
        self.collection = collection
        self.get_simulated_corpus()

        # for each candidate document, select the top t terms according to their tf-idf

    def scramble(self, queries):
        print("         * started scrambling")

        args = [[q, self.sampled_docs, self.get_filter_list(q), self.dct, self.index, self.model] for q in queries['query'].to_list()]
        with get_context("spawn").Pool(processes=70) as pool:
            futureCandidates = [pool.apply_async(get_candidates, a) for a in args]
            temp_Candidates = [fr.get() for fr in futureCandidates]
            # temp_Candidates = pool.map(get_candidates, args)

            pool.terminate()
            # pool.join()

        queries['candidate'] = temp_Candidates

        queries = queries.explode("candidate")
        queries[['candidate', 'similarity']] = queries['candidate'].apply(pd.Series)
        return queries[['qid', 'query', 'candidate', 'similarity']]

    def get_filter_list(self, query):
        query = query.lower().split(" ")
        filter_list = query
        morph = [wn.morphy(w) for w in query if wn.morphy(w) is not None]
        sset = [get_proper_synset(w, wn.synsets(w)) for w in morph]
        filter_list += [wn.morphy(w) for w in query if wn.morphy(w) is not None]

        synonyms = [s.name().split(".")[0] for w in morph for s in wn.synsets(w)]
        hypernyms = [h.name().split(".")[0] for s in sset for h in s.hypernyms()]
        hyponyms = [h.name().split(".")[0] for s in sset for h in s.hyponyms()]

        filter_list += synonyms + hyponyms + hypernyms
        stemmer = PorterStemmer()

        filter_list = [stemmer.stem(gensim.parsing.preprocessing.remove_stopwords(w)) for w in filter_list]
        filter_list = set(filter_list)
        # take all the synonyms, hypernyms and hyponyms of the query terms
        return filter_list

    def get_simulated_corpus(self, n=50000):

        punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
        translator = str.maketrans(punctuation, ' ' * len(punctuation))

        def robust_generate():
            if not pt.started():
                pt.init()

            dataset = pt.get_dataset("irds:disks45/nocr/trec-robust-2004")
            iterator = dataset.get_corpus_iter()

            def preprocess_document(doc):
                title = doc['title'].lower().replace("\n", " ").replace("-", "").translate(translator)
                body = doc['body'].lower().replace("\n", " ").replace("-", "").translate(translator)
                text = (title + " " + body)  # namedtuple<doc_id, title, body, marked_up_doc>

                return text

            for doc in iterator:
                yield preprocess_document(doc)

        def msmarco_generate():
            msmarcofile = '/ssd/data/faggioli/EXPERIMENTAL_COLLECTIONS/CORPORA/MSMARCO-PASSAGES/collection/collection.tsv'
            with pt.io.autoopen(msmarcofile, 'rt') as corpusfile:
                for l in corpusfile:
                    try:
                        docno, passage = l.split("\t")
                    except Exception as e:
                        print(l)
                        pass
                    yield passage.lower().replace("\n", " ").replace("-", "").translate(translator)

        # collections are inverted because we do not want to use the operative collection as a reference corpus
        generators = {'msmarco-passage': robust_generate,
                      'robust04': msmarco_generate}

        ndocs = {'msmarco-passage': 528155,
                 'robust04': 8841823}

        sampled_docs_path = f"../../data/local_corpora/{self.collection}_{n}.csv"
        if os.path.exists(sampled_docs_path):
            sampled_docs = pd.read_csv(sampled_docs_path, header=None, names=["did", "doc"])
        else:
            generator = generators[self.collection]()
            sampled_docs = set(npr.choice(np.arange(ndocs[self.collection]), size=n, replace=False))
            sampled_docs = [i for e, i in enumerate(generator) if e in sampled_docs]

            sampled_docs = zip(np.arange(len(sampled_docs)), sampled_docs)
            sampled_docs = pd.DataFrame(sampled_docs, columns=["did", "doc"])
            sampled_docs.to_csv(sampled_docs_path, header=False, index=False)

        sampled_docs['doc_cleaned'] = sampled_docs['doc'].apply(cleaning_pipe)

        self.sampled_docs = sampled_docs
        self.dct = Dictionary([d for d in sampled_docs['doc_cleaned'].to_list()])

        self.dct.filter_extremes(no_below=2, no_above=0.9)

        self.corpus = [self.dct.doc2bow(d) for d in sampled_docs['doc_cleaned'].to_list()]  # convert corpus to BoW format
        self.model = TfidfModel(self.corpus, id2word=self.dct)  # fit model
        self.index = similarities.SparseMatrixSimilarity(self.model[self.corpus], num_features=len(self.dct))
        '''
        self.stem2word = {}
        for d in sampled_docs.doc.values:
            for w in d.split():
                if stemmer.stem(w) not in self.stem2word:
                    self.stem2word[stemmer.stem(w)] = set()
                self.stem2word[stemmer.stem(w)].add(w)
        '''


def get_candidates(qry, docs, filter, dct, index, model):
    top_retrieved = retrieve_top_k_docs(qry, dct, index, model)
    top_c = get_candidate_queries(docs, top_retrieved, filter, dct, index, model)
    print(f"query: {qry} done with {len(top_c)} candidates")
    return top_c


def cleaning_pipe(string):
    stemmer = PorterStemmer()

    punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    translator = str.maketrans(punctuation, ' ' * len(punctuation))
    string = string.translate(translator)

    tokens = stemmer.stem_sentence(gensim.parsing.preprocessing.remove_stopwords(string)).split()
    tokens = [unidecode(t) for t in tokens if not t.isnumeric() and len(t) >= 2]
    return tokens


def retrieve_top_k_docs(query, dct, index, model, k=10, scores=False):
    """get the top matching docs as per cosine similarity
    between tfidf vector of query and all docs"""
    query_document = cleaning_pipe(query)
    query_bow = dct.doc2bow(query_document)
    sims = index[model[query_bow]]
    if k != -1:
        top_idx = sims.argsort()[-1 * k:][::-1]
        top_idx = [t for t in top_idx if sims[t] > 0]
    else:
        top_idx = sims.argsort()[::-1]
        top_idx = [t for t in top_idx if sims[t] > 0]

    if scores:
        scores = [sims[i] for i in top_idx]
        out = list(zip(top_idx, scores))

    else:
        out = top_idx

    return out


def get_candidate_queries(sampled_docs, docids, filter_list, dct, index, model, c=3, winsize=16, l=100):
    stemmer = PorterStemmer()

    cands = []
    for d in docids:
        doc = sampled_docs.doc.values[d]

        punctuation = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
        translator = str.maketrans(punctuation, ' ' * len(punctuation))
        doc = doc.translate(translator)

        # remove stopwords
        nostwd = gensim.parsing.preprocessing.remove_stopwords(doc)
        # remove forbidden words

        noforb = [w for w in nostwd.split() if stemmer.stem(w) not in filter_list]
        # remove numbers
        nonumb = [unidecode(w) for w in noforb if not w.isnumeric() and len(w) >= 3]
        windows = [nonumb[winsize * i:winsize * (i + 1)] for i in range(math.ceil(len(nonumb) / winsize))]
        for w in windows:
            for L in range(1, c):
                for subset in itertools.combinations(w, L):
                    cands.append(tuple(sorted(subset)))

    qrels = pd.DataFrame({'query_id': ['Q0'] * len(docids), 'doc_id': [str(idx) for idx in docids], 'relevance': [1] * len(docids)})
    already_used = set()

    cands_and_scores = []
    for cand in cands:
        q = " ".join(cand)
        if q not in already_used:
            top_retrieved = retrieve_top_k_docs(q, dct, index, model, k=-1, scores=True)
            if len(top_retrieved) > l:
                run = pd.DataFrame(
                    {'query_id': ['Q0'] * len(top_retrieved), 'doc_id': [str(t[0]) for t in top_retrieved],
                     'score': [t[1] for t in top_retrieved]})
                cands_and_scores.append((q, ir_measures.calc_aggregate([nDCG], qrels, run)[nDCG]))
            already_used.add(q)
    return cands_and_scores
