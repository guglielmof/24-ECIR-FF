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
from .FroebeEtAl2021 import FroebeEtAl2021, get_candidates, cleaning_pipe, retrieve_top_k_docs, get_candidate_queries

class FroebeEtAl2021b(FroebeEtAl2021):
    """
    Fröbe, Maik, Eric Oliver Schmidt, and Matthias Hagen. "Efficient query obfuscation with keyqueries."
    IEEE/WIC/ACM International Conference on Web Intelligence and Intelligent Agent Technology. 2021.
    https://dl.acm.org/doi/pdf/10.1145/3486622.3493950
    """

    def get_filter_list(self, query):
        query = query.lower().split(" ")
        filter_list = query
        morph = [wn.morphy(w) for w in query if wn.morphy(w) is not None]
        filter_list += morph

        #synonyms = [s.name().split(".")[0] for w in morph for s in wn.synsets(w)]
        #sset = [get_proper_synset(w, wn.synsets(w)) for w in morph]
        #hypernyms = [h.name().split(".")[0] for s in sset for h in s.hypernyms()]
        #hyponyms = [h.name().split(".")[0] for s in sset for h in s.hyponyms()]

        #filter_list += synonyms
        stemmer = PorterStemmer()

        filter_list = [stemmer.stem(gensim.parsing.preprocessing.remove_stopwords(w)) for w in filter_list]
        filter_list = set(filter_list)
        # take all the synonyms, hypernyms and hyponyms of the query terms
        return filter_list