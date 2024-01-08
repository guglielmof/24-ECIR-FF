import nltk
from nltk.corpus import wordnet as wn
from pattern.text.en import singularize
import itertools
import numpy as np
from .wn_utils import get_proper_synset


class ArampatzisEtAl2011:
    """

    Arampatzis, Avi, Pavlos Efraimidis, and George Drosatos. "Enhancing deniability against query-logs."
    Advances in Information Retrieval: 33rd European Conference on IR Research, ECIR 2011, Dublin,
    Ireland, April 18-21, 2011. Proceedings 33. Springer Berlin Heidelberg, 2011.

    http://euclid.ee.duth.gr/wp-content/uploads/2012/09/QueryScrambler.pdf
    """

    def __init__(self, **kwargs):
        nltk.download('averaged_perceptron_tagger')

    def scramble(self, queries):
        local_queries = queries.copy()
        local_queries['processed_query'] = queries['query'].apply(process_string)

        local_queries['processed_query'] = local_queries['processed_query'].apply(get_synset)
        local_queries['expanded_query'] = local_queries['processed_query'].apply(expand_synset)

        local_queries = local_queries.explode("expanded_query")

        local_queries['similarity'] = local_queries.apply(compute_similarity, axis=1)
        local_queries["candidate"] = local_queries["expanded_query"].apply(revert_synsets)

        queries = local_queries[["qid", "query", "candidate", "similarity"]]

        '''
        local_queries['processed_query'] = local_queries['processed_query'].apply(lambda x: (t[0] for t in x))
        local_queries['perturbed_queries'] = local_queries.apply(lambda x: compute_similarity(x['processed_query'], x['perturbed_queries']), axis=1)
        # nouns --> hypernyms and holonyms
        # verbs --> hypernyms
        print(local_queries)
        '''

        return queries


def revert_synsets(synsets_list):
    out = []
    for w in synsets_list:
        if w.__class__.__name__ == "Synset":
            out.append(w.name().split(".")[0])
        else:
            out.append(w[0])

    return " ".join(out)


def get_synset(words_list):
    processed = []
    for w in words_list:
        if 'VB' in w[1]:
            synsets = wn.synsets(w[0], wn.VERB)
            s = get_proper_synset(w[0], synsets)
            processed.append(s)
        elif 'NN' in w[1]:
            synsets = wn.synsets(w[0], wn.NOUN)
            s = get_proper_synset(w[0], synsets)
            processed.append(s)

        else:
            processed.append(w)
    return processed


def process_string(s):
    tokens = nltk.word_tokenize(s.lower())
    tags = nltk.pos_tag(tokens)
    # remove plurals
    tags = [t if t[1] != "NNS" else ((wn.morphy(t[0]) if wn.morphy(t[0]) is not None else t[0]), "NN") for t in tags]

    return tags


def expand_synset(query, min_n=300):
    wordsets = get_wordsets(query, 2)
    n_obf_2 = 1
    for s in wordsets:
        n_obf_2 *= len(s)
    if n_obf_2 < min_n:
        wordsets = get_wordsets(query, 3)

    n_obf_3 = 1
    for s in wordsets:
        n_obf_3 *= len(s)

    queries = list(itertools.product(*wordsets))
    return queries


def get_wordsets(l, level):
    out = []
    for word in l:
        obfuscation_set = [word]
        if word.__class__.__name__ == "Synset":
            try:
                if word.name().split(".")[1] == "v":
                    obfuscation_set = recursive_get_hypernyms(word, level)
                elif word.name().split(".")[1] == "n":
                    obfuscation_set = recursive_get_hypernyms(word, level) + recursive_get_holonyms(word, level)
            except Exception as e:
                pass

        out.append(list(set(obfuscation_set)))

    return out


def compute_similarity(transformed_q):
    orig = transformed_q["processed_query"]
    nois = transformed_q["expanded_query"]
    partials = []
    for worig in orig:
        if worig.__class__.__name__ == "Synset":
            temp = []
            for wnois in nois:
                if wnois.__class__.__name__ == "Synset":
                    sim = worig.wup_similarity(wnois)
                    if sim is None or sim == -1:
                        sim = 0
                    temp.append(sim)

            partials.append(np.max(temp))
        else:
            partials.append(1)

    return np.mean(partials)


def recursive_get_hypernyms(word, levels):
    out = [word]
    if levels == 0:
        return out
    else:
        try:
            hp = word.hypernyms()
            for hpw in hp:
                hypernims = recursive_get_hypernyms(hpw, levels - 1)
                out += hypernims
            return out
        except Exception as e:
            return out


def recursive_get_holonyms(word, levels):
    out = [word]
    if levels == 0:
        return out
    else:
        try:
            hp = word.holonyms()
            for hpw in hp:
                hypernims = recursive_get_holonyms(hpw, levels - 1)
                out += hypernims
            return out
        except Exception as e:
            return out
