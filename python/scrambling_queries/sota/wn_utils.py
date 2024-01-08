from nltk.corpus import wordnet as wn


def get_proper_synset(word, synsets):
    morph = wn.morphy(word)
    if morph is None:
        return word
    for s in synsets:
        if s.name().split(".")[0] == morph:
            return s
    if len(synsets) != 0:
        return synsets[0]
    else:
        return morph
