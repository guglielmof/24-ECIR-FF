

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
