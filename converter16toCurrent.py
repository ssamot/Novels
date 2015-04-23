from nltk.corpus import wordnet as wn
from nltk.corpus import WordNetCorpusReader, LazyCorpusLoader, CorpusReader
from os import listdir



def get_16_emotions():
    mypath = "./emotion_lists"
    from os.path import isfile, join
    onlyfiles = [ (join(mypath,f), f[:-4]) for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith(".txt")]
    return onlyfiles



def convert(emotion_file):
    synsets = {}
    WN16 = LazyCorpusLoader(
    'wordnet-1.6/dict', WordNetCorpusReader,
    LazyCorpusLoader('omw', CorpusReader, r'.*/wn-data-.*\.tab', encoding='utf8'))

    f_conv = open(emotion_file + ".conv",'w')
    with open(emotion_file, "r") as ins:
        for i, line in enumerate(ins):

            id = line.split(" ")[0]
            ssid = id[2:] + "-" + id[0]
            synset_16 = id2ss(ssid,WN16)
            #print i,
            synset_30 = _wn30_synsets_from_wn16_synset(synset_16)
            if(synset_30 is not None):
                id = str(synset_30.offset()).zfill(8) + "-" + synset_30.pos()
                if(id not in synsets):
                    #print synset_16, synset_30
                    synsets[id] = synset_30
                    f_conv.write(id + "\n")
            else:
                print "broken", synset_16

    f_conv.close()


def id2ss(ID,mywn):
    """Given a Synset ID (e.g. 01234567-n) return a synset"""
    #print int(ID[:8])
    #print str(ID[-1:])
    return mywn._synset_from_pos_and_offset(str(ID[-1:]), int(ID[:8]))


def _wn30_synsets_from_wn16_synset(synset):
    (word, p, index) = synset.name().split(".")
    # ADJ_SAT -> ADJ: DO NOT EXIST ADJ_SAT in wordnet.POS_LIST
    if p == 's': p = 'a'
    synsets = wn.synsets(word, p)
    if len(synsets) == 0: return

    synset_sims = {}
    for i in range(len(synsets)):
        try:
            synset_sims[i] = synset.wup_similarity(synsets[i])
        except (RuntimeError, TypeError, NameError):
            # Set similarity to 0 in case of RuntimeError
            synset_sims[i] = 0
    # Most similar synset index
    index = sorted(synset_sims.items(), key=lambda x:x[1], reverse=True)[0][0]

    return synsets[index]
