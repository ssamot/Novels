from nltk.corpus import wordnet as wn
import nltk
from os import listdir
from converter16toCurrent import id2ss
import numpy as np
from os.path import isfile, join




#======================================================================================================


class EmotionDetector():
    def __init__(self):
        self.emotions = self.__load_synsets()

        self.sorted_keys = sorted(self.emotions.keys())
        #print self.sorted_keys




    def __load_synsets(self):
        emotions_type =  self.get_all_emotion_types_from_disk()
        emotions = {}
        for emotion in emotions_type:
            print emotion
            emotions[emotion[1]] = self.load_synsets_from_file(emotion[0])

        return emotions

    def get_all_emotion_types_from_disk(self):
        mypath = "./emotion_lists"
        onlyfiles = [ (join(mypath,f), f[:-9]) for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith(".conv")]
        return onlyfiles

    def load_synsets_from_file(self,emotion_file):
        synsets = {}
        with open(emotion_file, "r") as ins:
            for i, line in enumerate(ins):
                s = id2ss(line.strip(), wn)
                synsets[s] = s

        return synsets

    def __zero_count_emotions(self, emotions):
        zero_counts = {}
        for emotion in emotions:
            zero_counts[emotion] = 0
        #print zero_counts
        return zero_counts

    def __detect_emotion_in_sslist(self, sslist, emotions):
        for ss in sslist:
                    #print ss
                    #print ssid, len(synset2domains)
                    for name, emotion in emotions.iteritems():
                        if ss in emotion: # not all synsets are in WordNet Domain.
                            #print "Detected", ss, name
                            return name

        return None


    def pos_from_raw(self, raw ):
        """ Segments the raw text into sentences, tokenizes them and
            and assigns to each a POS tag
        """
        sentences = nltk.sent_tokenize(raw)
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
        return tagged_sentences


    def detect_emotion_in_raw(self, raw):
        count = self.__zero_count_emotions(self.emotions)
        sentences = self.pos_from_raw( raw )
        for sentence in sentences:
            #print "original sentence: " + str(sentences) + "POS chunks: \n"
            for pos in sentence:
                #print "-"
                #print pos
                sslist = wn.synsets( pos[0] )
                #print sslist
                found_emotion = self.__detect_emotion_in_sslist(sslist,self.emotions)
                #print detect_emotion(sslist,emotions)
                if(found_emotion is not None ):
                    count[found_emotion] = count[found_emotion] + 1.0
        return count


    def detect_emotion_in_raw_np(self, raw):
        counts = self.detect_emotion_in_raw(raw)
        list_count = []
        for e in (self.sorted_keys):
            #print counts
            #print e
            list_count.append(counts[e])

        return np.array(list_count)



    def detect_emotion_in_file(self, file, resultfile):
        allvs = []
        with open(file, "r") as ins:
            data = ins.read().decode('utf8')
        sentences = data.split(".")
        #print len(sentences)
        for s in sentences:
            v = self.detect_emotion_in_raw_np(s)
            allvs.append(v)

        allvs = np.array(allvs)
        #print allvs.shape
        print resultfile
        np.savetxt(resultfile, allvs, header= ", ".join(self.sorted_keys))


if __name__ == '__main__':

    ed = EmotionDetector()


    raw = "Full of unfriendliness and not hate."

    #print ed.detect_emotion_in_raw(raw)
    #print ed.detect_emotion_in_raw_np(raw)
    print ed.detect_emotion_in_file("/home/ssamot/projects/github/gutenberg/processed/texts//0000019086_science fiction.txt","./results/0000019086_science fiction.txt")




