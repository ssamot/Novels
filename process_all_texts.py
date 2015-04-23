from os.path import isfile, join
from os import listdir
from wordaffect import EmotionDetector
from multiprocessing import Pool

def process_single_file(vars):
    inputfile, outputfile = vars
    print inputfile
    ed = EmotionDetector()
    ed.detect_emotion_in_file(inputfile, outputfile)



if __name__ == '__main__':
    #mypath = "/home/ssamot/projects/github/gutenberg/processed/texts"
    #mypath = "/scratch/ssamot/texts/"
    mypath = "./texts"
    #outpath = "/home/ssamot/projects/github/gutenberg/processed/results"
    outpath = "./results"
    onlyfiles = [ (join(mypath,f), f[:-4], f) for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith(".txt")]

    counts = {}
    input_output = []
    for files in onlyfiles:
        subject = files[1].split("_")[1]
        if(subject in counts):
            counts[subject]+= 1
        else:
            counts[subject] = 1
        input_output.append((files[0], join(outpath,files[2])))

    for key,val in counts.iteritems():
        print key, val

    p = Pool(200)
    print len(input_output)
    print input_output[0]

    #map(process_single_file, input_output)
    p.map(process_single_file, input_output)



