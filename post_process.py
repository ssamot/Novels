import numpy as np
import pylab as pl
from os.path import isfile, join
from os import listdir
from gutenberg.query import get_metadata
import re

markers=[',', '+', '>', '.', 'o', '*']


def average(arr, n):
     end =  n * int(len(arr)/n)
     return np.mean(arr[:end].reshape(-1, n), 1)



def smooth(x,window_len=201,window='hanning'):

         window_len = np.floor(len(x)/3.0)
         # #print window_len, "window_lean"
         # if(window_len% 2  == 0):
         #     window_len-=1
         # print "orignin", len(x)
         # if( (window_len - len(x)) % 2 == 0):
         #     print "correcting"
         #     window_len+=1
         if x.ndim != 1:
             raise ValueError, "smooth only accepts 1 dimension arrays."

         if x.size < window_len:
             raise ValueError, "Input vector needs to be bigger than window size."


         if window_len<3:
             return x


         if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
             raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


         s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
         #print(len(s))
         if window == 'flat': #moving average
            w=np.ones(window_len,'d')
         else:
             w=eval('np.'+window+'(window_len)')
         y=np.convolve(w/w.sum(),s,mode='valid')
         return y


def get_average(f, fig = False, length = 3.0):
    book = np.loadtxt(f)
    with open(f, 'r') as file:
        first_line = file.readline()
    first_line = first_line[2:].strip()
    headers = first_line.split(", ")
    for i in range(len(headers)):
        headers[i] = headers[i].upper()

    #print headers
    bookT = book.T
    #print bookT.shape
    all_processed = []
    for i, emotion in enumerate(bookT):
        #print emotion
        mv = emotion


        mv = average(mv, np.floor(len(mv)/1000.0))
        #print len(mv)
        #
        # mv =  moving_average(mv,100)
        # # while True:
        # #      mv =  moving_average(mv,10)
        # #      #print mv
        # #      if(len(mv) < 300):
        # #          break
        #
        # mv = average(mv, 10)

        if(len(mv) < length):
            return None
        mv = smooth(mv)
        #mv = savitzky_golay(mv, 711, 3)
        #stride = max( int(len(mv) / 500), 1)
        #print "proced", len(mv)
        #mv2 = average(mv, np.floor(len(mv)/98.0))
        mv = average(mv, np.floor(len(mv)/length))[-length:]
        #print len(mv), len(mv2)
        #print len(mv)
        assert(len(mv) == length)
        all_processed.append(mv)

        if(fig):
            pl.plot(mv, marker = markers[i], linewidth=2, markevery = 2)
            pl.legend(headers)
            pl.xlabel('Fiction portion')
            pl.ylabel('Emotion intensity')
            #pl.title('Emotion intensity graph')
            pl.grid(True)
    #
    if(fig):
        name = f.split("/")[-1][:-4] + "50.eps"
        name = join("./data/", name)
        print name
        pl.savefig(name, bbox_inches='tight')
        pl.cla()
        pl.clf()

    mv = np.array(all_processed).flatten()
    #print mv.shape
    return mv


if __name__ == '__main__':
    #mypath = "./results"
    mypath = "/home/ssamot/projects/github/gutenberg/processed/results/"

    onlyfiles = [ (join(mypath,f), f[:-4], f) for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith(".txt")]



    counts = {}
    input_output = []

    tbrfiles = []
    for files in onlyfiles:
        subject = files[1].split("_")[1]

        fictid = int( files[-1].split("_")[0])

        title = list(get_metadata('title', fictid))[0]
        author = list(get_metadata('author', fictid))


        #matchObj = re.search("London Charivari", title, flags=0)

        if (matchObj is None  and subject !="romance"):
            #print title, author

            if(subject in counts):
                counts[subject]+= 1
            else:
                counts[subject] = 1
            tbrfiles.append(files)


    classes = {}
    tpl = []
    for i,(key,val) in enumerate(counts.iteritems()):
        print key, val
        classes[key] = i
        tpl.append(["textsc{%s}"%key,val,i])

    print classes

    from tabulate import tabulate
    #print tpl
    print tabulate(tpl, headers = ["Class","nInstances", "Class Id"], tablefmt="latex")

    #print onlyfiles
    X = []
    y = []
    print len(tbrfiles)
    for i, file in enumerate(tbrfiles):
        #print file
        subject = file[1].split("_")[1]
        if(i < 5):
            data = get_average(file[0], fig = True)
        else:
            data = get_average(file[0], fig = False)
        if(data is not None):
            X.append(data)
            #print classes[subject]
            y.append(classes[subject])
        #print i
        # if(i == 3):
        #     break



    #print len(X)
    #print X.
    X = np.array(X).T
    y = np.array(y)
    print X.shape, y.shape

    np.savetxt("./data/X50-1.csv", X)
    np.savetxt("./data/y50-1.csv", y)

    # pl.legend(headers)
    # pl.show()
