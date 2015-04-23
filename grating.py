import urllib2
import numpy as np
import json
from os import listdir
from os.path import isfile, join
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata
import time

def get_rating(name):
    name = urllib2.quote(name)
    url = "https://www.googleapis.com/books/v1/volumes?q=<>&fields=items%2FvolumeInfo%2FaverageRating&key=AIzaSyDoq4RB-h13qZyy1tJLaJymxfmppdO40wc".replace("<>",name)
    #url = "https://www.googleapis.com/books/v1/volumes?q=<>&fields=items%2FvolumeInfo%2FaverageRating?key=AIzaSyDoq4RB-h13qZyy1tJLaJymxfmppdO40wc".replace("<>",name)
    print url
    response = urllib2.urlopen(url)
    return response


def get_mean(search_string):
    try:
        scores =  get_rating(search_string)
        scores = json.load(scores)
        values = []
        for score in scores["items"]:
             value = score["volumeInfo"]["averageRating"]
             values.append(value)

        values = np.array(values)
        time.sleep(1)
        return values.mean(), len(values)
    except urllib2.HTTPError:
        print "Recursing"
        time.sleep(60)
        return get_mean(search_string)




if __name__ == '__main__':
    mypath = "/home/ssamot/projects/github/gutenberg/processed/results/"

    onlyfiles = [ (join(mypath,f), f[:-4], f) for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith(".txt")]
    means = np.loadtxt("./data/ytotals.csv")
    means = list(means)
    totals = []
    print means

    starting_point = len(means)
    print starting_point
    for i, file in enumerate(onlyfiles):
        if(i> starting_point ):
            print i, starting_point
            fictid = int( file[-1].split("_")[0])
            title = list(get_metadata('title', fictid))
            author = list(get_metadata('author', fictid))
            print fictid, title, author
            if(author == []):
                author.append("")
            #print get_metadata("author", fictid)
            try:
                mean, total =  get_mean(title[0] + " " + author[0])
                #time.sleep(2.0)
                means.append(mean)
                totals.append(total)
            except KeyError:
                means.append(-1)
            #print means
            #print  np.array(means)

            np.savetxt("./data/yraitings2.csv", np.array(means))
            np.savetxt("./data/ytotals2.csv", np.array(totals))

