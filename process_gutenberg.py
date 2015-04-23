from bs4 import BeautifulSoup


from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers

from os.path import basename


i_subjects = ["science fiction", "horror", "western", "fantasy", "crime fiction", "mystery", "humor", "romance"]

from os import walk

f = []

gutenberg_path = "/home/ssamot/projects/github/gutenberg"
total = 0
for (dirpath, dirnames, filenames) in walk(gutenberg_path):
    for filename in filenames:
        f =  "/".join([dirpath, filename])
        total+=1

print total
i = 0
for (dirpath, dirnames, filenames) in walk(gutenberg_path):
    for filename in filenames:
        f =  "/".join([dirpath, filename])
        if(f.endswith(".rdf")):
            #print f
            i+=1
            bf = BeautifulSoup(open(f))
            subjects =  bf.find_all("dcterms:subject")
            if (subjects is not None and len(subjects) > 0):
                for subject in subjects:
                    val =  subject.find_all("rdf:value")[0].contents[0]
                    for i_subject in i_subjects:
                        if(i_subject in val.lower()):
                            #print f, val

                            id =  int(basename(f)[2:-4])
                            fn = str(id).zfill(10) + "_" +  i_subject + ".txt"
                            print fn
                            try:
                                text = strip_headers(load_etext(id)).strip().encode("utf-8")
                                wf = "./texts/" + fn
                                with open(wf, "w") as text_file:
                                    text_file.write(text)
                                print i, total, float(i)/total
                            except:
                                print "broken", id
            # for network in tree.findtext('dcterms subject'):
            #     print network
