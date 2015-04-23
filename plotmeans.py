import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import scipy as sp
import scipy.stats

markers=[',', '+', '>', '.', 'o', '*', "1"]
emotions = ["ANGER", "DISGUST", "FEAR", "JOY", "SADNESS", "SURPRISE"]


tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

def main():
    labels1 = ["mystery".upper(), "humor".upper(), "fantasy".upper(), "horror".upper(),  "science fiction".upper(), "western".upper()]
    #labels = ["mystery".upper(), "humor".upper(), "fantasy".upper(), "horror".upper(), "romance".upper(), "science fiction".upper(), "western".upper()]
    X = np.loadtxt("./data/X100.csv").T
    y = np.loadtxt("./data/y100.csv").T

    t_emotions = 50
    fig = plt.figure()
    j = 0
    for k,(emo) in enumerate(emotions):
        ax = fig.add_subplot(3,2,k+1)
        lines = []
        for i,label in enumerate(labels1):
            X_l = X[y == i]
            X_l = X_l.T
            n = X_l.shape[1]
            #print "n", n
            x_l = X_l.mean(1).flatten()[j:j+t_emotions]
            print label
            if(label != "romance".upper()):
                #x_l = np.zeros(x_l.shape)
            #     print "FOUFOUFOU"
            #     continue
                std = X_l.std(1).flatten()[j:j+t_emotions]
                #print std.shape
                sem1 =  std / np.sqrt(n)
                cf = sem1
                line = ax.plot(x_l,marker = markers[i], linewidth=2, markevery = 2, )

                lines.append(line[0])
            else:
                print "This is impossible"
                exit()

        break
                #print line[0]
                #ax.fill_between(list(range(0,t_emotions)), x_l+cf, x_l-cf, alpha=0.5, color = line[0].get_color())
        #ax.set_title(label)
        #ax.set_xlabel('Fiction portion')
        ax.set_ylabel(emo + ' intensity')
        j+=t_emotions
        print k, emo
    art = []
    plt.figlegend(lines, labels1, 9, ncol=3)
   # p_art = plt.legend(loc=9,bbox_to_anchor=(0.5, 1.0),  ncol=2)
    #art.append(p_art)

    #ax.set_xlabel('Fiction Portion')
    #ax.xaxis.set_label_position('bottom')
    #plt.xlabel("Fiction Portion")

    #fig.suptitle("Title centered above all subplots", fontsize=14, position = "")

        #ax.xlabel("Fiction Portion")




    # for i,label in enumerate(labels):
    #     ax = fig.add_subplot(4,2,i)
    #
    #
    #     #y_l = y[y == i]
    #     n = X_l.shape[1]
    #
    #     print x_l.shape, n
    #     std = X_l.std(1).flatten()
    #     sem1 =  std / np.sqrt(n)
    #     cf_j = sem1*1.96
    #     j = 0
    #
    #     #cf = cf*100
    #
    #
    #     #y = np.linspace(0,1,50)
    #
    #     #print y.shape
    #     #print (x+cf).shape
    #
    #         j = j+50
    #

    plt.show()
    plt.savefig("./data/allemotions50.eps", additional_artists=art, transparent=False, bbox_inches='tight')
    #plt.legend(headers)





    # plt.savefig("./data/means.pdf", transparent=False, bbox_inches='tight')
    #
    #     #print m
    #
    #
    # clf.fit(X,y)
    # importances = clf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in clf.estimators_],
    #          axis=0)
    # indices = np.argsort(importances)[::-1]
    #
    # sem1 =  std / np.sqrt(len(std))
    # cf = sem1*1.96
    #
    # print("Feature ranking:")
    #
    # for f in range(100):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    #
    # # Plot the feature importances of the forest
    # plt.figure()
    # #plt.title("Feature importances")
    # print importances[indices].shape
    # important = 20
    # plt.barh(range(important), width = importances[indices][:important][::-1],
    #        color="c", xerr=cf[indices][:important][::-1], height = 0.5,  align="center", ecolor='r')
    #
    # feature_names = ["ANGER", "DISGUST", "FEAR", "JOY", "SADNESS", "SURPRISE"]
    #
    # yticks = []
    # per_feature = 50
    # print std[indices][:important]
    # print importances[indices][:important]
    # print cf
    # for i in indices:
    #     yticks.append(feature_names[i/per_feature] + " " + str(i%per_feature))
    # #yticks[::-1]
    # #plt.tick_params(axis='x', labelsize=5)
    #
    #
    # plt.yticks(range(important)[::-1], yticks)
    # plt.ylim([-1, important])
    # plt.savefig("./data/importances.eps", bbox_inches='tight')
    #



if __name__ == '__main__':
    main()




