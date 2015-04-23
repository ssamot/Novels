import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import scipy as sp
import scipy.stats
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def main():
    X = np.loadtxt("./data/X1.csv").T
    y = np.loadtxt("./data/y1.csv").T



    # for rm in [3,4]:
    #     X = X[y != rm]
    #     y = y[y != rm]




    #X = X.T

    print X.shape, y.shape
    dc = DummyClassifier(strategy = "most_frequent")
    dc.fit(X,y)


    print "Most Frequent", dc.score(X,y)
    dc = DummyClassifier() ; dc.fit(X,y)
    print "Stratified", dc.score(X,y)
    lr = LogisticRegression(class_weight="auto", fit_intercept=False, penalty="l1")
    clf = make_pipeline(StandardScaler(), lr)
    #clf = RidgeClassifierCV(class_weight="auto", normalize=False)
    # #model =
    #model = clf.fit(X, y)


    from tabulate import tabulate
    #print tpl


    #print confusion_matrix(y,dc.predict(X))
    #exit()



    # check the accuracy on the training set
    #score = model.score(X, y)
    #print score
    #clf = ExtraTreesClassifier(n_estimators=1500, n_jobs = 2, random_state=10, class_weight = "auto")
    #clf = ORC(clf)
    #clf = RandomForestClassifier(n_estimators=500, min_samples_split=100, n_jobs = 2, random_state = 10)
   # clf = DecisionTreeClassifier(max_depth= 20)
    #clf = GradientBoostingClassifier(n_estimators= 30, min_samples_split=5, verbose=True, learning_rate=0.8, subsample=0.5)
    #clf = NuSVC()
    #clf = SVC(class_weight="auto")
    #clf = Pipeline([('anova', preprocessing.StandardScaler()), ('svc', SGDClassifier(shuffle=True, class_weight="auto", n_iter=20, l1_ratio = 1))])
    #clf = Pipeline([('anova', preprocessing.StandardScaler()), ('svc', SVC(kernel="poly", C = 2.0))])
    #clf = Pipeline([('anova', preprocessing.StandardScaler()), ('svc', NuSVC(nu=0.9))])





    ss = StratifiedShuffleSplit(y, n_iter=10, random_state=0)
    scores = []
    cms = []
    for i, (train_index, test_index) in enumerate(ss):
        print "Shuffle %d"%(i,),
        #print("%s %s" % (train_index, test_index))
        clf.fit(X[train_index], y[train_index])
        y_hat = clf.predict(X[test_index])
        #score = clf.score(X[test_index], y[test_index])
        score = accuracy_score(y[test_index], y_hat)
        print score
        cm = confusion_matrix(y[test_index], y_hat)
        #print m.type()
        scores.append(score)
        #print score, np.array(scores).mean()
        cm = np.array(cm,dtype="float")
        cm =  cm/cm.sum(axis = 1)
        #print cm.shape
        #exit()
        #print cm
        cms.append(cm)

    scores = np.array(scores)
    print "ERF", scores.mean()
    print clf.fit(X,y)
    print lr.coef_
    print lr.intercept_
    feature_names = ["--ANGER;".lower(), "--DISGUST;".lower(), "--FEAR;".lower(), "--JOY;".lower(), "--SADNESS;".lower(), "--SURPRISE;".lower()]

    labels = ["--mystery;", "--humor;", "--fantasy;", "--horror;", "--science fiction;", "--western;"]
    l = list(lr.coef_)

    for i in range(0,6):
        l[i] = list(l[i])
        for j in range(0,len(l[i])):
            l[i][j] = "%.3f"%(l[i][j])
        l[i].insert(0,labels[i])
    t =  tabulate(l, headers = feature_names,  tablefmt="latex")
    t = t.replace("--", "\\textsc{")
    t = t.replace(";", "}")
    print t
    exit()

    #print cms
    cms = np.array(cms)
    #print cms.shape
    cms = np.mean(cms, axis = 0)
    #print cms.shape
    plt.matshow(cms)
    #plt.title('Confusion matrix')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



    labels = ["mystery".upper(), "humor".upper(), "fantasy".upper(), "horror".upper(), "science fiction".upper(), "western".upper()]

    plt.xticks(range(6), labels )
    plt.yticks(range(6), labels)
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    plt.savefig("./data/confusion100.eps", transparent=False, bbox_inches='tight')

        #print m


    clf.fit(X,y)
    importances = clf.feature_importances_
    ft = np.array([tree.feature_importances_ for tree in clf.estimators_])
    std = np.std(ft, axis=0)
    print std.shape


    indices = np.argsort(importances)[::-1]

    sem1 =  std / np.sqrt(ft.shape[0])
    cf = sem1*1.96

    print("Feature ranking:")

    for f in range(100):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    #plt.title("Feature importances")
    print importances[indices].shape
    important = 30
    plt.barh(range(important), width = importances[indices][:important][::-1],
           color="c", xerr=cf[indices][:important][::-1], height = 0.5,  align="center", ecolor='r')

    feature_names = ["ANGER", "DISGUST", "FEAR", "JOY", "SADNESS", "SURPRISE"]

    yticks = []
    per_feature = 50
    print std[indices][:important]
    print importances[indices][:important]
    print cf
    for i in indices:
        yticks.append(feature_names[i/per_feature] + " " + str(i%per_feature))
    #yticks[::-1]
    #plt.tick_params(axis='x', labelsize=5)


    plt.yticks(range(important)[::-1], yticks)
    plt.ylim([-1, important])
    plt.savefig("./data/importances100.eps", bbox_inches='tight')




if __name__ == '__main__':
    main()




