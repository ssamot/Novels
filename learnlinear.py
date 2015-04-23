import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import scipy as sp
import scipy.stats
from sklearn.multiclass import OneVsRestClassifier as ORC


def main():
    X = np.loadtxt("./data/X50.csv").T
    y = np.loadtxt("./data/y50.csv").T



    # for rm in [3,4]:
    #     X = X[y != rm]
    #     y = y[y != rm]

    newX = []
    for x in X:



    #X = X.T

    print X.shape, y.shape
    dc = DummyClassifier(strategy = "most_frequent")
    dc.fit(X,y)


    print "Most Frequent", dc.score(X,y)
    dc = DummyClassifier() ; dc.fit(X,y)
    print "Stratified", dc.score(X,y)
    #clf = LogisticRegression(class_weight="auto")
    #clf = RidgeClassifierCV(class_weight="auto", normalize=False)
    # #model =
    #model = clf.fit(X, y)


    #print confusion_matrix(y,dc.predict(X))
    #exit()



    # check the accuracy on the training set
    #score = model.score(X, y)
    #print score
    clf = ExtraTreesClassifier(n_estimators=1500, n_jobs = 2, random_state=10, class_weight = "auto")
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




if __name__ == '__main__':
    main()




