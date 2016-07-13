import sys
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import datasets, svm, metrics
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot
from sklearn.cross_validation import KFold
import numpy as np
from pandas_confusion import ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from IPython.display import Image
from sklearn.externals.six import StringIO
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    print "in here"
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,scoring="accuracy")
    print("now here")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    print("boom")
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    print("almost there")
    plt.legend(loc="best")
    return plt

#http://stackoverflow.com/questions/32506951/how-to-explore-a-decision-tree-built-using-scikit-learn

vectorizer = CountVectorizer(min_df=1,stop_words="english")


f = open("full_house_hashtag.txt")

data = f.read()
data = data.split(";")
tweets = []
party = []
names = []

count = 0

for row in data:

    row = row.split(",")
    party.append(row[2])
    tweets.append(row[3])
    names.append(row[1])

print "done parsing file"

print "working on vectorizing"

X = vectorizer.fit_transform(tweets)
Y = np.asarray(party)

title = "Learning Curve for KNN"

#clf =  KNeighborsClassifier()
#clf = svm.SVC(probability=True)
clf = tree.DecisionTreeClassifier(max_depth=3)
cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=100,
                                   test_size=0.2, random_state=0)

plot_learning_curve(clf, title, X, Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
plt.show()





# folds = 10
# area_avg = 0;
# count = 0
# total_time = 0


# kf = KFold(X.shape[0], n_folds=folds, shuffle = True,random_state = 21)
# for train, test in kf:
#     dotfile = "tree" + str(count) +".dot"
#     print "Working on fold # " + str(count)
#     count = count + 1
#     #clf =  KNeighborsClassifier()
#     #clf = svm.SVC(probability=True)
#     clf = tree.DecisionTreeClassifier(max_depth=3)
#     t0 = time.time()

#     clf.fit(vectorizer.fit_transform(X[train]), Y[train])
#     #print(vectorizer.vocabulary_)
#     tree.export_graphviz(clf,feature_names=vectorizer.get_feature_names(),class_names=["D","R"],out_file=dotfile) 

#     expected = Y[test]
#     predicted_probs = clf.predict_proba(vectorizer.transform(X[test]))
#     total_time = total_time + (time.time()-t0)

#     fpr, tpr, thresholds = roc_curve(expected, [item[0] for item in predicted_probs],pos_label='D')

#     roc_auc = auc(fpr, tpr)
#     area_avg = area_avg + roc_auc

#     # plt.plot(fpr, tpr, color='magenta', label='ROC curve (area = %0.2f)' % roc_auc)
#     # plt.plot([0, 1], [0, 1], 'k--')
#     # plt.xlim([0.0, 1.0])
#     # plt.ylim([0.0, 1.05])
#     # plt.xlabel('False Positive Rate')
#     # plt.ylabel('True Positive Rate')
#     # plt.title('Receiver operating characteristic')
#     # plt.legend(loc="lower right")
#     # plt.show()

#     for i in range(0, len(predicted_probs)):
#         predicted_party = "D"
#         if predicted_probs[i][0] > predicted_probs[i][1]:
#             predicted_party = "R"
#         if (expected[i] != predicted_party):
            
#             f = open('myfile','a')
#             f.write(names[test[i]] + "\n") # python will convert \n to os.linesep
#             f.close()
 

#     import numpy as np
#     import matplotlib.pyplot as plt

#     # importances = clf.feature_importances_
#     # std = np.std(clf.feature_importances_,axis=0)
#     # indices = np.argsort(importances)[::-1]

#     # Print the feature ranking
#     # print("Feature ranking:")

#     # for f in range(10):
#     #     print("%d. feature %s (%f)" % (f + 1, vectorizer.get_feature_names()[indices[f]], importances[indices[f]]))

#     # cm = ConfusionMatrix(expected, predicted)
#     # cm.plot()
#     # plt.show()


# area_avg = area_avg / folds
# print "Average area = " + str(area_avg)
# print "Average time = " + str(float(total_time) / folds)
