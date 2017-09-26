#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### your code goes here 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

print precision_score(labels_test, pred)
print recall_score(labels_test, pred)

new_pred =  [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
new_label = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0
for i in range(len(new_pred)):
    if new_pred[i] == 1 and new_label[i] == 1:
        true_positive += 1
    elif new_pred[i] == 0 and new_label[i] == 0:
        true_negative += 1
    elif new_pred[i] == 1 and new_label[i] == 0:
        false_positive += 1
    else:
        false_negative += 1

print "true_positive", true_positive
print "true_negative", true_negative
print "false_positive", false_positive
print "false_negative", false_negative
print "precision_score", precision_score(new_label, new_pred)
print "recall_score", recall_score(new_label, new_pred)

