from sklearn.naive_bayes import *
import numpy as np

#print sklearn.__version__

X = np.array([ [1, 1, 1, 1, 1], 
               [0, 0, 0, 0, 0] ])
print "X: ", X
Y = np.array([ 1, 2 ])
print "Y: ", Y

clf = MultinomialNB()
clf.fit(X, Y)
print "Prediction:", clf.predict( [0, 0, 0, 0, 0] )  

print clf.predict_proba([0, 1, 0, 1, 1] )
