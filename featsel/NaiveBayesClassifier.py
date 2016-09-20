import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import math

def removeNans(array):
  for i in range(0,len(array)): 
    if math.isnan(array[i]):
      array[i] = 0
  return array  

def logsum(array):
  resp = 0
  for i in range(0,len(array)): 
    resp+=math.log(max(array[i],0.001))
  return resp

def dotprodcut(array1, array2):
  resp = np.zeros(len(array1))
  for i in range(0,len(array1)): 
    resp[i]=array1[i]*array2[i]
  return resp

def tobinary(array):
  resp = np.zeros(len(array))
  for i in range(0,len(array)):
    if array[i] > 0: resp[i] = 1
  return resp

class NaiveBayesClassifier:

 def __init__(self):
    params = []   
    self.nterms=0
    self.ndocs=0
    self.ncat=0
    self.probs = []
    self.conditionals = []
    self.logcats = []

 def train(self, X, Y):
  #categories = set(y)
  self.ndocs, self.nterms = X.shape
  self.ncat = len(categories)
  sumscounts = [] 

  countcat = [0 for x in range(self.ncat)]
  for i in Y:
    countcat[i]+=1
 
  sumall = np.zeros(self.nterms)
  for i in range(self.ncat):
    sumscounts.append( np.zeros(self.nterms) )

  self.logcats = [math.log(x/float(self.ndocs)) for x in countcat]
  sumcats = [0 for x in range(self.ncat)]

  start = 0
  for i, end in enumerate(X.indptr[1:]):
    sum_xi=0
    for j, val in zip(X.indices[start:end], X.data[start:end]):
       sumscounts[Y[i]][j] += val
       sum_xi+=val
       sumall[j]+=val
       #print "("+str(i)+","+str(j)+"): "+str(val)
       #arr[i,j] = val
    sumcats[Y[i]]+=sum_xi
    start = end

  
  self.conditionals = []
  for i in range(self.ncat):
     tmp = [x/float(sumcats[i]) for x in sumscounts[i]]
     self.conditionals.append(tmp)
  
  self.probs = []
  for j in range(self.ncat):
    with np.errstate(invalid='ignore'):
      self.probs.append( removeNans(sumscounts[j]/sumall) )
  '''  
  if debug: 
    print "\n**********************************\n  Model parameters acquisition:\n**********************************\n"
    print "Sum counts:"
    for i in range(len(sumscounts)):
      print "  Category "+str(i)+":" 
      print "    "+str(sumscounts[i])
    print "\nAggregate vector: "
    print "    "+str(sumall)
    print "\nProbabilities p(c|term):"
    for i in range(ncat):
      print "  "+str(probs[i])
  '''

 def predict(self, X):
   start = 0
   predictions = []
   for i, end in enumerate(X.indptr[1:]):
     vector = np.zeros(self.nterms)
     for j, val in zip(X.indices[start:end], X.data[start:end]):
       vector[j] = val
       #print "("+str(i)+","+str(j)+"): "+str(val)

     best = -10000000
     bestid = -1
     for j in range(self.ncat):
       prob = logsum( dotprodcut(tobinary(vector), self.conditionals[j]) )
       prob = self.logcats[j] + prob

       if best < prob:
         best = prob
         bestid = j

     predictions.append(j)
     start = end
    
   return predictions   


print "Preparing training and testing fata from 20newsgroups"

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)

train_data = twenty_train.data
train_targets = twenty_train.target
test_data = twenty_test.data
test_targets = twenty_test.target


print "Training the model"

count_vect = CountVectorizer()
X_train = count_vect.fit_transform(train_data)
X_new = count_vect.transform(test_data)

nbc = NaiveBayesClassifier()
nbc.train(X_train, train_targets)

print "Making predictions"

predicted = nbc.predict(X_new)

print "Finish, results:"

print "Accuracy: ",np.mean(predicted == test_targets) 


