from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import mixture

from collections import Counter
import numpy as np
import pymc as pm
import time


def print_results(predicted, target, outfile=None, lstart="", lend=""):

  successful_array = [] 
  for j in range(len(predicted)):
    if predicted[j] == target[j]:
      successful_array.append(predicted[j])

  successful = Counter(successful_array)
  retrieved = Counter(predicted)

  pmacro = 0
  rmacro = 0
  #print successful," ",relevant," ",retrieved
  for cat in targset:
    if retrieved[cat]!=0 : pmacro += float(successful[cat])/retrieved[cat]
    if relevant[cat]!=0 : rmacro += float(successful[cat])/relevant[cat]

  pmacro = pmacro/ncat
  rmacro = rmacro/ncat
  f1score = 2*(pmacro*rmacro)/(pmacro+rmacro)

  if outfile == None:
    print i,",",np.mean(predicted == target),",",pmacro,",",rmacro,",",f1score
  else:
    with open(outfile, "a") as myfile:
      myfile.write( str(lstart)+str(np.mean(predicted == target))+","+str(pmacro)+","+str(rmacro)+","+str(f1score)+str(lend) )


# *******************************
#    Data selection 
# *******************************

start = time.time()


categories = ['alt.atheism', 'comp.graphics', 'sci.med'] #'soc.religion.christian', 
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)      

nonlab_train_data = twenty_train.data
N_nonlab = len(nonlab_train_data)
nonlab_train_targets = twenty_train.target
test_data = twenty_test.data
test_targets = twenty_test.target
lab_train_data = []
lab_train_targets =[]
selected_mask = [False]*N_nonlab

N_lab = 0
targset = set(nonlab_train_targets)
ncat = len(targset)
next_tmp = 0

print N_nonlab

print "Data loaded... time: ",(time.time()-start),"s"
start = time.time()

# *******************************
#    Initialization random 
# *******************************

while len(targset) > 0:
  next = int(np.random.random()*N_nonlab)
  target = nonlab_train_targets[next]
  if target in targset:
    targset.remove(target)
    lab_train_data.append(nonlab_train_data[next])
    lab_train_targets.append(nonlab_train_targets[next])
    selected_mask[next] = True
    #np.delete(nonlab_train_data,next)
    #np.delete(nonlab_train_targets,next)
targset = set(nonlab_train_targets)


print "Initialization finished... time: ",(time.time()-start),"s"
start = time.time()


# *******************************
#    Sample selection process of AL  
# *******************************

relevant = Counter(test_targets) 
counts = np.ones(4)
for i in range(500):

  count_vect = CountVectorizer()
  X_train = count_vect.fit_transform(lab_train_data)
  X_test = count_vect.transform(test_data)
  start = time.time()



  # ********************************************************************
  #  Train the Naive Bayes model
  # ********************************************************************
  clf = MultinomialNB().fit(X_train, lab_train_targets)  
  predicted = clf.predict(X_test)
  
  print_results(predicted, test_targets, "outfile.dat", lstart=str(i)+",")

  print " AL Iteration",i,"... NB training time: ",(time.time()-start),"s"
  start = time.time()



  # **************************************************
  #  SVM classification
  # **************************************************
  clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=10, random_state=42).fit(X_train, lab_train_targets)
  #X_new = count_vect.transform(test_data)
  predicted = clf.predict(X_test)
 
  print_results(predicted, test_targets, "outfile.dat", lstart=",", lend="\n")

  print " AL Iteration",i,"... SVM training time: ",(time.time()-start),"s"
  start = time.time()



  # **************************************************
  #  Select new point
  # **************************************************
  #N_nonlab = len(nonlab_train_data)
  next = next_tmp
  lab_train_data.append(nonlab_train_data[next])
  lab_train_targets.append(nonlab_train_targets[next])
  counts[nonlab_train_targets[next]]+=1

  #nonlab_train_data = np.delete(nonlab_train_data, next)
  #nonlab_train_targets = np.delete(nonlab_train_targets, next)
  selected_mask[next] = True
  next_tmp+=1

  print " AL Iteration",i,"... sample selection time: ",(time.time()-start),"s"
  start = time.time()
  for j in range(3):
    print "  ",j," -> ",counts[j]

 


