from sklearn.feature_extraction.text import CountVectorizer
import itertools
import numpy as np

data = ['She did not cheat on the test, for it was not the right thing to do.','I think I will buy the red car, or I will lease the blue one.','I really want to go to work, but I am too sick to drive.','I am counting my calories, yet I really want dessert.']
count_vect = CountVectorizer()


cx = count_vect.fit_transform(data)

vocab = {}
for key, value in count_vect.vocabulary_.iteritems():
  vocab[value] = key

#print vocab

start = 0
for i, end in enumerate(cx.indptr[1:]):
  for j, val in zip(cx.indices[start:end], cx.data[start:end]):
    print "("+str(i)+","+str(j)+"): "+str(val)+" => '"+vocab[j]+"'"
    #print vocab[j]+" ",
  print ""
  start=end

print count_vect.get_params()
#for i,j,v in zip(cx.row, cx.col, cx.data):
#     print (i,j,v)
